/* computes integrated brightness of all visible and illuminated areas
   and its derivatives

   8.11.2006

   2026: rewritten as a warp-cooperative kernel for memory locality.

   Key ideas (measured on a Tesla V100, see the pull request for numbers):

   * The per-block matrix Dg is rank-1 redundant: curv() computes
	 Dg[i + k*Numfac1] = g_i * CUDA_Dsph[i][k], and bright consumed it as
	 dbr_i * Dg[..] with dbr_i = Darea_i * s. Since Area_i = Darea_i * g_i,
	 the g factor folds into the weight:

	 dbr_i * Dg[i][k] == (Area_i * s) * CUDA_Dsph[i][k]

	 so every block gathers from the ONE global, facet-major, read-only
	 CUDA_Dsph matrix (cache-resident for all blocks) instead of a private
	 ~116 KB Dg that thrashes L1/L2 with 8-byte scattered reads.

   * One block of CUDA_BLOCK_DIM threads processes one (frequency, pole)
	 context, one data point per thread (ported from the OpenCL build, which
	 ran ~2x faster on the same GPU than the former one-warp-per-context,
	 two-points-at-a-time version): see bright_curve1() below.

   * dytemp is stored transposed - dytempT[(jp-1)*DYT_STRIDE + l] - so the
	 derivative writes here and the tile reads in MrqcofCurve2 are coalesced.
	 This requires ma <= DYT_STRIDE-1 = 63 (any spherical-harmonics degree
	 up to 6, i.e. every production workunit); the host enforces it.

   * The per-point geometry (the former matrix_neo pass) is computed
	 in-kernel, by the thread that owns the point: the de, de0, e_1..e0_3 and
	 jp_Scale/jp_dphp per-point global buffers are gone entirely.
*/

#include <cmath>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <device_launch_parameters.h>

/* per-point geometry, replaces matrix_neo: everything bright needs about a
   single data point, written to po[GEOM_PT_SIZE] (shared memory).
   layout: 0..15 the 16 nonzero de/de0 sums' factors
		   (gde{e,e0}{col1,col2,col3} in the naming below),
		   16..21 e_1..e_3, e0_1..e0_3, 22 Scale, 23 dphp_1, 24 dphp_2,
		   25 dphp_3 (= alpha)
   The rotation math is the same as matrix_neo's Blmat/Dblm matrix products,
   just with the zero entries folded away; inv[] carries the four nonzero
   primitives taken from Blmat after blmatrix(). */


__device__ void __forceinline__ bright_point_geometry(int lnp,
	double const* __restrict__ inv,
	double* __restrict__ po)
{
	double ee_1 = CUDA_ee[lnp * 3 + 0];
	double ee0_1 = CUDA_ee0[lnp * 3 + 0];
	double ee_2 = CUDA_ee[lnp * 3 + 1];
	double ee0_2 = CUDA_ee0[lnp * 3 + 1];
	double ee_3 = CUDA_ee[lnp * 3 + 2];
	double ee0_3 = CUDA_ee0[lnp * 3 + 2];
	double t = CUDA_tim[lnp];

	/* ee and ee0 are unit vectors: their dot is mathematically in [-1,1],
	   but opposition geometry brings it within ~1e-7 of 1.0 and an
	   out-of-range rounding would poison every frequency with NaN */
	double alpha = acos(fmin(1.0, fmax(-1.0, ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3)));

        /* ee and ee0 are unit vectors, so the dot product is mathematically in
           [-1, 1] - but for observations near opposition (solar phase ~ 0) it
           lands within ~1e-7 of 1.0, and a different (equally legal) FMA
           contraction produced by another compiler/architecture can round it
           just past 1.0. acos would then return NaN, and a single NaN data
           point poisons the chi-square of every trial frequency. fmin/fmax
           pass in-range values through unchanged, so results on healthy
           inputs are bit-identical. */
        alpha = acos(fmin(1.0, fmax(-1.0, ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3)));
	/* Exp-lin model (const.term=1.) */
	double f = exp(-alpha / inv[2]);
	po[22] = 1 + inv[1] * f + inv[3] * alpha;   /* Scale */
	po[23] = f;                                 /* dphp_1 */
	po[24] = inv[1] * f * alpha / (inv[2] * inv[2]); /* dphp_2 */
	po[25] = alpha;                             /* dphp_3 */

	f = inv[0] * t + CUDA_Phi_0;
	f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
	double sf, cf;
	sincos(f, &sf, &cf);

	/* the four nonzero Blmat primitives (set by blmatrix):
	   inv[7] = Blmat[1][3] = -sin(beta)   inv[8]  = Blmat[3][3] = cos(beta)
	   inv[9] = Blmat[2][1] = -sin(lambda) inv[10] = Blmat[2][2] = cos(lambda) */
	double Blmat02 = inv[7], Blmat22 = inv[8], Blmat10 = inv[9], Blmat11 = inv[10];
	double Blmat00 = Blmat11 * Blmat22;
	double Blmat01 = Blmat22 * -Blmat10;
	double msf = -sf;
	double cbl00 = cf * Blmat00;
	double sbl10 = sf * Blmat10;
	double cbl10 = cf * Blmat10;
	double sbl11 = sf * Blmat11;
	double cbl11 = cf * Blmat11;
	double cbl01 = cf * Blmat01;
	double sbl00 = msf * Blmat00;
	double sbl01 = msf * Blmat01;

	double gde020 = Blmat00 * ee_1 + Blmat01 * ee_2 + Blmat02 * ee_3;
	double gde120 = Blmat00 * ee0_1 + Blmat01 * ee0_2 + Blmat02 * ee0_3;

	double tmat41 = -cbl01 - sbl11;
	double tmat51 = -sbl01 - cbl11;
	double tmat42 = cbl00 + sbl10;
	double tmat52 = sbl00 + cbl10;

	double gde001 = tmat41 * ee_1 + tmat42 * ee_2;
	double gde101 = tmat41 * ee0_1 + tmat42 * ee0_2;
	double gde011 = tmat51 * ee_1 + tmat52 * ee_2;
	double gde111 = tmat51 * ee0_1 + tmat52 * ee0_2;

	double tmat01 = cbl00 + sbl10;
	double tmat11 = sbl00 + cbl10;
	double tmat02 = cbl01 + sbl11;
	double tmat12 = sbl01 + cbl11;
	double tmat03 = cf * Blmat02;
	double tmat13 = msf * Blmat02;

	double ge00 = tmat01 * ee_1 + tmat02 * ee_2 + tmat03 * ee_3;
	double ge10 = tmat01 * ee0_1 + tmat02 * ee0_2 + tmat03 * ee0_3;
	double ge01 = tmat11 * ee_1 + tmat12 * ee_2 + tmat13 * ee_3;
	double ge11 = tmat11 * ee0_1 + tmat12 * ee0_2 + tmat13 * ee0_3;

	double Blmat20 = Blmat11 * -Blmat02;
	double Blmat21 = Blmat02 * Blmat10;
	double gde002 = t * ge01;
	double gde102 = t * ge11;
	double gde012 = -t * ge00;
	double gde112 = -t * ge10;

	double ge02 = Blmat20 * ee_1 + Blmat21 * ee_2 + Blmat22 * ee_3;
	double ge12 = Blmat20 * ee0_1 + Blmat21 * ee0_2 + Blmat22 * ee0_3;
	double gde021 = -Blmat21 * ee_1 + Blmat20 * ee_2;
	double gde121 = -Blmat21 * ee0_1 + Blmat20 * ee0_2;

	double tmat31 = sf * Blmat20;
	double tmat32 = sf * Blmat21;
	double tmat33 = sf * Blmat22;
	double tmat21 = cf * -Blmat20;
	double tmat22 = cf * -Blmat21;
	double tmat23 = cf * -Blmat22;

	double gde000 = tmat21 * ee_1 + tmat22 * ee_2 + tmat23 * ee_3;
	double gde100 = tmat21 * ee0_1 + tmat22 * ee0_2 + tmat23 * ee0_3;
	double gde010 = tmat31 * ee_1 + tmat32 * ee_2 + tmat33 * ee_3;
	double gde110 = tmat31 * ee0_1 + tmat32 * ee0_2 + tmat33 * ee0_3;

	po[0] = gde000;  po[1] = gde010;  po[2] = gde020;
	po[3] = gde100;  po[4] = gde110;  po[5] = gde120;
	po[6] = gde001;  po[7] = gde011;  po[8] = gde021;
	po[9] = gde101;  po[10] = gde111; po[11] = gde121;
	po[12] = gde002; po[13] = gde012;
	po[14] = gde102; po[15] = gde112;
	po[16] = ge00;   po[17] = ge01;   po[18] = ge02;
	po[19] = ge10;   po[20] = ge11;   po[21] = ge12;
}

/* the whole former matrix_neo + per-point bright loop for one lightcurve,
   ported from the OpenCL build: ONE THREAD PER POINT, the CUDA_BLOCK_DIM
   threads of the block take the points round-robin. Handles both relative
   (Inrel=1) and absolute (Inrel=0) lightcurves; updates dave/ave/np exactly
   as the old mrqcof_curve1 did.

   Per point, two passes over the facets: the cheap visibility test first
   builds this thread's list of visible facets (all threads walk the facets
   in lockstep, so the CUDA_Nor reads broadcast from constant memory), then
   the division-heavy terms run over that list only (a warp executes them
   max(visible count) times instead of for every facet any lane can see).
   The derivatives w.r.t. the shape coefficients are gathered BRIGHT_GB
   columns per pass over the list, i.e. BRIGHT_GB fma per CUDA_Dsph row
   visit. */
#define BRIGHT_GB 16

__device__ void bright_curve1(freq_context* __restrict__ CUDA_LCC,
	double const* __restrict__ a,
	int Inrel, int Lpoints)
{
	const int tid = threadIdx.x;

	const int nc = CUDA_ncoef0;
	const int ma = CUDA_ma;
	const int nshape = nc - 3;           /* last shape-coefficient row */
	const int nf = CUDA_Numfac;
	const int lnp0 = (*CUDA_LCC).np;
	const int iStart = Inrel + 1;        /* absolute lightcurves keep row 1 */

	/* per-curve invariants: shared, not per-thread registers - they would
	   stay live across the whole point loop */
	double* __restrict__ inv = mrq_share_block()->b.inv;
	if (tid == 0)
	{
		inv[0] = a[nc];          /* omega */
		inv[1] = a[nc + 1];
		inv[2] = a[nc + 2];
		inv[3] = a[nc + 3];
		inv[4] = 0;              /* unused (kept for layout clarity) */
		inv[5] = exp(a[ma - 1]); /* Lambert */
		inv[6] = a[ma];          /* Lommel-Seeliger */
		inv[7] = (*CUDA_LCC).Blmat[1][3];
		inv[8] = (*CUDA_LCC).Blmat[3][3];
		inv[9] = (*CUDA_LCC).Blmat[2][1];
		inv[10] = (*CUDA_LCC).Blmat[2][2];
	}
	__syncthreads();
	const double cl = inv[5], cls = inv[6];

	double const* __restrict__ areap = &CUDA_Area[blockIdx.x * CUDA_Numfac1];
	double* __restrict__ dytemp = (*CUDA_LCC).dytemp;
	double* __restrict__ ytemp = (*CUDA_LCC).ytemp;

	/* per-thread visible-facet list (local memory, interleaved per thread,
	   so the same list position is coalesced across a warp) */
	short incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];

#pragma unroll 1
	for (int jp = tid + 1; jp <= Lpoints; jp += CUDA_BLOCK_DIM)
	{
		double po[GEOM_PT_SIZE];
		bright_point_geometry(lnp0 + jp, inv, po);

		int cnt = 0;
#pragma unroll 4
		for (int i = 1; i <= nf; i++)
		{
			double n0 = CUDA_Nor[i][0], n1 = CUDA_Nor[i][1], n2 = CUDA_Nor[i][2];
			double lmu = po[16] * n0 + po[17] * n1 + po[18] * n2;
			double lmu0 = po[19] * n0 + po[20] * n1 + po[21] * n2;
			if ((lmu > TINY) && (lmu0 > TINY))
				incl[cnt++] = (short)i;
		}

		double br = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
#pragma unroll 1
		for (int c = 0; c < cnt; c++)
		{
			const int i = incl[c];
			/* divergent facet index: global copy, not the constant bank */
			double n0 = CUDA_NorG[i][0], n1 = CUDA_NorG[i][1], n2 = CUDA_NorG[i][2];
			double ar = areap[i];
			double lmu = po[16] * n0 + po[17] * n1 + po[18] * n2;
			double lmu0 = po[19] * n0 + po[20] * n1 + po[21] * n2;

			double dnom = lmu + lmu0;
			double s = lmu * lmu0 * (cl + cls / dnom);
			br += ar * s;
			dbr[c] = ar * s;   /* == (Darea*s) * g : the Dg fold, see above */
			double lmu0_dnom = lmu0 / dnom;
			double dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
			double lmu_dnom = lmu / dnom;
			double dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;

			double sum1 = n0 * po[0] + n1 * po[1] + n2 * po[2];
			double sum10 = n0 * po[3] + n1 * po[4] + n2 * po[5];
			double sum2 = n0 * po[6] + n1 * po[7] + n2 * po[8];
			double sum20 = n0 * po[9] + n1 * po[10] + n2 * po[11];
			double sum3 = n0 * po[12] + n1 * po[13];
			double sum30 = n0 * po[14] + n1 * po[15];

			t1 += ar * (dsmu * sum1 + dsmu0 * sum10);
			t2 += ar * (dsmu * sum2 + dsmu0 * sum20);
			t3 += ar * (dsmu * sum3 + dsmu0 * sum30);
			t4 += lmu * lmu0 * ar;
			t5 += ar * lmu * lmu0 / (lmu + lmu0);
		}

		const double Scale = po[22];
		double* __restrict__ row = dytemp + (size_t)(jp - 1) * DYT_STRIDE;

		/* Ders. of brightness w.r.t. rotation parameters */
		row[nshape + 1] = Scale * t1;
		row[nshape + 2] = Scale * t2;
		row[nshape + 3] = Scale * t3;
		/* Ders. of br. w.r.t. phase function params. */
		row[nc + 1] = br * po[23];
		row[nc + 2] = br * po[24];
		row[nc + 3] = br * po[25];
		/* Ders. of br. w.r.t. cl, cls */
		row[ma - 1] = Scale * t4 * cl;
		row[ma] = Scale * t5;
		/* Scaled brightness */
		ytemp[jp] = br * Scale;

		/* Derivatives of brightness w.r.t. the shape coefficients: BRIGHT_GB
		   columns per pass over the visible-facet list. Up to BRIGHT_GB - 1
		   columns past nshape are read (inside the Dsph row) but not stored. */
		if (cnt)
		{
#pragma unroll 1
			for (int i0 = iStart; i0 <= nshape; i0 += BRIGHT_GB)
			{
				double t[BRIGHT_GB];
				{
					const double w = dbr[0];
					double const* __restrict__ r = &CUDA_Dsph[incl[0]][i0];
#pragma unroll
					for (int b = 0; b < BRIGHT_GB; b++)
						t[b] = w * r[b];
				}
#pragma unroll 1
				for (int j = 1; j < cnt; j++)
				{
					const double w = dbr[j];
					double const* __restrict__ r = &CUDA_Dsph[incl[j]][i0];
#pragma unroll
					for (int b = 0; b < BRIGHT_GB; b++)
						t[b] += w * r[b];
				}
#pragma unroll
				for (int b = 0; b < BRIGHT_GB; b++)
					if (i0 + b <= nshape)
						row[i0 + b] = Scale * t[b];
			}
		}
		else
		{
			for (int l = iStart; l <= nshape; l++)
				row[l] = 0;
		}
	} /* jp */

	/* every thread has read np and written its dytemp/ytemp rows */
	__syncthreads();

	if (Inrel == 1)
	{
		/* column sums over the points in point order: the same sums (and
		   summation order) the per-lane accumulators of the warp version
		   produced */
		for (int l = tid + 2; l <= ma; l += CUDA_BLOCK_DIM)
		{
			double s = 0;
			double const* __restrict__ col = dytemp + l;
			for (int jp = 0; jp < Lpoints; jp++)
				s += col[(size_t)jp * DYT_STRIDE];
			(*CUDA_LCC).dave[l] = s;
		}
	}
	if (tid == 0)
	{
		(*CUDA_LCC).np = lnp0 + Lpoints;
		if (Inrel == 1)
		{
			double lave = 0;
			for (int jp = 1; jp <= Lpoints; jp++)
				lave += ytemp[jp];
			(*CUDA_LCC).ave = lave;
		}
	}
}
