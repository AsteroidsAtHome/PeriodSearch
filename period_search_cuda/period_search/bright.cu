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

   * One warp processes one (frequency, pole) block, two data points at a
	 time: the visibility pass runs lanes-across-facets (coalesced), the
	 derivative pass runs lanes-across-coefficients reading CUDA_Dsph rows
	 coalesced, and each row load feeds both points' accumulators.

   * dytemp is stored transposed - dytempT[(jp-1)*DYT_STRIDE + l] - so the
	 derivative writes here and the tile reads in MrqcofCurve2 are coalesced.
	 This requires ma <= DYT_STRIDE-1 = 63 (any spherical-harmonics degree
	 up to 6, i.e. every production workunit); the host enforces it.

   * The per-point geometry (the former matrix_neo pass) is computed
	 in-kernel, one lane per point, into shared memory: the de, de0, e_1..e0_3 and
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
   executed by ONE WARP per block. Handles both relative (Inrel=1) and
   absolute (Inrel=0) lightcurves; updates dave/ave/np exactly as the old
   mrqcof_curve1 did. */
__device__ void bright_curve1_warp(freq_context* __restrict__ CUDA_LCC,
	double const* __restrict__ a,
	int Inrel, int Lpoints)
{
	const int tid = threadIdx.x;
	brightshare* __restrict__ shw = &mrq_share_block()->b;
	double* __restrict__ wcA = shw->wcA;
	double* __restrict__ wcB = shw->wcB;
	int* __restrict__ fc = shw->fc;
	double* __restrict__ inv = shw->inv;

	const int nc = CUDA_ncoef0;
	const int ma = CUDA_ma;
	const int nshape = nc - 3;           /* last shape-coefficient row */
	const int nf = CUDA_Numfac;
	const int lnp0 = (*CUDA_LCC).np;
	const int iStart = Inrel + 1;        /* absolute lightcurves keep row 1 */

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
	__syncwarp();

	/* each lane owns two parameter rows across the whole curve */
	const int c1 = 1 + tid;   /* rows 1..32 */
	const int c2 = 33 + tid;  /* rows 33..64 */
	double dave1 = 0, dave2 = 0;
	double lave = 0;

	double const* __restrict__ areap = &CUDA_Area[blockIdx.x * CUDA_Numfac1];
	double* __restrict__ dytemp = (*CUDA_LCC).dytemp;
	double* __restrict__ ytemp = (*CUDA_LCC).ytemp;

#pragma unroll 1
	for (int jp0 = 1; jp0 <= Lpoints; jp0 += GEO_BATCH)
	{
		int nb = Lpoints - jp0 + 1;
		if (nb > GEO_BATCH) nb = GEO_BATCH;

		/* one lane computes one point's geometry (the acos/sincos/exp-heavy
		   part runs once per point instead of once per lane per point) */
		if (tid < nb)
			bright_point_geometry(lnp0 + jp0 + tid, inv, shw->geo[tid]);
		__syncwarp();

		/* two points (A = jp, B = jp+1) share every CUDA_Dsph row load */
#pragma unroll 1
		for (int jp = jp0; jp < jp0 + nb; jp += 2)
		{
			const int haveB = (jp + 1 < jp0 + nb);
			double const* __restrict__ ptA = shw->geo[jp - jp0];
			double const* __restrict__ ptB = shw->geo[jp - jp0 + (haveB ? 1 : 0)];

			double brA = 0, t1A = 0, t2A = 0, t3A = 0, t4A = 0, t5A = 0;
			double brB = 0, t1B = 0, t2B = 0, t3B = 0, t4B = 0, t5B = 0;
			double accA1 = 0, accA2 = 0, accB1 = 0, accB2 = 0;

#pragma unroll 1
			for (int f0 = 1; f0 <= nf; f0 += 32)
			{
				const int i = f0 + tid;
				double dbrA = 0.0, dbrB = 0.0;
				if (i <= nf)
				{
					double n0 = CUDA_Nor[i][0], n1 = CUDA_Nor[i][1], n2 = CUDA_Nor[i][2];
					double ar = areap[i];
					double cl = inv[5], cls = inv[6];

					{
						double lmu = ptA[16] * n0 + ptA[17] * n1 + ptA[18] * n2;
						double lmu0 = ptA[19] * n0 + ptA[20] * n1 + ptA[21] * n2;
						if ((lmu > TINY) && (lmu0 > TINY))
						{
							double dnom = lmu + lmu0;
							double s = lmu * lmu0 * (cl + cls / dnom);
							brA += ar * s;
							dbrA = ar * s;   /* == (Darea*s) * g : the fold */
							double lmu0_dnom = lmu0 / dnom;
							double dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
							double lmu_dnom = lmu / dnom;
							double dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;

							double sum1 = n0 * ptA[0] + n1 * ptA[1] + n2 * ptA[2];
							double sum10 = n0 * ptA[3] + n1 * ptA[4] + n2 * ptA[5];
							double sum2 = n0 * ptA[6] + n1 * ptA[7] + n2 * ptA[8];
							double sum20 = n0 * ptA[9] + n1 * ptA[10] + n2 * ptA[11];
							double sum3 = n0 * ptA[12] + n1 * ptA[13];
							double sum30 = n0 * ptA[14] + n1 * ptA[15];

							t1A += ar * (dsmu * sum1 + dsmu0 * sum10);
							t2A += ar * (dsmu * sum2 + dsmu0 * sum20);
							t3A += ar * (dsmu * sum3 + dsmu0 * sum30);
							t4A += lmu * lmu0 * ar;
							t5A += ar * lmu * lmu0 / (lmu + lmu0);
						}
					}
					if (haveB)
					{
						double lmu = ptB[16] * n0 + ptB[17] * n1 + ptB[18] * n2;
						double lmu0 = ptB[19] * n0 + ptB[20] * n1 + ptB[21] * n2;
						if ((lmu > TINY) && (lmu0 > TINY))
						{
							double dnom = lmu + lmu0;
							double s = lmu * lmu0 * (cl + cls / dnom);
							brB += ar * s;
							dbrB = ar * s;
							double lmu0_dnom = lmu0 / dnom;
							double dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
							double lmu_dnom = lmu / dnom;
							double dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;

							double sum1 = n0 * ptB[0] + n1 * ptB[1] + n2 * ptB[2];
							double sum10 = n0 * ptB[3] + n1 * ptB[4] + n2 * ptB[5];
							double sum2 = n0 * ptB[6] + n1 * ptB[7] + n2 * ptB[8];
							double sum20 = n0 * ptB[9] + n1 * ptB[10] + n2 * ptB[11];
							double sum3 = n0 * ptB[12] + n1 * ptB[13];
							double sum30 = n0 * ptB[14] + n1 * ptB[15];

							t1B += ar * (dsmu * sum1 + dsmu0 * sum10);
							t2B += ar * (dsmu * sum2 + dsmu0 * sum20);
							t3B += ar * (dsmu * sum3 + dsmu0 * sum30);
							t4B += lmu * lmu0 * ar;
							t5B += ar * lmu * lmu0 / (lmu + lmu0);
						}
					}
				}

				/* compact the visible facets (union of both points) so the
				   derivative sweep below is branch-free with independent,
				   pipelineable loads */
				unsigned vis = __ballot_sync(0xffffffff, (dbrA != 0.0) || (dbrB != 0.0));
				int cnt = __popc(vis);
				if ((dbrA != 0.0) || (dbrB != 0.0))
				{
					int pos = __popc(vis & ((1u << tid) - 1u));
					wcA[pos] = dbrA;
					wcB[pos] = dbrB;
					fc[pos] = i;
				}
				__syncwarp();

#pragma unroll 4
				for (int j = 0; j < cnt; j++)
				{
					double wA = wcA[j];
					double wB = wcB[j];
					double const* __restrict__ row = CUDA_Dsph[fc[j]];
					double v1 = row[c1];
					accA1 += wA * v1;
					accB1 += wB * v1;
					if (c2 <= nshape)
					{
						double v2 = row[c2];
						accA2 += wA * v2;
						accB2 += wB * v2;
					}
				}
				__syncwarp();
			} /* facet chunks */

			/* butterfly-reduce both points' sums so every lane has them */
#pragma unroll
			for (int off = 16; off > 0; off >>= 1)
			{
				brA += __shfl_xor_sync(0xffffffff, brA, off);
				t1A += __shfl_xor_sync(0xffffffff, t1A, off);
				t2A += __shfl_xor_sync(0xffffffff, t2A, off);
				t3A += __shfl_xor_sync(0xffffffff, t3A, off);
				t4A += __shfl_xor_sync(0xffffffff, t4A, off);
				t5A += __shfl_xor_sync(0xffffffff, t5A, off);
				brB += __shfl_xor_sync(0xffffffff, brB, off);
				t1B += __shfl_xor_sync(0xffffffff, t1B, off);
				t2B += __shfl_xor_sync(0xffffffff, t2B, off);
				t3B += __shfl_xor_sync(0xffffffff, t3B, off);
				t4B += __shfl_xor_sync(0xffffffff, t4B, off);
				t5B += __shfl_xor_sync(0xffffffff, t5B, off);
			}

			/* one transposed dytemp row per point, lanes = parameters */
			{
				double Scale = ptA[22], dphp1 = ptA[23], dphp2 = ptA[24], dphp3 = ptA[25];
				double cl = inv[5];
				double ymod = brA * Scale;
				double* __restrict__ row = dytemp + (size_t)(jp - 1) * DYT_STRIDE;

				double v1, v2;
				if (c1 <= nshape)           v1 = Scale * accA1;
				else if (c1 == nshape + 1)  v1 = Scale * t1A;
				else if (c1 == nshape + 2)  v1 = Scale * t2A;
				else if (c1 == nshape + 3)  v1 = Scale * t3A;
				else if (c1 == nc + 1)      v1 = brA * dphp1;
				else if (c1 == nc + 2)      v1 = brA * dphp2;
				else if (c1 == nc + 3)      v1 = brA * dphp3;
				else if (c1 == ma - 1)      v1 = Scale * t4A * cl;
				else                        v1 = Scale * t5A; /* c1 == ma */
				if (c2 <= nshape)           v2 = Scale * accA2;
				else if (c2 == nshape + 1)  v2 = Scale * t1A;
				else if (c2 == nshape + 2)  v2 = Scale * t2A;
				else if (c2 == nshape + 3)  v2 = Scale * t3A;
				else if (c2 == nc + 1)      v2 = brA * dphp1;
				else if (c2 == nc + 2)      v2 = brA * dphp2;
				else if (c2 == nc + 3)      v2 = brA * dphp3;
				else if (c2 == ma - 1)      v2 = Scale * t4A * cl;
				else                        v2 = Scale * t5A; /* c2 == ma */

				if (c1 >= iStart && c1 <= ma) { row[c1] = v1; if (c1 >= 2) dave1 += v1; }
				if (c2 <= ma) { row[c2] = v2; dave2 += v2; }
				if (tid == 0) ytemp[jp] = ymod;
				lave += ymod;
			}
			if (haveB)
			{
				double Scale = ptB[22], dphp1 = ptB[23], dphp2 = ptB[24], dphp3 = ptB[25];
				double cl = inv[5];
				double ymod = brB * Scale;
				double* __restrict__ row = dytemp + (size_t)jp * DYT_STRIDE;

				double v1, v2;
				if (c1 <= nshape)           v1 = Scale * accB1;
				else if (c1 == nshape + 1)  v1 = Scale * t1B;
				else if (c1 == nshape + 2)  v1 = Scale * t2B;
				else if (c1 == nshape + 3)  v1 = Scale * t3B;
				else if (c1 == nc + 1)      v1 = brB * dphp1;
				else if (c1 == nc + 2)      v1 = brB * dphp2;
				else if (c1 == nc + 3)      v1 = brB * dphp3;
				else if (c1 == ma - 1)      v1 = Scale * t4B * cl;
				else                        v1 = Scale * t5B;
				if (c2 <= nshape)           v2 = Scale * accB2;
				else if (c2 == nshape + 1)  v2 = Scale * t1B;
				else if (c2 == nshape + 2)  v2 = Scale * t2B;
				else if (c2 == nshape + 3)  v2 = Scale * t3B;
				else if (c2 == nc + 1)      v2 = brB * dphp1;
				else if (c2 == nc + 2)      v2 = brB * dphp2;
				else if (c2 == nc + 3)      v2 = brB * dphp3;
				else if (c2 == ma - 1)      v2 = Scale * t4B * cl;
				else                        v2 = Scale * t5B;

				if (c1 >= iStart && c1 <= ma) { row[c1] = v1; if (c1 >= 2) dave1 += v1; }
				if (c2 <= ma) { row[c2] = v2; dave2 += v2; }
				if (tid == 0) ytemp[jp + 1] = ymod;
				lave += ymod;
			}

			/* geo[]/wc/fc are re-written next pass; every lane must be done
			   reading them (lanes run independently since Volta) */
			__syncwarp();
		} /* jp pair */
	} /* geometry batch */

	if (Inrel == 1)
	{
		/* per-lane column sums ARE the dave entries (rows 2..ma) */
		if (c1 >= 2 && c1 <= ma) (*CUDA_LCC).dave[c1] = dave1;
		if (c2 <= ma) (*CUDA_LCC).dave[c2] = dave2;
	}
	if (tid == 0)
	{
		(*CUDA_LCC).np = lnp0 + Lpoints;
		if (Inrel == 1)
			(*CUDA_LCC).ave = lave;
	}
	__syncwarp();
}
