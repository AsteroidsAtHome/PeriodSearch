#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

#define POINTS_MAX         2000             /* max number of data points in one lc. */
#define MAX_N_OBS         20000             /* max number of data points */
#define MAX_LC              200             /* max number of lightcurves */
#define MAX_LINE_LENGTH    1000             /* max length of line in an input file */
#define MAX_N_FPOINTS    500000             /* max number of frequency points */
#define MAX_N_FAC          1000             /* max number of facets */
#define MAX_N_ITER          100             /* maximum number of iterations */
#define MAX_N_PAR           200             /* maximum number of parameters */
#define MAX_LM               10             /* maximum degree and order of sph. harm. */
#define N_PHOT_PAR            5             /* maximum number of parameters in scattering  law */
#define TINY                  1e-8          /* precision parameter for mu, mu0*/
#define N_POLES              10             /* number of initial poles */

/* dytemp is stored transposed - dytemp[(jp-1)*DYT_STRIDE + l], l = 1..ma - so
   consecutive work-items reading consecutive parameters hit consecutive
   addresses. Requires ma <= DYT_STRIDE-1 (spherical-harmonics degree <= 6,
   i.e. every production workunit); enforced on the host. */
#define DYT_STRIDE           64

/* normal-equation accumulation tile: points per rank-K update in
   mrqcof_curve2 */
#define CURVE2_K             8

#define PI                 M_PI             /* 3.14159265358979323846 */
#define AU            149597870.691         /* Astronomical Unit [km] */
#define C_SPEED       299792458             /* speed of light [m/s]*/

#define DEG2RAD      (PI / 180)
#define RAD2DEG      (180 / PI)

#define BLOCK_DIM 128
#pragma OPENCL FP_CONTRACT ON

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

//struct __attribute__((packed)) freq_context
//struct mfreq_context
//struct __attribute__((aligned(8))) mfreq_context
typedef struct mfreq_context
{
	//double* Area;
	//double* Dg;
	//double* alpha;
	//double* covar;
	//double* dytemp;
	//double* ytemp;

	double Area[MAX_N_FAC + 1];
	/* The point- and fit-dimensioned work arrays (alpha, covar, dytemp,
	   ytemp, jp_*, e_*, de, de0) live in a separate runtime-sized scratch
	   buffer - one slice of freq_context.scrStride doubles per work-group,
	   at the offsets recorded in freq_context - instead of compile-time
	   worst-case arrays here. That cuts per-context memory ~6x (2.27 MB ->
	   ~0.4 MB for typical workunits). */
	double beta[MAX_N_PAR + 1];
	double atry[MAX_N_PAR + 1];
	double da[MAX_N_PAR + 1];
	double cg[MAX_N_PAR + 1];
	double Blmat[4][4];
	double Dblm[3][4][4];
	double dave[MAX_N_PAR + 1];
	double dyda[MAX_N_PAR + 1];

	double sh_big[BLOCK_DIM];
	double chck[4];
	double pivinv;
	double ave;
	double freq;
	double Alamda;
	double Chisq;
	double Ochisq;
	double rchisq;
	double trial_chisq;
	double iter_diff, dev_old, dev_new;

	int Niter;
	int np, np1, np2;
	int isInvalid, isAlamda, isNiter;
	int icol;
	//double conw_r;

	int ipiv[MAX_N_PAR + 1];
	int indxc[MAX_N_PAR + 1];
	int indxr[MAX_N_PAR + 1];
	int sh_icol[BLOCK_DIM];
	int sh_irow[BLOCK_DIM];
} CUDA_LCC;

//struct freq_context
//typedef struct __attribute__((aligned(8))) freq_context
struct freq_context
{
	double Phi_0;
	double logCl;
	double cl;
	//double logC;
	double lambda_pole[N_POLES + 1];
	double beta_pole[N_POLES + 1];


	double par[4];
	double Alamda_start;
	double Alamda_incr;

	//double cgFirst[MAX_N_PAR + 1];
	double tim[MAX_N_OBS + 1];
	double ee[MAX_N_OBS + 1][3];	// double* ee;
	double ee0[MAX_N_OBS + 1][3];	// double* ee0;
	double Sig[MAX_N_OBS + 1];
	double Weight[MAX_N_OBS + 1];
	double Brightness[MAX_N_OBS + 1];
	double Fc[MAX_N_FAC + 1][MAX_LM + 1];
	double Fs[MAX_N_FAC + 1][MAX_LM + 1];
	double Darea[MAX_N_FAC + 1];
	double Nor[MAX_N_FAC + 1][3];
	double Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	double Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
	double conw_r;

	int ia[MAX_N_PAR + 1];

	int Dg_block;
	int lastone;
	int lastma;
	int ma;
	int Mfit, Mfit1;
	int Mmax, Lmax;
	int n;
	int Ncoef, Ncoef0;
	int Numfac;
	int Numfac1;
	int Nphpar;
	int ndata;
	int Is_Precalc;

	/* runtime dimensions + per-context offsets (in doubles) into the
	   scratch buffer that replaced the fixed-size work arrays */
	int lcPoints1;
	int scrStride;
	int offAlpha;
	int offCovar;
	int offDytemp;
	int offYtemp;
	int offJpScale;
	int offJpDphp1;
	int offJpDphp2;
	int offJpDphp3;
	int offE1;
	int offE2;
	int offE3;
	int offE01;
	int offE02;
	int offE03;
	int offDe;
	int offDe0;
};

//struct freq_result
//struct __attribute__((aligned(8))) freq_result
struct freq_result
{
	double dark_best, per_best, dev_best, dev_best_x2, la_best, be_best, freq;
	int isReported, isInvalid, isNiter;
};
/* WORKAROUND(rusticl / aco): runtime f64 '/' returns results with ~3*2^-29
   relative error (verified by [DIVTEST]); fma() and '*' are exact.
   Markstein sequence: two Newton steps refine the reciprocal, the final
   fused correction restores correct rounding.
   NATIVE_DIV_OK=1 (set by the host after the startup probe) replaces the
   helper with plain '/' on drivers whose division is correctly rounded;
   both paths then produce identical bits, so determinism is preserved. */
#ifndef NATIVE_DIV_OK
#define NATIVE_DIV_OK 0
#endif
#if NATIVE_DIV_OK
#define ddiv(a, b) ((a) / (b))
#else
inline double ddiv(double a, double b)
{
    double r = 1.0 / b;
    double e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    double q = a * r;
    return fma(fma(-b, q, a), r, q);
}
#endif

/*
    FROM stackoverflow: https://stackoverflow.com/questions/42856717/intrinsics-equivalent-to-the-cuda-type-casting-intrinsics-double2loint-doub
    You can express these operations via a union. This will not create extra overhead with modern compilers as long as optimization is on (nvcc -O3 ...).
*/

//struct HiLo
//{
//    int lo;
//    int hi;
//};
//
//typedef struct HiLo hilo;
//
//union U {
//    double val;
//    hilo hiLo;
//};
//
//double HiLoint2double(int hi, int lo)
//{
//    union U u;
//
//    u.hiLo.hi = hi;
//    u.hiLo.lo = lo;
//
//    return u.val;
//}

typedef union {
    double val;
    struct {
        int lo;
        int hi;
    };
} un;

double HiLoint2double(int hi, int lo)
{
    /*union {
        double val;
        struct {
            int lo;
            int hi;
        };
    } u;*/
    un u;

    u.hi = hi;
    u.lo = lo;
    return u.val;
}


int double2hiint(double val)
{
    un u;
    u.val = val;
    return u.hi;
}

int double2loint(double val)
{
    un u;
    u.val = val;
    return u.lo;
}

//int __double2hiint(double val)
//{
//    union {
//        double val;
//        struct {
//            int lo;
//            int hi;
//        };
//    } u;
//    u.val = val;
//
//    return u.hi;
//}
//
//int __double2loint(double val)
//{
//    union {
//        double val;
//        struct {
//            int lo;
//            int hi;
//        };
//    } u;
//    u.val = val;
//
//    return u.lo;
//}
//
//int2 __double2int2(double val) {
//    int2 result;
//
//    result.x = __double2hiint(val);
//    result.y = __double2loint(val);
//
//    return result;
//}

void SwapDouble(double a, double b) 
{ 
	double temp = a; 
	a = b; 
	b = temp; 
} //beta, lambda rotation matrix and its derivatives

 //  8.11.2006


//#include <math.h>
//#include "globals_CUDA.h"

void blmatrix(__global struct mfreq_context* CUDA_LCC, double bet, double lam)
{
	double cb, sb, cl, sl;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	sb = sincos(bet, &cb);
  	sl = sincos(lam, &cl);
	(*CUDA_LCC).Blmat[1][1] = cb * cl;
	(*CUDA_LCC).Blmat[1][2] = cb * sl;
	(*CUDA_LCC).Blmat[1][3] = -sb;
	(*CUDA_LCC).Blmat[2][1] = -sl;
	(*CUDA_LCC).Blmat[2][2] = cl;
	(*CUDA_LCC).Blmat[2][3] = 0;
	(*CUDA_LCC).Blmat[3][1] = sb * cl;
	(*CUDA_LCC).Blmat[3][2] = sb * sl;
	(*CUDA_LCC).Blmat[3][3] = cb;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("bet: %10.7f, lam: %10.7f\n", bet, lam);
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	/* Ders. of Blmat w.r.t. bet */
	(*CUDA_LCC).Dblm[1][1][1] = -sb * cl;
	(*CUDA_LCC).Dblm[1][1][2] = -sb * sl;
	(*CUDA_LCC).Dblm[1][1][3] = -cb;
	(*CUDA_LCC).Dblm[1][2][1] = 0;
	(*CUDA_LCC).Dblm[1][2][2] = 0;
	(*CUDA_LCC).Dblm[1][2][3] = 0;
	(*CUDA_LCC).Dblm[1][3][1] = cb * cl;
	(*CUDA_LCC).Dblm[1][3][2] = cb * sl;
	(*CUDA_LCC).Dblm[1][3][3] = -sb;
	/* Ders. w.r.t. lam */
	(*CUDA_LCC).Dblm[2][1][1] = -cb * sl;
	(*CUDA_LCC).Dblm[2][1][2] = cb * cl;
	(*CUDA_LCC).Dblm[2][1][3] = 0;
	(*CUDA_LCC).Dblm[2][2][1] = -cl;
	(*CUDA_LCC).Dblm[2][2][2] = -sl;
	(*CUDA_LCC).Dblm[2][2][3] = 0;
	(*CUDA_LCC).Dblm[2][3][1] = -sb * sl;
	(*CUDA_LCC).Dblm[2][3][2] = sb * cl;
	(*CUDA_LCC).Dblm[2][3][3] = 0;
}
 //Curvature function (and hence facet area) from Laplace series

 //  8.11.2006


void curv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int brtmpl,
	int brtmph)
{
	int n;
	double fsum, g;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//        brtmpl:  1, 4, 7... 382
	//		  brtmph:  3, 6, 9... 288
	int q = 0;
	for (int i = brtmpl; i <= brtmph; i++, q++)
	{
		//if (blockIdx.x == 0)
		//	printf("i: %d\n", i);

		g = 0;
		n = 0;
		for (int m = 0; m <= (*CUDA_CC).Mmax; m++) // Mmax = 6
		{
			for (int l = m; l <= (*CUDA_CC).Lmax; l++)  // Lmax = 6
			{
				n++;
				//if (blockIdx.x == 0 && threadIdx.x == 0)
				//	printf("cg[%3d]: %10.7f\n", n, cg[n]);

				fsum = cg[n] * (*CUDA_CC).Fc[i][m];
				if (m != 0)
				{
					n++;
					//if (blockIdx.x == 0 && threadIdx.x == 0)
					//	printf("cg[%3d]: %10.7f\n", n, cg[n]);

					fsum = fsum + cg[n] * (*CUDA_CC).Fs[i][m];
				}

				g = g + (*CUDA_CC).Pleg[i][l][m] * fsum;
			}
		}

		g = exp(g);
		(*CUDA_LCC).Area[i] = (*CUDA_CC).Darea[i] * g;

		//if (blockIdx.x == 0)
		//	printf("[%3d - %3d] i: %3d\n", q, threadIdx.x, i);

		//if (blockIdx.x == 0)
		//	printf("Area[%d]: %.7f\n", i, Area[i]);

		/* Dg is no longer materialized: Dg[i][k] == g * Dsph[i][k] folds into the
		   facet weights through Area (= Darea * g) - see bright.cl and conv.cl */
	}
}

/* main-triangle elements per work-item: (DYT_STRIDE-1) rows at most */
#define C2_EPT (((DYT_STRIDE - 1) * DYT_STRIDE / 2 + BLOCK_DIM - 1) / BLOCK_DIM)

void mrqcof_curve2(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* alpha,
	__global double* beta,
	__local double (*dydaT)[DYT_STRIDE],
	__local double* s2wS,
	__local double* dwsS,
	__local double* dyS,
	__local double* coefS,		/* 2 * CURVE2_K: per-point coef, coef1 */
	int inrel,
	int lpoints,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
	int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
	double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);


	//precalc thread boundaries
	int tmph, tmpl;
	tmph = lpoints / BLOCK_DIM;
	if (lpoints % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	lnp1 = (*CUDA_LCC).np1 + tmpl;
	tmph = tmpl + tmph;
	if (tmph > lpoints) tmph = lpoints;
	tmpl++;

	int matmph, matmpl;									// threadIdx.x == 1
	matmph = (*CUDA_CC).ma / BLOCK_DIM;					// 0
	if ((*CUDA_CC).ma % BLOCK_DIM) matmph++;			// 1
	matmpl = threadIdx.x * matmph;						// 1
	matmph = matmpl + matmph;							// 2
	if (matmph > (*CUDA_CC).ma) matmph = (*CUDA_CC).ma;
	matmpl++;											// 2

	int latmph, latmpl;
	latmph = (*CUDA_CC).lastone / BLOCK_DIM;
	if ((*CUDA_CC).lastone % BLOCK_DIM) latmph++;
	latmpl = threadIdx.x * latmph;
	latmph = latmpl + latmph;
	if (latmph > (*CUDA_CC).lastone) latmph = (*CUDA_CC).lastone;
	latmpl++;

	/* The relative-lightcurve renormalization (ytemp *= coef, dytemp column 1
	   zeroed, dytemp[l] = coef * (dytemp[l] - coef1 * dave[l]) for l >= 2) used
	   to be a separate in-place pass over the whole curve in global memory
	   before the tiles re-read it. It is now applied while each tile is staged
	   into local memory - same expressions, same values - which saves a full
	   read + write of dytemp per curve (port of the HIP tree's fused I1
	   renormalization). dytemp/ytemp are per-curve scratch: nothing reads the
	   renormalized global copies afterwards. */
	const int lnp1b = (*CUDA_LCC).np1;	/* point jp uses Sig[lnp1b + jp] */
	const double ave = (*CUDA_LCC).ave;
	__local double* coef1S = coefS + CURVE2_K;

	/* everyone has read np1 before work-item 0 advances it */
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np1 += lpoints;
	}

	lnp2 = (*CUDA_LCC).np2;
	ltrial_chisq = (*CUDA_LCC).trial_chisq;

	/* 2026 rewrite: the normal equations are accumulated once per
	   CURVE2_K-point tile (a rank-K update from a local-memory-staged dyda
	   tile) instead of once per data point. The old code swept the whole
	   triangular alpha matrix in global memory with a read-modify-write per
	   point, plus TWO work-group barriers per matrix row per point; those
	   barriers protected nothing (the staged derivatives are read-only during
	   the sweep and every alpha/beta slot has exactly one writer), so the
	   tile needs just two barriers total. Both original index variants -
	   absolute (ia[1]!=0) and relative (ia[1]==0, column shift m-1, frozen
	   first parameter, gated tail rows) - are reproduced element for element;
	   within a tile only the summation order over the K points changes.

	   dydaT[p][l] is point jp0+p's staged derivative row (renormalization
	   applied while staging for a relative curve), 1-based parameter l. */
	int jp0, p, P;
	double wp[CURVE2_K];

	/* Main triangle (rows l = l0..lastone, columns m = l0..l) flattened over
	   the work-group: work-item t owns elements t, t + BLOCK_DIM, ... for the
	   whole curve and keeps their running alpha values in registers, adding
	   each tile's contribution exactly as the in-memory
	   alpha = alpha + acc update did (same values, same order), and writes
	   them back once at the end. Before, rows were swept one at a time with
	   only l of BLOCK_DIM work-items busy and every tile did a global
	   read-modify-write of the whole triangle. The two original index
	   variants, element for element:
	     absolute (ia[1] != 0): l0 = 1, alpha[l][m], beta[l]
	     relative (ia[1] == 0): l0 = 2, alpha[l-1][m-1], beta[l-1]
	   (frozen size scale). The ia-gated tail rows keep their code. */
	const int rel = !(*CUDA_CC).ia[1];
	const int l0 = rel ? 2 : 1;
	const int nrows = (*CUDA_CC).lastone - l0 + 1;
	const int ntri = nrows > 0 ? nrows * (nrows + 1) / 2 : 0;
	const int Mfit1 = (*CUDA_CC).Mfit1;
	int triLM[C2_EPT];		/* l * 64 + m of each owned element */
	double triA[C2_EPT];	/* its running alpha value */
	#pragma unroll
	for (int t = 0; t < C2_EPT; t++)
	{
		int e = threadIdx.x + t * BLOCK_DIM;
		triLM[t] = 0;
		triA[t] = 0;
		if (e < ntri)
		{
			/* row r of the triangle holds r + 1 elements */
			int r = (int)((sqrt(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
			while ((r + 1) * (r + 2) / 2 <= e) r++;
			while (r * (r + 1) / 2 > e) r--;
			int lr = l0 + r, mr = l0 + (e - r * (r + 1) / 2);
			triLM[t] = lr * 64 + mr;
			triA[t] = alpha[(lr - rel) * Mfit1 + (mr - rel)];
		}
	}
	const int browL = l0 + threadIdx.x;	/* beta row owned by this work-item */
	const int ownB = browL <= (*CUDA_CC).lastone;
	double betaR = ownB ? beta[browL - rel] : 0;

	for (jp0 = 1; jp0 <= lpoints; jp0 += CURVE2_K)
	{
		P = lpoints - jp0 + 1;
		if (P > CURVE2_K) P = CURVE2_K;

		/* per-point scalars; for a relative curve also the renormalization
		   factors, with the expressions of the former in-place pass */
		if (threadIdx.x < P)
		{
			jp = jp0 + threadIdx.x;
			ymod = ytempG[jp];
			if (inrel)
			{
				coef = ddiv((*CUDA_CC).Sig[lnp1b + jp] * lpoints, ave);
				coef1 = ddiv(ymod, ave);
				coefS[threadIdx.x] = coef;
				coef1S[threadIdx.x] = coef1;
				ymod = coef * ymod;
			}
			sig2i = ddiv(1.0, ((*CUDA_CC).Sig[lnp2 + jp] * (*CUDA_CC).Sig[lnp2 + jp]));
			wght = (*CUDA_CC).Weight[lnp2 + jp];
			dy = (*CUDA_CC).Brightness[lnp2 + jp] - ymod;
			double sig2iwght = sig2i * wght;
			s2wS[threadIdx.x] = sig2iwght;
			dwsS[threadIdx.x] = dy * sig2iwght;
			dyS[threadIdx.x] = dy;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* stage the tile (consecutive work-items copy consecutive addresses),
		   renormalizing on the way for a relative curve */
		for (m = threadIdx.x; m < P * DYT_STRIDE; m += BLOCK_DIM)
		{
			double v = dytempG[(jp0 - 1) * DYT_STRIDE + m];
			if (inrel)
			{
				p = m / DYT_STRIDE;
				l = m % DYT_STRIDE;
				if (l == 1)
					v = 0;	/* size-scale derivative is explicitly zero */
				else if (l >= 2 && l <= (*CUDA_CC).ma)
					v = coefS[p] * (v - coef1S[p] * (*CUDA_LCC).dave[l]);
			}
			((__local double*)&dydaT[0][0])[m] = v;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* main triangle: register-resident elements */
		#pragma unroll
		for (int t = 0; t < C2_EPT; t++)
		{
			if (threadIdx.x + t * BLOCK_DIM < ntri)
			{
				int lr = triLM[t] / 64, mr = triLM[t] % 64;
				double acc = 0;
				for (int pp = 0; pp < P; pp++)
				{
					double w = dydaT[pp][lr] * s2wS[pp];
					acc += w * dydaT[pp][mr];
				}
				triA[t] = triA[t] + acc;
			}
		}
		if (ownB)
		{
			double bacc = 0;
			for (int pp = 0; pp < P; pp++)
				bacc += dwsS[pp] * dydaT[pp][browL];
			betaR = betaR + bacc;
		}

		/* ia-gated tail rows l = lastone+1..lastma: unchanged */
		j = nrows > 0 ? nrows : 0;
		l = (*CUDA_CC).lastone + 1;
		if (l < l0) l = l0;
		for (; l <= (*CUDA_CC).lastma; l++)
		{
			if ((*CUDA_CC).ia[l])
			{
				j++;
				for (p = 0; p < P; p++)
					wp[p] = dydaT[p][l] * s2wS[p];

				tmpl = latmpl;
				if (rel && tmpl == 1) tmpl++;	//m==1
				for (m = tmpl; m <= latmph; m++)
				{
					double acc = 0;
					for (p = 0; p < P; p++)
						acc += wp[p] * dydaT[p][m];
					alpha[j * Mfit1 + m - rel] = alpha[j * Mfit1 + m - rel] + acc;
				} /* m */
				if (threadIdx.x == 0)
				{
					k = (*CUDA_CC).lastone - rel;
					for (m = (*CUDA_CC).lastone + 1; m <= l; m++)
					{
						if ((*CUDA_CC).ia[m])
						{
							k++;
							double acc = 0;
							for (p = 0; p < P; p++)
								acc += wp[p] * dydaT[p][m];
							alpha[j * Mfit1 + k] = alpha[j * Mfit1 + k] + acc;
						}
					} /* m */
					double bacc = 0;
					for (p = 0; p < P; p++)
						bacc += dwsS[p] * dydaT[p][l];
					beta[j] = beta[j] + bacc;
				}
			}
		} /* l */

		/* chi-square: same per-point terms in the same ascending order */
		for (p = 0; p < P; p++)
		{
			ltrial_chisq = ltrial_chisq + dyS[p] * dyS[p] * s2wS[p];
		}

		/* everyone must finish reading dydaT before the next tile overwrites it */
		barrier(CLK_LOCAL_MEM_FENCE);
	} /* jp0 */

	#pragma unroll
	for (int t = 0; t < C2_EPT; t++)
	{
		if (threadIdx.x + t * BLOCK_DIM < ntri)
		{
			int lr = triLM[t] / 64, mr = triLM[t] % 64;
			alpha[(lr - rel) * Mfit1 + (mr - rel)] = triA[t];
		}
	}
	if (ownB)
		beta[browL - rel] = betaR;

	lnp2 += lpoints;

	if (threadIdx.x == 0)
	{
		//printf("[%d] ltrial_chisq: %10.7f\n", blockIdx.x, ltrial_chisq);

		(*CUDA_LCC).np2 = lnp2;
		(*CUDA_LCC).trial_chisq = ltrial_chisq;
	}
}

//computes integrated brightness of all visible and iluminated areas
//  and its derivatives

//  8.11.2006


void matrix_neo(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int lnp1,
	int Lpoints,
	int num,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global double* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global double* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global double* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global double* e_1G = scr + (*CUDA_CC).offE1;
	__global double* e_2G = scr + (*CUDA_CC).offE2;
	__global double* e_3G = scr + (*CUDA_CC).offE3;
	__global double* e0_1G = scr + (*CUDA_CC).offE01;
	__global double* e0_2G = scr + (*CUDA_CC).offE02;
	__global double* e0_3G = scr + (*CUDA_CC).offE03;
	__global double* deG = scr + (*CUDA_CC).offDe;
	__global double* de0G = scr + (*CUDA_CC).offDe0;
	__private double f, cf, sf, pom, pom0, alpha;
	__private double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	__private int lnp;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	lnp = lnp1 + brtmpl - 1;
	//printf("lnp: %3d = lnp1: %3d + brtmpl: %3d - 1 | lnp++: %3d\n", lnp, lnp1, brtmpl, lnp + 1);

	int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%3d] cg[%3d]: %10.7f\n", blockIdx.x,  q, (*CUDA_LCC).cg[q]);

	for (int jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;

		ee_1 = (*CUDA_CC).ee[lnp][0];		// position vectors
		ee0_1 = (*CUDA_CC).ee0[lnp][0];
		ee_2 = (*CUDA_CC).ee[lnp][1];
		ee0_2 = (*CUDA_CC).ee0[lnp][1];
		ee_3 = (*CUDA_CC).ee[lnp][2];
		ee0_3 = (*CUDA_CC).ee0[lnp][2];
		t = (*CUDA_CC).tim[lnp];

		//if (blockIdx.x == 0)
		//	printf("jp[%3d] lnp[%3d], %10.7f, %10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",
		//		jp, lnp, ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3);

		//printf("tim[%3d]: %10.7f\n", lnp, t);
		//printf("lnp: %3d, ee[%d]: %.7f, ee0[%d]: %.7f\n", lnp, lnp * 3 + 0, (*CUDA_CC).ee[lnp][0], lnp, (*CUDA_CC).ee0[lnp][0]);

		alpha = acos(clamp(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3, -1.0, 1.0));


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] alpha[%3d]: %.7f, cg[%3d]: %10.7f\n", jp, alpha, q, (*CUDA_LCC).cg[q]);

		/* Exp-lin model (const.term=1.) */
		double f = exp(-ddiv(alpha, cg[(*CUDA_CC).Ncoef0 + 2]));	//f is temp here

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] [%2d][%3d] jp[%3d] f: %10.7f, cg[%3d] %10.7f, alpha %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, f, (*CUDA_CC).Ncoef0 + 2, cg[(*CUDA_CC).Ncoef0 + 2], alpha);

		jp_ScaleG[jp] = 1 + cg[(*CUDA_CC).Ncoef0 + 1] * f + (cg[(*CUDA_CC).Ncoef0 + 3] * alpha);
		jp_dphp_1G[jp] = f;
		jp_dphp_2G[jp] = ddiv(cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha, cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
		jp_dphp_3G[jp] = alpha;

		//if (blockIdx.x == 0)
		//	printf("[neo] [%d][%3d] jp_Scale[%3d]: %10.7f, jp_dphp_1[]: %10.7F, jp_dphp_2[]: %10.7f, jp_dphp_3[]: %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, jp_ScaleG[jp], jp_dphp_1G[jp], jp_dphp_2G[jp], jp_dphp_3G[jp]);

		//  matrix start
		f = cg[(*CUDA_CC).Ncoef0] * t + (*CUDA_CC).Phi_0;
		f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
		sf = sincos(f, &cf);

		//if (threadIdx.x == 0)
		//	printf("jp[%3d] [%3d] cf: %10.7f, sf: %10.7f\n", jp, blockIdx.x, cf, sf);

		//if (num == 1 && blockIdx.x == 0 && jp == brtmpl)
		//{
		//	printf("[%2d][%3d][%3d] f: % .6f, cosF: % .6f, sinF: % .6f\n", blockIdx.x, threadIdx.x, jp, f, cf, sf);
		//}

		//	/* rotation matrix, Z axis, angle f */

		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3];
		e_1G[jp] = pom + tmat * ee_3;
		e0_1G[jp] = pom0 + tmat * ee0_3;

		//if (blockIdx.x == 0)
		//	printf("[%3d] jp[%3d] %10.7f, %10.7f\n", threadIdx.x, jp, e_1G[jp], e0_1G[jp]);

		tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3];
		e_2G[jp] = pom + tmat * ee_3;
		e0_2G[jp] = pom0 + tmat * ee0_3;

		tmat = (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Blmat[3][3];
		e_3G[jp] = pom + tmat * ee_3;
		e0_3G[jp] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3];
		deG[(jp) * 16 + (1) * 4 + (1)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (1) * 4 + (1)] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3];
		deG[(jp) * 16 + (1) * 4 + (2)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (1) * 4 + (2)] = pom0 + tmat * ee0_3;

		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3];
		deG[(jp) * 16 + (1) * 4 + (3)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (1) * 4 + (3)] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3];
		deG[(jp) * 16 + (2) * 4 + (1)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (2) * 4 + (1)] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3];
		deG[(jp) * 16 + (2) * 4 + (2)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (2) * 4 + (2)] = pom0 + tmat * ee0_3;

		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3];
		deG[(jp) * 16 + (2) * 4 + (3)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (2) * 4 + (3)] = pom0 + tmat * ee0_3;

		tmat = (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Dblm[1][3][3];
		deG[(jp) * 16 + (3) * 4 + (1)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (3) * 4 + (1)] = pom0 + tmat * ee0_3;

		tmat = (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Dblm[2][3][3];
		deG[(jp) * 16 + (3) * 4 + (2)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (3) * 4 + (2)] = pom0 + tmat * ee0_3;


		deG[(jp) * 16 + (3) * 4 + (3)] = 0;
		de0G[(jp) * 16 + (3) * 4 + (3)] = 0;
	}
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int jp,
	int Lpoints1,
	int Inrel,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
	__global double* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global double* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global double* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global double* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global double* e_1G = scr + (*CUDA_CC).offE1;
	__global double* e_2G = scr + (*CUDA_CC).offE2;
	__global double* e_3G = scr + (*CUDA_CC).offE3;
	__global double* e0_1G = scr + (*CUDA_CC).offE01;
	__global double* e0_2G = scr + (*CUDA_CC).offE02;
	__global double* e0_3G = scr + (*CUDA_CC).offE03;
	__global double* deG = scr + (*CUDA_CC).offDe;
	__global double* de0G = scr + (*CUDA_CC).offDe0;
	double cl, cls, dnom, s, Scale;
	double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	int ncoef0, ncoef, i, j, incl_count = 0;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = (*CUDA_CC).ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	/* matrix from neo */
	/* derivatives */
	e_1 = e_1G[jp];
	e_2 = e_2G[jp];
	e_3 = e_3G[jp];
	e0_1 = e0_1G[jp];
	e0_2 = e0_2G[jp];
	e0_3 = e0_3G[jp];
	de[1][1] = deG[(jp) * 16 + (1) * 4 + (1)];
	de[1][2] = deG[(jp) * 16 + (1) * 4 + (2)];
	de[1][3] = deG[(jp) * 16 + (1) * 4 + (3)];
	de[2][1] = deG[(jp) * 16 + (2) * 4 + (1)];
	de[2][2] = deG[(jp) * 16 + (2) * 4 + (2)];
	de[2][3] = deG[(jp) * 16 + (2) * 4 + (3)];
	de[3][1] = deG[(jp) * 16 + (3) * 4 + (1)];
	de[3][2] = deG[(jp) * 16 + (3) * 4 + (2)];
	de[3][3] = deG[(jp) * 16 + (3) * 4 + (3)];
	de0[1][1] = de0G[(jp) * 16 + (1) * 4 + (1)];
	de0[1][2] = de0G[(jp) * 16 + (1) * 4 + (2)];
	de0[1][3] = de0G[(jp) * 16 + (1) * 4 + (3)];
	de0[2][1] = de0G[(jp) * 16 + (2) * 4 + (1)];
	de0[2][2] = de0G[(jp) * 16 + (2) * 4 + (2)];
	de0[2][3] = de0G[(jp) * 16 + (2) * 4 + (3)];
	de0[3][1] = de0G[(jp) * 16 + (3) * 4 + (1)];
	de0[3][2] = de0G[(jp) * 16 + (3) * 4 + (2)];
	de0[3][3] = de0G[(jp) * 16 + (3) * 4 + (3)];

	/*Integrated brightness (phase coeff. used later) */
	double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	short int incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];

	br = 0;
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	tmp5 = 0;

	/* Two passes: the cheap visibility test first builds this work-item's
	   list of visible facets, then the division-heavy terms run over that
	   list. In a single pass a wavefront executed the heavy block for every
	   facet that ANY of its lanes could see, i.e. for nearly all facets;
	   now it runs max(incl_count) times per wavefront. lmu/lmu0 are
	   recomputed with the same expressions and the sums still run over the
	   visible facets in ascending order. */
	for (i = 1; i <= (*CUDA_CC).Numfac; i++)
	{
		lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
		lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];
		if ((lmu > TINY) && (lmu0 > TINY))
		{
			incl[incl_count] = i;
			incl_count++;
		}
	}

	for (int c = 0; c < incl_count; c++)
	{
		i = incl[c];
		j = i;
		lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
		lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + ddiv(cls, dnom));
			ar = (*CUDA_LCC).Area[j];
			br += ar * s;

			/* Darea[i] * s * Dg[i][k] == Darea[i] * s * g * Dsph[i][k]
			   == (Area[i] * s) * Dsph[i][k]: fold g into the weight and
			   gather from the one read-only, facet-major Dsph shared by
			   all work-groups instead of the per-context Dg matrix */
			dbr[c] = ar * s;

			double lmu0_dnom = ddiv(lmu0, dnom);
			dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
			double lmu_dnom = ddiv(lmu, dnom);
			dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;


			sum1 = (*CUDA_CC).Nor[i][0] * de[1][1] + (*CUDA_CC).Nor[i][1] * de[2][1] + (*CUDA_CC).Nor[i][2] * de[3][1];
			sum10 = (*CUDA_CC).Nor[i][0] * de0[1][1] + (*CUDA_CC).Nor[i][1] * de0[2][1] + (*CUDA_CC).Nor[i][2] * de0[3][1];
			tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
			sum2 = (*CUDA_CC).Nor[i][0] * de[1][2] + (*CUDA_CC).Nor[i][1] * de[2][2] + (*CUDA_CC).Nor[i][2] * de[3][2];
			sum20 = (*CUDA_CC).Nor[i][0] * de0[1][2] + (*CUDA_CC).Nor[i][1] * de0[2][2] + (*CUDA_CC).Nor[i][2] * de0[3][2];
			tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
			sum3 = (*CUDA_CC).Nor[i][0] * de[1][3] + (*CUDA_CC).Nor[i][1] * de[2][3] + (*CUDA_CC).Nor[i][2] * de[3][3];
			sum30 = (*CUDA_CC).Nor[i][0] * de0[1][3] + (*CUDA_CC).Nor[i][1] * de0[2][3] + (*CUDA_CC).Nor[i][2] * de0[3][3];
			tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

			tmp4 += lmu * lmu0 * ar;
			tmp5 += ar * ddiv(lmu * lmu0, lmu + lmu0);
		}
	}

	Scale = jp_ScaleG[jp];
	i = (jp - 1) * DYT_STRIDE + (ncoef0 - 3 + 1);
	/* Ders. of brightness w.r.t. rotation parameters */
	dytempG[i] = Scale * tmp1;

	i++;
	dytempG[i] = Scale * tmp2;
	i++;
	dytempG[i] = Scale * tmp3;

	i++;
	/* Ders. of br. w.r.t. phase function params. */
	dytempG[i] = br * jp_dphp_1G[jp];
	i++;
	dytempG[i] = br * jp_dphp_2G[jp];
	i++;
	dytempG[i] = br * jp_dphp_3G[jp];

	/* Ders. of br. w.r.t. cl, cls */
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef - 1)] = Scale * tmp4 * cl;
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef)] = Scale * tmp5;

	/* Scaled brightness */
	ytempG[jp] = br * Scale;

	ncoef0 -= 3;
	int iStart;
	int d;

	iStart = Inrel + 1;
	d = (jp - 1) * DYT_STRIDE + iStart;


	/* Derivatives of brightness w.r.t. g-coeffs: BRIGHT_GB columns per pass
	   over the visible-facet list (was 2); each column is still
	   dbr[0] * Dsph[..] followed by the fma chain over the visible facets
	   in ascending order. Up to BRIGHT_GB - 1 columns past ncoef0 are read
	   (inside the Dsph row) but not stored. */
#define BRIGHT_GB 16
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += BRIGHT_GB)
		{
			double t[BRIGHT_GB];
			{
				double l_dbr = dbr[0];
				__global double* row = (*CUDA_CC).Dsph[incl[0]] + i;
				for (int b = 0; b < BRIGHT_GB; b++)
					t[b] = l_dbr * row[b];
			}

			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				__global double* row = (*CUDA_CC).Dsph[incl[j]] + i;
				for (int b = 0; b < BRIGHT_GB; b++)
					t[b] += l_dbr * row[b];
			}

			for (int b = 0; b < BRIGHT_GB; b++)
			{
				if (i + b <= ncoef0)
					dytempG[(jp - 1) * DYT_STRIDE + i + b] = Scale * t[b];
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d++)
			dytempG[d] = 0;
	}

	//return(0);
}
//Convexity regularization function

//  8.11.2006


double conv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* res,
	int nc,
	int brtmpl,
	int brtmph)
{
	int i, j, k;
	double tmp = 0.0;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//j = blockIdx.x * (CUDA_Numfac1)+brtmpl;
	j = brtmpl;
	for (i = brtmpl; i <= brtmph; i++, j++)
	{
		//tmp += CUDA_Area[j] * CUDA_Nor[i][nc];
		tmp += (*CUDA_LCC).Area[j] * (*CUDA_CC).Nor[i][nc];
	}

	res[threadIdx.x] = tmp;

	//if (threadIdx.x == 0)
	//    printf("conv>>> [%d] jp-1[%3d] res[%3d]: %10.7f\n", blockIdx.x, nc, threadIdx.x, res[threadIdx.x]);

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	//parallel reduction
	k = BLOCK_DIM >> 1;
	while (k > 1)
	{
		if (threadIdx.x < k)
			res[threadIdx.x] += res[threadIdx.x + k];
		k = k >> 1;
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		tmp = res[0] + res[1];
	}

	/* the derivatives w.r.t. the shape coefficients are computed for all
	   points at once in mrqcof_curve1_last */

	return (tmp);
}
 //slighly changed code from Numerical Recipes
 //  converted from Mikko's fortran code

 //  8.11.2006


//#include <stdio.h>
//#include <stdlib.h>
//#include "globals_CUDA.h"
//#include "declarations_CUDA.h"


/* comment the following line if no YORP */
/*#define YORP*/

void mrqcof_start(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	__global double* alpha,
	__global double* beta)
{
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);
	int x = threadIdx.x;

	int brtmph, brtmpl;
	// brtmph = 288 / 128 = 2 (2.25)
	brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
	if ((*CUDA_CC).Numfac % BLOCK_DIM)
	{
		brtmph++; // brtmph = 3
	}

	brtmpl = threadIdx.x * brtmph;	// 0 * 3 = 0, 1 * 3 = 3, 6,  9, 12, 15, 18... 381(127 * 3)
	brtmph = brtmpl + brtmph;		//		   3,         6, 9, 12, 15, 18, 21... 384(381 + 3)
	if (brtmph > (*CUDA_CC).Numfac) //  97 * 3 = 201 > 288
	{
		brtmph = (*CUDA_CC).Numfac; // 3, 6, ... max 288
	}

	brtmpl++; // 1..382
	//if(blockIdx.x == 0)
	//	printf("Idx: %d | Numfac: %d | brtmpl: %d | brtmph: %d\n", threadIdx.x, (*CUDA_CC).Numfac, brtmpl, brtmph);

		/*  ---   CURV  ---  */
	curv(CUDA_LCC, CUDA_CC, cg, brtmpl, brtmph);

	if (threadIdx.x == 0)
	{
		//   #ifdef YORP
		//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
		  // #else

		//if (blockIdx.x == 0)
		//	printf("[mrqcof_start] a[%3d]: %10.7f, a[%3d]: %10.7f\n",
		//		(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar, cg[(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar],
		//		(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar, cg[(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar]);

		  /*  ---  BLMATRIX ---  */
		blmatrix(CUDA_LCC, cg[(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar], cg[(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar]);
		//   #endif
		(*CUDA_LCC).trial_chisq = 0.0;
		(*CUDA_LCC).np = 0;
		(*CUDA_LCC).np1 = 0;
		(*CUDA_LCC).np2 = 0;
		(*CUDA_LCC).ave = 0;
	}

	brtmph = (*CUDA_CC).Mfit / BLOCK_DIM;
	if ((*CUDA_CC).Mfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Mfit) brtmph = (*CUDA_CC).Mfit;
	brtmpl++;

	__private int idx, k, j;

	for (j = brtmpl; j <= brtmph; j++)
	{
		for (k = 1; k <= j; k++)
		{
			idx = j * (*CUDA_CC).Mfit1 + k;
			alpha[idx] = 0;
			//if (blockIdx.x == 0 && j < 3)
			//	printf("[%3d] j: %d, k: %d, Mfit1: %2d, alpha[%3d]: %.7f\n", threadIdx.x, j, k, (*CUDA_CC).Mfit1, idx, alpha[idx]);
		}
		beta[j] = 0;
	}


	//int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%d][%3d] cg[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, q, (*CUDA_LCC).cg[q]);


}

void mrqcof_matrix(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int Lpoints,
	int num,
	__global double* scr)
{
	matrix_neo(CUDA_LCC, CUDA_CC, cg, (*CUDA_LCC).np, Lpoints, num, scr);
}

void mrqcof_curve1(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	__local double* tmave,
	int Inrel,
	int Lpoints,
	int num,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
	//__local double tmave[BLOCK_DIM];  // __shared__
	__private int Lpoints1 = Lpoints + 1;
	__private int k, lnp, jp;
	__private double lave;

	lnp = (*CUDA_LCC).np;
	lave = (*CUDA_LCC).ave;

	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//precalc thread boundaries
	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	/* points are dealt out round-robin (jp = t+1, t+1+BLOCK_DIM, ...) rather than
	   in contiguous blocks of ceil(Lpoints/BLOCK_DIM): every point is
	   independent, and this packs the last partial round into as few
	   wavefronts as possible (156 points: 5 wave32 rounds instead of 6). The
	   ytemp partial sums below keep the contiguous blocks, so ave is summed
	   in the same order as before. */
	for (jp = threadIdx.x + 1; jp <= Lpoints; jp += BLOCK_DIM)
	{
			/*  ---  BRIGHT  ---  */
		bright(CUDA_LCC, CUDA_CC, cg, jp, Lpoints1, Inrel, scr);
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	if (Inrel == 1)
	{
		int tmph, tmpl;
		tmph = (*CUDA_CC).ma / BLOCK_DIM;
		if ((*CUDA_CC).ma % BLOCK_DIM) tmph++;
		tmpl = threadIdx.x * tmph;
		tmph = tmpl + tmph;
		if (tmph > (*CUDA_CC).ma) tmph = (*CUDA_CC).ma;
		tmpl++;
		if (tmpl == 1) tmpl++;

		int ixx;
		for (int l = tmpl; l <= tmph; l++)
		{
			//jp==1
			ixx = l;
			(*CUDA_LCC).dave[l] = dytempG[ixx];

			//jp>=2
			ixx += DYT_STRIDE;
			for (int jp = 2; jp <= Lpoints; jp++, ixx += DYT_STRIDE)
			{
				//(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + dytempG[ixx];
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + dytempG[ixx];

				//if (threadIdx.x == 1)
				//	printf("[Device | mrqcof_curv1] [%3d] dytemp[%3d]: %10.7f, dave[%3d]: %10.7f\n", blockIdx.x, ixx, dytempG[ixx], l, (*CUDA_LCC).dave[l]);
			}
		}

		tmave[threadIdx.x] = 0;
		for (int jp = brtmpl; jp <= brtmph; jp++)
		{
			tmave[threadIdx.x] += ytempG[jp];
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		//parallel reduction
		k = BLOCK_DIM >> 1;
		while (k > 1)
		{
			if (threadIdx.x < k) tmave[threadIdx.x] += tmave[threadIdx.x + k];
			k = k >> 1;
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			lave = tmave[0] + tmave[1];
		}
		//parallel reduction end
	}

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp + Lpoints;
		(*CUDA_LCC).ave = lave;
	}
}

void mrqcof_curve1_last(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* a,
	__global double* alpha,
	__global double* beta,
	__local double* res,
	int Inrel,
	int Lpoints,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
	int l, jp, lnp;
	double ymod, lave;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	lnp = (*CUDA_LCC).np;
	//
	if (threadIdx.x == 0)
	{
		if (Inrel == 1) /* is the LC relative? */
		{
			lave = 0;
			for (l = 1; l <= (*CUDA_CC).ma; l++)
				(*CUDA_LCC).dave[l] = 0;
		}
		else
			lave = (*CUDA_LCC).ave;
	}
	//precalc thread boundaries
	int tmph, tmpl;
	tmph = (*CUDA_CC).ma / BLOCK_DIM;
	if ((*CUDA_CC).ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > (*CUDA_CC).ma) tmph = (*CUDA_CC).ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
	if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
	brtmpl++;

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	//if (threadIdx.x == 0)
	//	printf("conv>>> [%d] \n", blockIdx.x);

	/* convexity derivatives dyda[l] = sum_i Area[i] * Dsph[i][l] * Nor[i][nc]
	   of every point (nc = jp-1) in ONE pass over the facets - it used to be
	   one pass per point inside conv(). Area/Dsph are read once for all
	   points and the per-point sums become independent chains; each sum keeps
	   its facet order and operand rounding, and dave[l] still accumulates the
	   points in ascending order (each l belongs to one work-item). */
	for (l = tmpl; l <= tmph; l++)
	{
		for (int jb = 0; jb < Lpoints; jb += 3)
		{
			double d[3] = { 0, 0, 0 };
			if (l <= (*CUDA_CC).Ncoef)
			{
				for (int i = 1; i <= (*CUDA_CC).Numfac; i++)
				{
					/* Darea[i] * Dg[i][l] == Area[i] * Dsph[i][l] (Area = Darea*g) */
					double ad = (*CUDA_LCC).Area[i] * (*CUDA_CC).Dsph[i][l];
					for (int q = 0; q < 3; q++)
						d[q] += ad * (*CUDA_CC).Nor[i][jb + q];
				}
			}
			for (int q = 0; q < 3 && jb + q < Lpoints; q++)
			{
				dytempG[(jb + q) * DYT_STRIDE + l] = d[q];
				if (Inrel == 1)
					(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + d[q];
			}
		}
	}

	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;
		// *--- CONV() ---* //
		ymod = conv(CUDA_LCC, CUDA_CC, res, jp - 1, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			ytempG[jp] = ymod;

			if (Inrel == 1)
				lave = lave + ymod;
		}
		/* save lightcurves */
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		/*         if ((*CUDA_LCC).Lastcall == 1) always ==0
					 (*CUDA_LCC).Yout[np] = ymod;*/
	} /* jp, lpoints */

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp;
		(*CUDA_LCC).ave = lave;
	}
}

double mrqcof_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* alpha)
{
	int j, k;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	/* mirror the lower triangle; each row is split over the work-group
	   (reads are contiguous, source and destination never overlap) */
	int lsize = get_local_size(0);
	for (int j = 2; j <= (*CUDA_CC).Mfit; j++)
	{
		for (k = 1 + threadIdx.x; k <= j - 1; k += lsize)
		{
			alpha[k * (*CUDA_CC).Mfit1 + j] = alpha[j * (*CUDA_CC).Mfit1 + k];
			//if (blockIdx.x ==0 && threadIdx.x == 0)
			//	printf("[mrqcof_end] [%d][%3d] alpha[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, k * (*CUDA_CC).Mfit1 + j, alpha[k * (*CUDA_CC).Mfit1 + j]);
		}
	}

	return (*CUDA_LCC).trial_chisq;
}

//from Numerical Recipes

/* 2026: the damped normal matrix is staged into local memory and the whole
   Gauss-Jordan elimination runs there; global memory is only touched to read
   alpha/beta on entry and to write the step vector da at the end. The old
   version swept covar in global memory on every pivot step. Two consequences
   of the caller's structure are used:

   * the inverted matrix itself is dead - ClCalculateIter1Mrqcof2Start rezeroes
	 covar before mrqcof2 accumulates into it, and mrqmin_2_end copies that
	 fresh accumulation - so neither the solved matrix nor the final
	 column-unscramble pass (and its indxr/indxc bookkeeping) is needed;
	 only da and the return code leave this function;

   * the icol/pivinv broadcast scalars and the pivot-reduction arrays move
	 from per-context global struct members to local memory.

   The local buffers are declared at kernel scope (OpenCL requirement) in
   ClCalculateIter1Mrqmin1End and passed through mrqmin_1_end. Pivot choice
   and elimination order are unchanged, so the computed step is bit-identical
   to the global-memory version. */
int gauss_errc(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* covL,   /* [DYT_STRIDE * DYT_STRIDE], indexed with Mfit1 stride */
	__local double* daL,    /* [DYT_STRIDE] */
	__local int* ipivL,     /* [DYT_STRIDE] */
	__local double* shBig,  /* [BLOCK_DIM] */
	__local int* shIrow,    /* [BLOCK_DIM] */
	__local int* shIcol,    /* [BLOCK_DIM] */
	__local double* pivBC,  /* [1] pivinv broadcast */
	__local int* icolBC,    /* [1] icol broadcast */
	__global double* alphaG)
{
	double big, dum;
	double tmpSwap;
	int i, licol = 0, irow = 0, j, k, l, ll;
	int n = (*CUDA_CC).Mfit;
	int mfit1 = (*CUDA_CC).Mfit1;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = n / BLOCK_DIM;
	if (n % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > n) brtmph = n;
	brtmpl++;

	/* stage the damped matrix and the right-hand side straight from
	   alpha/beta (this replaces the covar staging that mrqmin_1_end used to
	   do in global memory; covar itself is no longer written at all) */
	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * mfit1 + 1;
		for (k = 1; k <= n; k++, ixx++)
		{
			covL[ixx] = alphaG[ixx];
		}
		int qq = j * mfit1 + j;
		covL[qq] = alphaG[qq] * (1 + (*CUDA_LCC).Alamda);
		daL[j] = (*CUDA_LCC).beta[j];
	}

	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++) ipivL[j] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

	for (i = 1; i <= n; i++)
	{
		big = -1.0;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)
		{
			if (ipivL[j] != 1)
			{
				int ixx = j * mfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if (ipivL[k] == 0)
					{
						double tmpcov = fabs(covL[ixx]);
						if (tmpcov >= big)
						{
							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if (ipivL[k] > 1)
					{
						barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
						return(1);
					}
				}
			}
		}
		shBig[threadIdx.x] = big;
		shIrow[threadIdx.x] = irow;
		shIcol[threadIdx.x] = licol;

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		if (threadIdx.x == 0)
		{
			big = shBig[0];
			icolBC[0] = shIcol[0];
			irow = shIrow[0];

			for (j = 1; j < BLOCK_DIM; j++)
			{
				if (shBig[j] >= big)
				{
					big = shBig[j];
					irow = shIrow[j];
					icolBC[0] = shIcol[j];
				}
			}

			++ipivL[icolBC[0]];

			if (irow != icolBC[0])
			{
				for (l = 1; l <= n; l++)
				{
					tmpSwap = covL[irow * mfit1 + l];
					covL[irow * mfit1 + l] = covL[icolBC[0] * mfit1 + l];
					covL[icolBC[0] * mfit1 + l] = tmpSwap;
				}

				tmpSwap = daL[irow];
				daL[irow] = daL[icolBC[0]];
				daL[icolBC[0]] = tmpSwap;
			}

			int covarIdx = icolBC[0] * mfit1 + icolBC[0];

			if (covL[covarIdx] == 0.0)
			{
				for (int l2 = 1; l2 <= (*CUDA_CC).ma; l2++)
				{
					(*CUDA_LCC).atry[l2] = (*CUDA_LCC).cg[l2];
				}

				icolBC[0] = -1;
			}
			else
			{
				pivBC[0] = ddiv(1.0, covL[covarIdx]);
				covL[covarIdx] = 1.0;

				daL[icolBC[0]] = daL[icolBC[0]] * pivBC[0];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		if (icolBC[0] < 0)
		{
			return(2);
		}

		for (l = brtmpl; l <= brtmph; l++)
		{
			int qq = icolBC[0] * mfit1 + l;
			double covar1 = covL[qq] * pivBC[0];
			covL[qq] = covar1;
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		for (ll = brtmpl; ll <= brtmph; ll++)
		{
			if (ll != icolBC[0])
			{
				int ixx = ll * mfit1;
				int jxx = icolBC[0] * mfit1;
				dum = covL[ixx + icolBC[0]];
				covL[ixx + icolBC[0]] = 0.0;
				ixx++;
				jxx++;
				for (l = 1; l <= n; l++, ixx++, jxx++)
				{
					covL[ixx] -= covL[jxx] * dum;
				}

				daL[ll] -= daL[icolBC[0]] * dum;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	/* only the step vector leaves the solver (the column unscramble of the
	   classic routine acted on the inverse, which nothing reads) */
	for (j = brtmpl; j <= brtmph; j++)
	{
		(*CUDA_LCC).da[j] = daL[j];
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return(0);
}
//N.B. The foll. L-M routines are modified versions of Press et al.
//  converted from Mikko's fortran code

//  8.11.2006


int mrqmin_1_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* covL,
	__local double* daL,
	__local int* ipivL,
	__local double* shBig,
	__local int* shIrow,
	__local int* shIcol,
	__local double* pivBC,
	__local int* icolBC,
	__global double* alphaG)
{
	int j;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int ma = (*CUDA_CC).ma;

	//precalc thread boundaries
	int tmph, tmpl;
	tmph = ma / BLOCK_DIM;
	if (ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > ma) tmph = ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = (*CUDA_CC).Mfit / BLOCK_DIM;
	if ((*CUDA_CC).Mfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Mfit) brtmph = (*CUDA_CC).Mfit;
	brtmpl++;

	// <<< Iter1Mrqmin1EndPre1
	if ((*CUDA_LCC).isAlamda)
	{
		for (j = tmpl; j <= tmph; j++)
		{
			(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
		}
	}

	// The damped matrix is staged straight from alpha into local memory by
	// gauss_errc; covar is not touched at all (it is rezeroed by
	// ClCalculateIter1Mrqcof2Start before mrqcof2 accumulates into it).

	// <<< gauss_errc    ---- GAUS ERROR CODE ----
	int err_code = gauss_errc(CUDA_LCC, CUDA_CC, covL, daL, ipivL, shBig, shIrow, shIcol, pivBC, icolBC, alphaG);
	if (err_code)
	{
		return err_code;
	}
	//     __syncthreads(); inside gauss
	// <<< gaus_errc END

	// >>> Iter1Mrqmin1EndPost
	if (threadIdx.x == 0)
	{
		//		if (err_code != 0) return(err_code);  "bacha na sync threads" - Watch out for Sync Threads
		j = 0;
		for (int l = 1; l <= ma; l++)
			if ((*CUDA_CC).ia[l])
			{
				j++;
				(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
			}
	}

	return err_code;
}

void mrqmin_2_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* scr)
{
	__global double* alphaG = scr + (*CUDA_CC).offAlpha;
	__global double* covarG = scr + (*CUDA_CC).offCovar;

	int j, k, l;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	/* the copies are split over the work-group, the scalar updates are done
	   by work-item 0. The branch stays uniform: the only write to Chisq /
	   Ochisq (else branch) sets Chisq = Ochisq, which keeps the test false. */
	int lsize = get_local_size(0);

	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		if (threadIdx.x == 0)
			(*CUDA_LCC).Alamda = ddiv((*CUDA_LCC).Alamda, (*CUDA_CC).Alamda_incr);
		for (j = 1; j <= (*CUDA_CC).Mfit; j++)
		{
			for (k = 1 + threadIdx.x; k <= (*CUDA_CC).Mfit; k += lsize)
			{
				alphaG[j * (*CUDA_CC).Mfit1 + k] = covarG[j * (*CUDA_CC).Mfit1 + k];

				//if (blockIdx.x == 0)
				//	printf("alpha[%3d]: %10.7f\n", alphaG[j * (*CUDA_CC).Mfit1 + k]);
			}
		}
		for (j = 1 + threadIdx.x; j <= (*CUDA_CC).Mfit; j += lsize)
		{
			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		}
		for (l = 1 + threadIdx.x; l <= (*CUDA_CC).ma; l += lsize)
		{
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
		}
	}
	else if (threadIdx.x == 0)
	{
		(*CUDA_LCC).Alamda = (*CUDA_CC).Alamda_incr * (*CUDA_LCC).Alamda;
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}


}
__kernel void ClCalculatePrepare(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_result* CUDA_FR,
    __global int* CUDA_End,
    double freq_start,
    double freq_step,
    int n_max,
    int n_start)
{
    int3 blockIdx;
    blockIdx.x = get_group_id(0);
    int x = blockIdx.x;

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

    /* one work-group per (frequency, pole) pair: N_POLES consecutive groups
       share the same trial frequency and each of them will run one of the
       initial poles, all concurrently (the poles used to be a serial host-side
       loop) */
    int n = n_start + blockIdx.x / N_POLES;


    //zero context
    if (n > n_max)
    {
        //CUDA_mCC[x].isInvalid = 1;
        (*CUDA_LCC).isInvalid = 1;
        (*CUDA_FR).isInvalid = 1;
        return;
    }
    else
    {
        //CUDA_mCC[x].isInvalid = 0;
        (*CUDA_LCC).isInvalid = 0;
        (*CUDA_FR).isInvalid = 0;
    }

    //printf("[%d] n_start: %d | n_max: %d | n: %d \n", blockIdx.x, n_start, n_max, n);

    //printf("Idx: %d | isInvalid: %d\n", x, CUDA_CC[x].isInvalid);
    //printf("Idx: %d | isInvalid: %d\n", x, (*CUDA_LCC).isInvalid);

    //CUDA_mCC[x].freq = freq_start - (n - 1) * freq_step;
    (*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;

    ///* initial poles */
    (*CUDA_LFR).per_best = 0.0;
    (*CUDA_LFR).dark_best = 0.0;
    (*CUDA_LFR).la_best = 0.0;
    (*CUDA_LFR).be_best = 0.0;
    (*CUDA_LFR).dev_best = 1e40;

    //printf("n: %4d, CUDA_CC[%3d].freq: %10.7f, CUDA_FR[%3d].la_best: %10.7f, isInvalid: %4d \n", n, x, (*CUDA_LCC).freq, x, (*CUDA_LFR).la_best, (*CUDA_LCC).isInvalid);

    //if (blockIdx.x == 0)
        //printf("Prepare CUDA_End: %2d\n", *CUDA_End);
}

__kernel void ClCalculatePreparePole(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global struct freq_result* CUDA_FR,
    __global double* CUDA_cg_first,
    __global int* CUDA_End,
    __global struct freq_context* CUDA_CC2)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

    /* launched with BLOCK_DIM work-items per group: the per-group loops are
       strided over the group, the scalar setup is done by work-item 0 */
    const int lsize = get_local_size(0);

    /* CUDA_CC2 is no longer used: it only received a Brightness copy for a
       host-side debug check that has been removed (kept so the kernel
       argument indices stay unchanged) */

    //*CUDA_End = 13;
    //printf("[%d] PreparePole t: %d, CUDA_End: %d\n", x, t, *CUDA_End);


    /* invalid contexts (n > n_max) are not counted here: the host knows how
       many there are and starts CUDA_End at that count; isReported is already
       0 for every context (host initialises CUDA_FR before each batch) */
    if ((*CUDA_LCC).isInvalid)
        return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("[Device] PreparePole > ma: %d\n", (*CUDA_CC).ma);

    //* starts from the initial ellipsoid */
    for (int i = 1 + threadIdx.x; i <= (*CUDA_CC).Ncoef; i += lsize)
    {
        (*CUDA_LCC).cg[i] = CUDA_cg_first[i];
        //if(blockIdx.x == 0)
        //	printf("cg[%3d]: %10.7f\n", i, CUDA_cg_first[i]);
    }
    //printf("Idx: %d | m: %d | Ncoef: %d\n", x, m, (*CUDA_CC).Ncoef);
    //printf("cg[%d]: %.7f\n", x, CUDA_CC[x].cg[CUDA_CC[x].Ncoef + 1]);
    //printf("Idx: %d | beta_pole[%d]: %.7f\n", x, m, CUDA_CC[x].beta_pole[m]);

    for (int i = 1 + threadIdx.x; i <= (*CUDA_CC).Nphpar; i += lsize)
    {
        (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i] = (*CUDA_CC).par[i];
        //              ia[Ncoef+3+i] = ia_par[i]; moved to global
        //if (blockIdx.x == 0)
        //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + i, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i]);

    }

    /* the remaining cg entries and the scalar state: work-item 0 only (the
       indices are disjoint from the strided loops above) */
    if (threadIdx.x != 0)
        return;

    double period = ddiv(1.0, (*CUDA_LCC).freq);

    /* which of the initial poles this group runs (see ClCalculatePrepare) */
    const int m = blockIdx.x % N_POLES + 1;

    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = (*CUDA_CC).beta_pole[m];
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = (*CUDA_CC).lambda_pole[m];
    //if (blockIdx.x == 0)
    //{
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);
    //}
    //printf("cg[%d]: %.7f | cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1], (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);

    /* The formulas use beta measured from the pole */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = 90.0 - (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];
    //printf("90 - cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);

    /* conversion of lambda, beta to radians */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = DEG2RAD * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = DEG2RAD * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2];
    //printf("cg[%d]: %.7f | cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1], (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);

    /* Use omega instead of period */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3] = ddiv(24.0 * 2.0 * PI, period);

    //if (threadIdx.x == 0)
    //{
    //	printf("[%3d] cg[%3d]: %10.7f, period: %10.7f\n", blockIdx.x, (*CUDA_CC).Ncoef + 3, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3], period);
    //}

    /* Lommel-Seeliger part */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2] = 1;
    //if (blockIdx.x == 0)
    //{
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2]);
    //}

    /* Use logarithmic formulation for Lambert to keep it positive */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1] = log((*CUDA_CC).cl);
    //(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1] = (*CUDA_CC).logCl;   //log((*CUDA_CC).cl);


    //if (blockIdx.x == 0)
    //{
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1]);
    //}
    //printf("cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1]);

    /* Levenberg-Marquardt loop */
    // moved to global iter_max,iter_min,iter_dif_max
    //
    (*CUDA_LCC).rchisq = -1;
    (*CUDA_LCC).Alamda = -1;
    (*CUDA_LCC).Niter = 0;
    (*CUDA_LCC).iter_diff = 1e40;
    (*CUDA_LCC).dev_old = 1e30;
    (*CUDA_LCC).dev_new = 0;
    //	(*CUDA_LCC).Lastcall=0; always ==0
    (*CUDA_LFR).isReported = 0;
}

__kernel void ClCalculateIter1Begin(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_result* CUDA_FR,
    __global int* CUDA_End,
    int CUDA_n_iter_min,
    int CUDA_n_iter_max,
    double CUDA_iter_diff_max,
    double CUDA_Alamda_start,
    int n_contexts)
{
    int x = get_global_id(0);
    if (x >= n_contexts)
        return;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[x];

    if ((*CUDA_LCC).isInvalid)
    {
        return;
    }

    //                                   ?    < 50                                 ?       > 0                                   ?      < 0
    (*CUDA_LCC).isNiter = (((*CUDA_LCC).Niter < CUDA_n_iter_max) && ((*CUDA_LCC).iter_diff > CUDA_iter_diff_max)) || ((*CUDA_LCC).Niter < CUDA_n_iter_min);
    (*CUDA_FR).isNiter = (*CUDA_LCC).isNiter;

    //printf("[%d] isNiter: %d, Alamda: %10.7f\n", blockIdx.x, (*CUDA_LCC).isNiter, (*CUDA_LCC).Alamda);

    if ((*CUDA_LCC).isNiter)
    {
        if ((*CUDA_LCC).Alamda < 0)
        {
            (*CUDA_LCC).isAlamda = 1;
            (*CUDA_LCC).Alamda = CUDA_Alamda_start; /* initial alambda */
        }
        else
        {
            (*CUDA_LCC).isAlamda = 0;
        }
    }
    else
    {
        if (!(*CUDA_LFR).isReported)
        {
            //int oldEnd = *CUDA_End;
            //atomic_add(CUDA_End, 1);
            int t = *CUDA_End;
            atomic_inc(CUDA_End);

            //printf("[%d] t: %2d, Begin %2d\n", blockIdx.x, t, *CUDA_End);

            (*CUDA_LFR).isReported = 1;
        }
    }

    //if (threadIdx.x == 1)
    //	printf("[begin] Alamda: %10.7f\n", (*CUDA_LCC).Alamda);
    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST
}

__kernel void ClCalculateIter1Mrqcof1Start(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global double* scratch)
    //__global int* CUDA_End)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    //double* Area = &CUDA_mCC[0].Area;

    //if (blockIdx.x == 0)
    //	printf("[%d][%3d] [Mrqcof1Start]\n", blockIdx.x, threadIdx.x);
        //printf("isInvalid: %3d, isNiter: %3d, isAlamda: %3d\n", (*CUDA_LCC).isInvalid, (*CUDA_LCC).isNiter, (*CUDA_LCC).isAlamda);

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return; //>> 0

    // => mrqcof_start(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
    mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, scr + (*CUDA_CC).offAlpha, (*CUDA_LCC).beta);
}

__kernel void ClCalculateIter1Mrqcof1Matrix(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int lpoints,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx;
    blockIdx.x = get_group_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    __local int num; // __shared__

    int3 localIdx;
    localIdx.x = get_local_id(0);
    if (localIdx.x == 0)
    {
        num = 0;
    }

    mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, lpoints, num, scr);
}

/* mrqcof pass over one lightcurve, curve1 part. trial == 0: the current
   parameters (cg), skipped when the previous trial was rejected (isAlamda
   == 0); trial == 1: the trial parameters (atry). One kernel for both passes
   gives mrqcof_curve1 (and its bright()) a single call site, which the
   compiler then inlines - with two callers it could emit a real function
   call, costing the kernel ~248 VGPRs (occupancy 5 instead of 12). */
__kernel void ClCalculateIter1MrqcofCurve1(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global double* scratch,
    const int trial)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!trial && !(*CUDA_LCC).isAlamda) return;

    __local int num;  // __shared__
    __local double tmave[BLOCK_DIM];

    if (threadIdx.x == 0)
    {
        num = 0;
    }

    mrqcof_curve1(CUDA_LCC, CUDA_CC, trial ? (*CUDA_LCC).atry : (*CUDA_LCC).cg, tmave, inrel, lpoints, num, scr);
}

__kernel void ClCalculateIter1Mrqcof1Curve1Last(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    __local double res[BLOCK_DIM];

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof1Curve1Last\n");

    mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, scr + (*CUDA_CC).offAlpha, (*CUDA_LCC).beta, res, inrel, lpoints, scr);
    //if (threadIdx.x == 0)
    //{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++) {
    //		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    //}
}

/* mrqcof pass over one lightcurve, curve2 part (normal equations).
   trial == 0: into alpha/beta, skipped when isAlamda == 0; trial == 1: into
   covar/da. One kernel for both passes gives mrqcof_curve2 a single call
   site (see ClCalculateIter1MrqcofCurve1). */
__kernel void ClCalculateIter1MrqcofCurve2(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global double* scratch,
    const int trial)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!trial && !(*CUDA_LCC).isAlamda) return;

    /* OpenCL requires __local declarations at kernel scope */
    __local double dydaT[CURVE2_K][DYT_STRIDE];
    __local double tileS[5 * CURVE2_K];   /* s2w, dws, dy, coef, coef1 */

    mrqcof_curve2(CUDA_LCC, CUDA_CC,
        scr + (trial ? (*CUDA_CC).offCovar : (*CUDA_CC).offAlpha),
        trial ? (*CUDA_LCC).da : (*CUDA_LCC).beta,
        dydaT, tileS, tileS + CURVE2_K, tileS + 2 * CURVE2_K, tileS + 3 * CURVE2_K, inrel, lpoints, scr);
}

__kernel void ClCalculateIter1Mrqcof1End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof1End\n");


    double ochisq = mrqcof_end(CUDA_LCC, CUDA_CC, scr + (*CUDA_CC).offAlpha);
    if (threadIdx.x == 0)
        (*CUDA_LCC).Ochisq = ochisq;


    ////if (threadIdx.x == 0)
    ////{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++) {
    //	printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    ////}
}

__kernel void ClCalculateIter1Mrqmin1End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    /* runtime-sized by the host to Mfit1*Mfit1 doubles (~24 KB for real
       workunits) so the kernel also fits GCN's 32 KB local-memory limit */
    __local double* covL,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    //if (threadIdx.x == 0)
    //{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++)
    //	//{
    //		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    //}

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqmin1End\n");

    // gauss_err =
    //mrqmin_1_end(CUDA_LCC, CUDA_CC, sh_icol, sh_irow, sh_big, icol, pivinv);


    /* OpenCL requires __local declarations at kernel scope; the solver runs
       entirely in local memory (see gauss_errc.cl). covL comes in as a
       runtime-sized kernel argument. */
    __local double daL[DYT_STRIDE];
    __local int ipivL[DYT_STRIDE];
    __local double shBig[BLOCK_DIM];
    __local int shIrow[BLOCK_DIM];
    __local int shIcol[BLOCK_DIM];
    __local double pivBC[1];
    __local int icolBC[1];

    mrqmin_1_end(CUDA_LCC, CUDA_CC, covL, daL, ipivL, shBig, shIrow, shIcol, pivBC, icolBC, scr + (*CUDA_CC).offAlpha);

    //if (blockIdx.x == 0) {
    //	printf("[%3d] sh_icol[%3d]: %3d\n", threadIdx.x, threadIdx.x, sh_icol[threadIdx.x]);
    //}
}

__kernel void ClCalculateIter1Mrqcof2Start(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof2Start\n");


    //mrqcof_start(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
    mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, scr + (*CUDA_CC).offCovar, (*CUDA_LCC).da);

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("alpha[56]: %10.7f\n", (*CUDA_LCC).alpha[56]);
}

__kernel void ClCalculateIter1Mrqcof2Matrix(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int lpoints,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    __local int num; // __shared__

    int3 localIdx;
    localIdx.x = get_local_id(0);
    if (localIdx.x == 0)
    {
        num = 0;
    }

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof2Matrix\n");

    //mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).atry, lpoints);
    mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, lpoints, num, scr);
}

__kernel void ClCalculateIter1Mrqcof2Curve1Last(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    __local double res[BLOCK_DIM];

    //mrqcof_curve1_last(CUDA_LCC, CUDA_CC, dytemp, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, res, inrel, lpoints);
    mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, scr + (*CUDA_CC).offCovar, (*CUDA_LCC).da, res, inrel, lpoints, scr);
}

__kernel void ClCalculateIter1Mrqcof2End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    double chisq = mrqcof_end(CUDA_LCC, CUDA_CC, scr + (*CUDA_CC).offCovar);
    if (threadIdx.x == 0)
        (*CUDA_LCC).Chisq = chisq;

    //if (blockIdx.x == 0)
    //	printf("[%3d] Chisq: %10.7f\n", threadIdx.x, (*CUDA_LCC).Chisq);
}

__kernel void ClCalculateIter1Mrqmin2End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global double* scratch)
{
    __global double* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqmin2End\n");

    //mrqmin_2_end(CUDA_LCC, CUDA_ia, CUDA_ma);
    mrqmin_2_end(CUDA_LCC, CUDA_CC, scr);

    if (threadIdx.x == 0)
        (*CUDA_LCC).Niter++;

    //if (blockIdx.x == 0)
    //	printf("[%3d] Niter: %d\n", threadIdx.x, (*CUDA_LCC).Niter);
    //printf("|");
}

__kernel void ClCalculateIter2(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
{
    int i, j;
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid)
    {
        return;
    }

    //if (blockIdx.x == 0)
    //	printf("[%3d] isNiter: %d\n", threadIdx.x, (*CUDA_LCC).isNiter);

    if ((*CUDA_LCC).isNiter)
    {
        /* evaluated once, before anyone updates Ochisq: work-item 0 used to
           write Ochisq inside this branch while other wavefronts could still
           be evaluating the condition, which made the branch - and the
           barriers in it - divergent */
        const int improved = (*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq;
        if (improved)
        {
            int brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
            if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
            int brtmpl = threadIdx.x * brtmph;
            brtmph = brtmpl + brtmph;
            if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
            brtmpl++;

            curv(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, brtmpl, brtmph);
            /* work-item 0 sums the Area of every facet; this also orders every
               read of Ochisq above before the write below */
            barrier(CLK_GLOBAL_MEM_FENCE);

            if (threadIdx.x == 0)
            {
                (*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;

                for (i = 1; i <= 3; i++)
                {
                    (*CUDA_LCC).chck[i] = 0;


                    for (j = 1; j <= (*CUDA_CC).Numfac; j++)
                    {
                        double qq;
                        qq = (*CUDA_LCC).chck[i] + (*CUDA_LCC).Area[j] * (*CUDA_CC).Nor[j][i - 1];

                        //if (blockIdx.x == 0)
                        //	printf("[%d] [%d][%3d] qq: %10.7f, chck[%d]: %10.7f, Area[%3d]: %10.7f, Nor[%3d][%d]: %10.7f\n",
                        //		blockIdx.x, i, j, qq, i, (*CUDA_LCC).chck[i], j, (*CUDA_LCC).Area[j], j, i - 1, (*CUDA_CC).Nor[j][i - 1]);

                        (*CUDA_LCC).chck[i] = qq;
                    }

                    //if (blockIdx.x == 0)
                    //	printf("[%d] chck[%d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).chck[i]);
                }

                //printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3]);

                (*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - (pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0)) * pow((*CUDA_CC).conw_r, 2.0);
                //(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1] + (*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2] + (*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]) * ((*CUDA_CC).conw_r * (*CUDA_CC).conw_r);
            }
        }


        if (threadIdx.x == 0)
        {
            //if (blockIdx.x == 0)
            //	printf("ndata - 3: %3d\n", (*CUDA_CC).ndata - 3);

            (*CUDA_LCC).dev_new = sqrt(ddiv((*CUDA_LCC).rchisq, (double)((*CUDA_CC).ndata - 3)));

            //if (blockIdx.x == 233)
            //{
            //	double dev_best = (*CUDA_LCC).dev_new * (*CUDA_LCC).dev_new * ((*CUDA_CC).ndata - 3);
            //	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
            //		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LCC).dev_new, dev_best);
            //}

            // NOTE: only if this step is better than the previous, 1e-10 is for numeric errors
            if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
            {
                (*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
                (*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
            }
            //		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
        }

    }
}

__kernel void ClCalculateFinishPole(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global struct freq_result* CUDA_FR)
{
    int i;
    int3 blockIdx;
    blockIdx.x = get_group_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    double totarea = 0;
    for (i = 1; i <= (*CUDA_CC).Numfac; i++)
    {
        totarea = totarea + (*CUDA_LCC).Area[i];
    }

    //if(blockIdx.x == 2)
    //	printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f, conw_r: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3], (*CUDA_CC).conw_r);

    //if (blockIdx.x == 2)
    //	printf("rchisq: %10.7f, Chisq: %10.7f \n", (*CUDA_LCC).rchisq, (*CUDA_LCC).Chisq);

    //const double sum = pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0);
    const double sum = ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1]) + ((*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2]) + ((*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]);
    //printf("[FinishPole] [%d] sum: %10.7f\n", blockIdx.x, sum);

    const double dark = sqrt(sum);

    //if (blockIdx.x == 232 || blockIdx.x == 233)
    //	printf("[%d] sum: %12.8f, dark: %12.8f, totarea: %12.8f, dark_best: %12.8f\n", blockIdx.x, sum, dark, totarea, dark / totarea * 100);

    /* period solution */
    const double period = ddiv(2 * PI, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3]);

    /* pole solution */
    const double la_tmp = RAD2DEG * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2];

    //if (la_tmp < 0.0)
    //	printf("[CalculateFinishPole] la_best: %4.0f\n", la_tmp);

    const double be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];

    //if (blockIdx.x == 2)
        //printf("[%d] dev_new: %10.7f, dev_best: %10.7f\n", blockIdx.x, (*CUDA_LCC).dev_new, (*CUDA_LFR).dev_best);

    if ((*CUDA_LCC).dev_new < (*CUDA_LFR).dev_best)
    {
        (*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
        (*CUDA_LFR).dev_best_x2 = (*CUDA_LCC).rchisq;
        (*CUDA_LFR).per_best = period;
        (*CUDA_LFR).dark_best = ddiv(dark, totarea) * 100;
        (*CUDA_LFR).la_best = la_tmp < 0 ? la_tmp + 360.0 : la_tmp;
        (*CUDA_LFR).be_best = be_tmp;

        //printf("[%d] dev_best: %12.8f\n", blockIdx.x, (*CUDA_LFR).dev_best);

        //if (blockIdx.x == 232)
        //{
        //	double dev_best = (*CUDA_LFR).dev_best * (*CUDA_LFR).dev_best * ((*CUDA_CC).ndata - 3);
        //	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
        //		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LFR).dev_best, dev_best);
        //}
    }

    if (isnan((*CUDA_LFR).dark_best) == 1)
    {
        (*CUDA_LFR).dark_best = 1.0;
    }

    //if (blockIdx.x == 2)
    //	printf("dark_best: %10.7f \n", (*CUDA_LFR).dark_best);

    //debug
    /*	(*CUDA_LFR).dark=dark;
    (*CUDA_LFR).totarea=totarea;
    (*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
    (*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
    (*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
}
