#define POINTS_MAX         2000             /* max number of data points in one lc. */
#define MAX_N_OBS         20000             /* max number of data points */
#define MAX_LC              200             /* max number of lightcurves */
#define MAX_LINE_LENGTH    1000             /* max length of line in an input file */
#define MAX_N_FAC          1000             /* max number of facets */
#define MAX_N_ITER          100             /* maximum number of iterations */
#define MAX_N_PAR           200             /* maximum number of parameters */
#define MAX_LM               10             /* maximum degree and order of sph. harm. */
#define N_PHOT_PAR            5             /* maximum number of parameters in scattering  law */
#define TINY                  1e-8          /* precision parameter for mu, mu0*/
#define N_POLES              10             /* number of initial poles */

#define PI                 M_PI             /* 3.14159265358979323846 */
#define AU            149597870.691         /* Astronomical Unit [km] */
#define C_SPEED       299792458             /* speed of light [m/s]*/

#define DEG2RAD      (PI / 180)
#define RAD2DEG      (180 / PI)

#if defined INTEL
#define BLOCK_DIM 64
#else
#define BLOCK_DIM 128
#endif
#pragma OPENCL FP_CONTRACT ON

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

//struct __attribute__((packed)) freq_context
//struct mfreq_context
//struct __attribute__((aligned(8))) mfreq_context
//#ifdef NVIDIA
//struct mfreq_context
//#else
//typedef struct mfreq_context
//#endif
typedef struct mfreq_context
{
	//double* Area;
	//double* Dg;
	//double* alpha;
	//double* covar;
	//double* dytemp;
	//double* ytemp;

	double Area[MAX_N_FAC + 1];
	double Dg[(MAX_N_FAC + 1) * (MAX_N_PAR + 1)];
	double alpha[(MAX_N_PAR + 1) * (MAX_N_PAR + 1)];
	double covar[(MAX_N_PAR + 1) * (MAX_N_PAR + 1)];
	double dytemp[(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	double ytemp[POINTS_MAX + 1];

	double beta[MAX_N_PAR + 1];
	double atry[MAX_N_PAR + 1];
	double da[MAX_N_PAR + 1];
	double cg[MAX_N_PAR + 1];
	double Blmat[4][4];
	double Dblm[3][4][4];
	double jp_Scale[POINTS_MAX + 1];
	double jp_dphp_1[POINTS_MAX + 1];
	double jp_dphp_2[POINTS_MAX + 1];
	double jp_dphp_3[POINTS_MAX + 1];
	double e_1[POINTS_MAX + 1];
	double e_2[POINTS_MAX + 1];
	double e_3[POINTS_MAX + 1];
	double e0_1[POINTS_MAX + 1];
	double e0_2[POINTS_MAX + 1];
	double e0_3[POINTS_MAX + 1];
	double de[POINTS_MAX + 1][4][4];
	double de0[POINTS_MAX + 1][4][4];
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
//#ifdef NVIDIA
//struct freq_context
//#else
//typedef struct freq_context
//#endif
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
};

//struct freq_result
//struct __attribute__((aligned(8))) freq_result
//#ifdef NVIDIA
//struct freq_result
//#else
//typedef struct freq_result
//#endif
struct freq_result
{
	double dark_best, per_best, dev_best, dev_best_x2, la_best, be_best, freq;
	int isReported, isInvalid, isNiter;
};
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

		for (int k = 1; k <= n; k++)
		{
			// 290(1 + 1 * 289)    ...    867(288 + 2 * 289)
			int idx = i + k * (*CUDA_CC).Numfac1;
			(*CUDA_LCC).Dg[idx] = g * (*CUDA_CC).Dsph[i][k];

			//printf("Dg[%4d]: %.7f\n", i + k * (*CUDA_CC).Numfac1, (*CUDA_LCC).Dg[i + k * (*CUDA_CC).Numfac1]);

			//if (blockIdx.x == 0 && i == 1)
			//	printf("[%d] i: %d, n: %d, k: %d, Dg[%4d]: %.7f\n", blockIdx.x, i, n, k, idx, (*CUDA_LCC).Dg[idx]);

		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();
}

void mrqcof_curve2(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* alpha,
	__global double* beta,
	int inrel,
	int lpoints)
{
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

	/*   if ((*CUDA_LCC).Lastcall != 1) always ==0
		 {*/
	if (inrel /*==1*/)
	{
		for (jp = tmpl; jp <= tmph; jp++)
		{
			lnp1++;
			int ixx = jp + 1 * Lpoints1;
			/* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
			(*CUDA_LCC).dytemp[ixx] = 0;

			//if (blockIdx.x == 0)
			//	printf("[%d][%d] dytemp[%3d]: %10.7f\n", blockIdx.x, jp, ixx, (*CUDA_LCC).dytemp[ixx]);

			coef = (*CUDA_CC).Sig[lnp1] * lpoints / (*CUDA_LCC).ave;

			//if (threadIdx.x == 0)
			//	printf("[%d][%3d][%d] coef: %10.7f\n", blockIdx.x, threadIdx.x, jp, coef);

			double yytmp = (*CUDA_LCC).ytemp[jp];
			coef1 = yytmp / (*CUDA_LCC).ave;

			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("[Device | mrqcof_curve2_1] [%3d]  yytmp[%3d]: %10.7f, ave: %10.7f\n", threadIdx.x, jp, yytmp, (*CUDA_LCC).ave);

			(*CUDA_LCC).ytemp[jp] = coef * yytmp;

			//if (blockIdx.x == 0)
			//	printf("[Device][%d][%3d] ytemp[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).ytemp[jp]);

			ixx += Lpoints1;

			//if (threadIdx.x == 0)
			//	printf("[%3d] jp[%3d] dytemp[%3d]: %10.7f\n", blockIdx.x, jp, ixx, (*CUDA_LCC).dytemp[ixx]);

			for (l = 2; l <= (*CUDA_CC).ma; l++, ixx += Lpoints1)
			{
				(*CUDA_LCC).dytemp[ixx] = coef * ((*CUDA_LCC).dytemp[ixx] - coef1 * (*CUDA_LCC).dave[l]);

				//if (blockIdx.x == 0 && threadIdx.x == 0)
				//	printf("[Device | mrqcof_curve2_1] [%3d]  coef1: %10.7f, dave[%3d]: %10.7f, dytemp[%3d]: %10.7f\n",
				//		threadIdx.x, coef1, l, (*CUDA_LCC).dave[l], ixx, (*CUDA_LCC).dytemp[ixx]);
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np1 += lpoints;
	}

	lnp2 = (*CUDA_LCC).np2;
	ltrial_chisq = (*CUDA_LCC).trial_chisq;

	if ((*CUDA_CC).ia[1]) //not relative
	{
		for (jp = 1; jp <= lpoints; jp++)
		{
			ymod = (*CUDA_LCC).ytemp[jp];

			int ixx = jp + matmpl * Lpoints1;
			for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
				(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

			lnp2++;

			//xx = tex1Dfetch(texsig, lnp2);
			//sig2i = 1 / (__hiloint2double(xx.y, xx.x) * __hiloint2double(xx.y, xx.x));
			sig2i = 1 / ((*CUDA_CC).Sig[lnp2] * (*CUDA_CC).Sig[lnp2]);

			//xx = tex1Dfetch(texWeight, lnp2);
			//wght = __hiloint2double(xx.y, xx.x);
			wght = (*CUDA_CC).Weight[lnp2];

			//xx = tex1Dfetch(texbrightness, lnp2);
			//dy = __hiloint2double(xx.y, xx.x) - ymod;
			dy = (*CUDA_CC).Brightness[lnp2] - ymod;

			j = 0;
			//
			double sig2iwght = sig2i * wght;
			//
			for (l = 1; l <= (*CUDA_CC).lastone; l++)
			{
				j++;
				wt = (*CUDA_LCC).dyda[l] * sig2iwght;
				//				   k = 0;
				//precalc thread boundaries
				tmph = l / BLOCK_DIM;
				if (l % BLOCK_DIM) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				for (m = tmpl; m <= tmph; m++)
				{
					//				  k++;
					alpha[j * (*CUDA_CC).Mfit1 + m] = alpha[j * (*CUDA_CC).Mfit1 + m] + wt * (*CUDA_LCC).dyda[m];
				} /* m */
				barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
				if (threadIdx.x == 0)
				{
					beta[j] = beta[j] + dy * wt;
				}
				barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
			} /* l */
			for (; l <= (*CUDA_CC).lastma; l++)
			{
				if ((*CUDA_CC).ia[l])
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					//				   k = 0;

					for (m = latmpl; m <= latmph; m++)
					{
						//					  k++;
						alpha[j * (*CUDA_CC).Mfit1 + m] = alpha[j * (*CUDA_CC).Mfit1 + m] + wt * (*CUDA_LCC).dyda[m];
					} /* m */
					barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
					if (threadIdx.x == 0)
					{
						k = (*CUDA_CC).lastone;
						m = (*CUDA_CC).lastone + 1;
						for (; m <= l; m++)
						{
							if ((*CUDA_CC).ia[m])
							{
								k++;
								alpha[j * (*CUDA_CC).Mfit1 + k] = alpha[j * (*CUDA_CC).Mfit1 + k] + wt * (*CUDA_LCC).dyda[m];
							}
						} /* m */
						beta[j] = beta[j] + dy * wt;
					}
					barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
				}
			} /* l */
			ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
		} /* jp */
	}
	else //relative ia[1]==0
	{

		//if (threadIdx.x == 0)
		//	printf("[%d] lastone: %3d\n", blockIdx.x, (*CUDA_CC).lastone);

		for (jp = 1; jp <= lpoints; jp++)
		{
			ymod = (*CUDA_LCC).ytemp[jp];

			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("Curve2_2b >>> [%3d][%3d] jp[%3d] ymod: %10.7f\n", blockIdx.x, threadIdx.x, jp, ymod);

			int ixx = jp + matmpl * Lpoints1;
			for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
			{
				(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];  // jp[1] dytemp[315] 0.0 - ?!?  must be -1051420.6747227

				//if (blockIdx.x == 0 && threadIdx.x == 1 && jp == 1)
				//	printf("[%2d][%3d] dytemp[%d]: %10.7f\n", blockIdx.x, jp, ixx, (*CUDA_LCC).dytemp[ixx]);
			}
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

			lnp2++;

			//xx = tex1Dfetch(texsig, lnp2);
			//sig2i = 1 / (__hiloint2double(xx.y, xx.x) * __hiloint2double(xx.y, xx.x));
			sig2i = 1 / ((*CUDA_CC).Sig[lnp2] * (*CUDA_CC).Sig[lnp2]);

			//xx = tex1Dfetch(texWeight, lnp2);
			//wght = __hiloint2double(xx.y, xx.x);
			wght = (*CUDA_CC).Weight[lnp2];

			//xx = tex1Dfetch(texbrightness, lnp2);
			//dy = __hiloint2double(xx.y, xx.x) - ymod;
			dy = (*CUDA_CC).Brightness[lnp2] - ymod;

			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("Curve2_2b >>> [%3d][%3d] jp[%3d] sig2i: %10.7f, wght: %10.7f, dy: %10.7f\n", blockIdx.x, threadIdx.x, jp, sig2i, wght, dy);  // dy - ?

			j = 0;
			//
			double sig2iwght = sig2i * wght;
			//l==1
			//
			for (l = 2; l <= (*CUDA_CC).lastone; l++)
			{

				j++;
				wt = (*CUDA_LCC).dyda[l] * sig2iwght; // jp[1]  dyda[2] == 0    - ?!? must be -1051420.6747227   *) See dytemp[]
													  // jp 2, dyda[9] == 0 - ?!? must be 7.9447669

				//if (blockIdx.x == 0 && threadIdx.x == 1 && jp == 1 && j == 1)
				//	printf("[%2d][%2d] jp[%3d] j[%3d] wt: %10.7f, dyda[%d]: %10.7f, sig2iwght: %10.7f\n",
				//		blockIdx.x, threadIdx.x, jp, j, wt, l, (*CUDA_LCC).dyda[l], sig2iwght);

				//				   k = 0;
				//precalc thread boundaries
				tmph = l / BLOCK_DIM;
				if (l % BLOCK_DIM) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				//m==1
				if (tmpl == 1) tmpl++;
				//
				for (m = tmpl; m <= tmph; m++)
				{
					//if (blockIdx.x == 0)
					//	printf("[%3d] tmpl: %3d, tmph: %3d\n", threadIdx.x, tmpl, tmph);
					//if (blockIdx.x == 0 && threadIdx.x == 1)
					//	printf(".");
					//					  k++;
					alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + wt * (*CUDA_LCC).dyda[m];

					//int qq = j * (*CUDA_CC).Mfit1 + m - 1;											// After the "_" in  Mrqcof1Curve2 "wt" & "dyda[2]" has ZEROES - ?!?
					//if (blockIdx.x == 0 && threadIdx.x == 1 && l == 2) // j == 1 like l = 2
					//	printf("curv2_2b>>>> [%2d][%3d] l[%3d] jp[%3d] alpha[%4d]: %10.7f, wt: %10.7f, dyda[%3d]: %10.7f\n",
					//		blockIdx.x, threadIdx.x, l, jp, qq, (*CUDA_LCC).alpha[qq], wt, m, (*CUDA_LCC).dyda[m]);
				} /* m */
				barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
				if (threadIdx.x == 0)
				{
					beta[j] = beta[j] + dy * wt;
				}
				barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
			} /* l */
			for (; l <= (*CUDA_CC).lastma; l++)
			{

				if ((*CUDA_CC).ia[l])
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					//				   k = 0;

					tmpl = latmpl;
					//m==1
					if (tmpl == 1) tmpl++;
					//
					for (m = tmpl; m <= latmph; m++)
					{
						//k++;
						alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + wt * (*CUDA_LCC).dyda[m];
					} /* m */
					barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
					if (threadIdx.x == 0)
					{
						k = (*CUDA_CC).lastone - 1;
						m = (*CUDA_CC).lastone + 1;
						for (; m <= l; m++)
						{
							if ((*CUDA_CC).ia[m])
							{
								k++;
								alpha[j * (*CUDA_CC).Mfit1 + k] = alpha[j * (*CUDA_CC).Mfit1 + k] + wt * (*CUDA_LCC).dyda[m];
							}
						} /* m */
						beta[j] = beta[j] + dy * wt;
					}
					barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
				}
			} /* l */
			ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
		} /* jp */
	}
	//     } always ==0 // Lastcall != 1

	 // if (((*CUDA_LCC).Lastcall == 1) && (CUDA_Inrel[i] == 1)) always ==0
		//(*CUDA_LCC).Sclnw[i] = (*CUDA_LCC).Scale * CUDA_Lpoints[i] * CUDA_sig[np]/ave;

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
	int num)
{
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

		alpha = acos(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3);


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] alpha[%3d]: %.7f, cg[%3d]: %10.7f\n", jp, alpha, q, (*CUDA_LCC).cg[q]);

		/* Exp-lin model (const.term=1.) */
		double f = exp(-alpha / cg[(*CUDA_CC).Ncoef0 + 2]);	//f is temp here

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] [%2d][%3d] jp[%3d] f: %10.7f, cg[%3d] %10.7f, alpha %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, f, (*CUDA_CC).Ncoef0 + 2, cg[(*CUDA_CC).Ncoef0 + 2], alpha);

		(*CUDA_LCC).jp_Scale[jp] = 1 + cg[(*CUDA_CC).Ncoef0 + 1] * f + (cg[(*CUDA_CC).Ncoef0 + 3] * alpha);
		(*CUDA_LCC).jp_dphp_1[jp] = f;
		(*CUDA_LCC).jp_dphp_2[jp] = cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha / (cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
		(*CUDA_LCC).jp_dphp_3[jp] = alpha;

		//if (blockIdx.x == 0)
		//	printf("[neo] [%d][%3d] jp_Scale[%3d]: %10.7f, jp_dphp_1[]: %10.7F, jp_dphp_2[]: %10.7f, jp_dphp_3[]: %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).jp_Scale[jp], (*CUDA_LCC).jp_dphp_1[jp], (*CUDA_LCC).jp_dphp_2[jp], (*CUDA_LCC).jp_dphp_3[jp]);

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

		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_1[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_1[jp] = pom0 + tmat * ee0_3;

		//if (blockIdx.x == 0)
		//	printf("[%3d] jp[%3d] %10.7f, %10.7f\n", threadIdx.x, jp, (*CUDA_LCC).e_1[jp], (*CUDA_LCC).e0_1[jp]);

		tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_2[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_2[jp] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 1 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 1 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 1 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_3[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_3[jp] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][1][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][1] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][1][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][2] = pom0 + tmat * ee0_3;

		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][1][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][3] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][2][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][1] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][2][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][2] = pom0 + tmat * ee0_3;

		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][2][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][3] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[1][1][1] + 0 * (*CUDA_LCC).Dblm[1][2][1] + 1 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][2] + 0 * (*CUDA_LCC).Dblm[1][2][2] + 1 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][3] + 0 * (*CUDA_LCC).Dblm[1][2][3] + 1 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][3][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][1] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[2][1][1] + 0 * (*CUDA_LCC).Dblm[2][2][1] + 1 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][2] + 0 * (*CUDA_LCC).Dblm[2][2][2] + 1 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][3] + 0 * (*CUDA_LCC).Dblm[2][2][3] + 1 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][3][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][2] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][3][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][3] = pom0 + tmat * ee0_3;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);  //__syncthreads();
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int jp,
	int Lpoints1,
	int Inrel)
{
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
	e_1 = (*CUDA_LCC).e_1[jp];
	e_2 = (*CUDA_LCC).e_2[jp];
	e_3 = (*CUDA_LCC).e_3[jp];
	e0_1 = (*CUDA_LCC).e0_1[jp];
	e0_2 = (*CUDA_LCC).e0_2[jp];
	e0_3 = (*CUDA_LCC).e0_3[jp];
	de[1][1] = (*CUDA_LCC).de[jp][1][1];
	de[1][2] = (*CUDA_LCC).de[jp][1][2];
	de[1][3] = (*CUDA_LCC).de[jp][1][3];
	de[2][1] = (*CUDA_LCC).de[jp][2][1];
	de[2][2] = (*CUDA_LCC).de[jp][2][2];
	de[2][3] = (*CUDA_LCC).de[jp][2][3];
	de[3][1] = (*CUDA_LCC).de[jp][3][1];
	de[3][2] = (*CUDA_LCC).de[jp][3][2];
	de[3][3] = (*CUDA_LCC).de[jp][3][3];
	de0[1][1] = (*CUDA_LCC).de0[jp][1][1];
	de0[1][2] = (*CUDA_LCC).de0[jp][1][2];
	de0[1][3] = (*CUDA_LCC).de0[jp][1][3];
	de0[2][1] = (*CUDA_LCC).de0[jp][2][1];
	de0[2][2] = (*CUDA_LCC).de0[jp][2][2];
	de0[2][3] = (*CUDA_LCC).de0[jp][2][3];
	de0[3][1] = (*CUDA_LCC).de0[jp][3][1];
	de0[3][2] = (*CUDA_LCC).de0[jp][3][2];
	de0[3][3] = (*CUDA_LCC).de0[jp][3][3];

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

	j = 1;
	for (i = 1; i <= (*CUDA_CC).Numfac; i++, j++)
	{
		lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
		lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];

		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + cls / dnom);
			ar = (*CUDA_LCC).Area[j];
			br += ar * s;

			incl[incl_count] = i;
			dbr[incl_count] = (*CUDA_CC).Darea[i] * s;
			incl_count++;

			double lmu0_dnom = lmu0 / dnom;
			dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
			double lmu_dnom = lmu / dnom;
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
			tmp5 += ar * lmu * lmu0 / (lmu + lmu0);
		}
	}

	Scale = (*CUDA_LCC).jp_Scale[jp];
	i = jp + (ncoef0 - 3 + 1) * Lpoints1;
	/* Ders. of brightness w.r.t. rotation parameters */
	(*CUDA_LCC).dytemp[i] = Scale * tmp1;

	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp2;
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp3;

	i += Lpoints1;
	/* Ders. of br. w.r.t. phase function params. */
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_1[jp];
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_2[jp];
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_3[jp];

	/* Ders. of br. w.r.t. cl, cls */
	(*CUDA_LCC).dytemp[jp + (ncoef - 1) * (Lpoints1)] = Scale * tmp4 * cl;
	(*CUDA_LCC).dytemp[jp + (ncoef) * (Lpoints1)] = Scale * tmp5;

	/* Scaled brightness */
	(*CUDA_LCC).ytemp[jp] = br * Scale;

	ncoef0 -= 3;
	int m, m1, mr, iStart;
	int d, d1, dr;
	if (Inrel)
	{
		iStart = 2;
		//m = blockIdx.x * CUDA_Dg_block + 2 * (CUDA_Numfac1);
		m = 2 * (*CUDA_CC).Numfac1;
		d = jp + 2 * (Lpoints1);
	}
	else
	{
		iStart = 1;
		//m = blockIdx.x * CUDA_Dg_block + (CUDA_Numfac1);
		m = (*CUDA_CC).Numfac1;
		d = jp + (Lpoints1);
	}

	m1 = m + (*CUDA_CC).Numfac1;
	mr = 2 * (*CUDA_CC).Numfac1;
	d1 = d + Lpoints1;
	dr = 2 * Lpoints1;

	/* Derivatives of brightness w.r.t. g-coeffs */
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += 2, m += mr, m1 += mr, d += dr, d1 += dr)
		{
			double tmp = 0, tmp1 = 0;
			double l_dbr = dbr[0];
			int l_incl = incl[0];
			tmp = l_dbr * (*CUDA_LCC).Dg[m + l_incl];
			if ((i + 1) <= ncoef0)
			{
				tmp1 = l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
			}

			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				int l_incl = incl[j];
				tmp += l_dbr * (*CUDA_LCC).Dg[m + l_incl];
				if ((i + 1) <= ncoef0)
				{
					tmp1 += l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
				}
			}

			(*CUDA_LCC).dytemp[d] = Scale * tmp;
			if ((i + 1) <= ncoef0)
			{
				(*CUDA_LCC).dytemp[d1] = Scale * tmp1;
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d += Lpoints1)
			(*CUDA_LCC).dytemp[d] = 0;
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
	int tmpl,
	int tmph,
	int brtmpl,
	int brtmph)
{
	int i, j, k;
	double tmp = 0.0;
	double dtmp;
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
	//parallel reduction end
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	//int m = blockIdx.x * (*CUDA_CC).Dg_block + tmpl * (*CUDA_CC).Numfac1);   // <<<<<<<<<<<<<<<<<<<<<<<<<<<<< !!!
	int m = tmpl * (*CUDA_CC).Numfac1;
	for (j = tmpl; j <= tmph; j++)  //, m += (*CUDA_CC).Numfac1)
	{
		// printf("m: %4d\n", m);
		dtmp = 0;
		if (j <= (*CUDA_CC).Ncoef)
		{
			int mm = m + 1;
			for (i = 1; i <= (*CUDA_CC).Numfac; i++, mm++)
			{
				// dtmp += CUDA_Darea[i] * CUDA_Dg[mm] * CUDA_Nor[i][nc];
				dtmp += (*CUDA_CC).Darea[i] * (*CUDA_LCC).Dg[mm] * (*CUDA_CC).Nor[i][nc];

				//if (blockIdx.x == 0 && j == 8)
				//	printf("[%d][%3d]  Darea[%4d]: %.7f, Dg[%4d]: %.7f, Nor[%3d][%3d]: %10.7f\n",
				//		blockIdx.x, threadIdx.x, i, (*CUDA_CC).Darea[i], mm, (*CUDA_LCC).Dg[mm], i, nc, (*CUDA_CC).Nor[i][nc]);
			}
		}

		(*CUDA_LCC).dyda[j] = dtmp;

		//if (blockIdx.x == 0) // && threadIdx.x == 1)
		//    printf("[mrqcof_curve1_last -> conv] [%d][%3d] jp - 1: %3d, j[%3d] dyda[%3d]: %10.7f\n",
		//        blockIdx.x, threadIdx.x, nc, j, j, (*CUDA_LCC).dyda[j]);
	}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

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

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads(); //pro jistotu

	//int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%d][%3d] cg[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, q, (*CUDA_LCC).cg[q]);


}

void mrqcof_matrix(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int Lpoints,
	int num)
{
	matrix_neo(CUDA_LCC, CUDA_CC, cg, (*CUDA_LCC).np, Lpoints, num);
}

void mrqcof_curve1(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	__local double* tmave,
	int Inrel,
	int Lpoints,
	int num)
{
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

	for (jp = brtmpl; jp <= brtmph; jp++)
	{
			/*  ---  BRIGHT  ---  */
		bright(CUDA_LCC, CUDA_CC, cg, jp, Lpoints1, Inrel);
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
		ixx = tmpl * Lpoints1;

		for (int l = tmpl; l <= tmph; l++)
		{
			//jp==1
			ixx++;
			(*CUDA_LCC).dave[l] = (*CUDA_LCC).dytemp[ixx];

			//jp>=2
			ixx++;
			for (int jp = 2; jp <= Lpoints; jp++, ixx++)
			{
				//(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];

				//if (threadIdx.x == 1)
				//	printf("[Device | mrqcof_curv1] [%3d] dytemp[%3d]: %10.7f, dave[%3d]: %10.7f\n", blockIdx.x, ixx, (*CUDA_LCC).dytemp[ixx], l, (*CUDA_LCC).dave[l]);
			}
		}

		tmave[threadIdx.x] = 0;
		for (int jp = brtmpl; jp <= brtmph; jp++)
		{
			tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];
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
	int Lpoints)
{
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

	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;
		// *--- CONV() ---* //
		ymod = conv(CUDA_LCC, CUDA_CC, res, jp - 1, tmpl, tmph, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).ytemp[jp] = ymod;

			if (Inrel == 1)
				lave = lave + ymod;
		}
		for (l = tmpl; l <= tmph; l++)
		{
			//(*CUDA_LCC).dytemp[jp + l * (Lpoints + 1)] = (*CUDA_LCC).dyda[l];
			(*CUDA_LCC).dytemp[jp + l * (Lpoints + 1)] = (*CUDA_LCC).dyda[l];

			if (Inrel == 1)
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dyda[l];
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

	for (int j = 2; j <= (*CUDA_CC).Mfit; j++)
	{
		for (k = 1; k <= j - 1; k++)
		{
			alpha[k * (*CUDA_CC).Mfit1 + j] = alpha[j * (*CUDA_CC).Mfit1 + k];
			//if (blockIdx.x ==0 && threadIdx.x == 0)
			//	printf("[mrqcof_end] [%d][%3d] alpha[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, k * (*CUDA_CC).Mfit1 + j, alpha[k * (*CUDA_CC).Mfit1 + j]);
		}
	}

	return (*CUDA_LCC).trial_chisq;
}

//int gauss_errc(freq_context* CUDA_LCC, const int ma)
//mrqmin_1_end(CUDA_LCC, CUDA_ma, CUDA_mfit, CUDA_mfit1, block);
//int gauss_errc(struct mfreq_context* CUDA_LCC, struct freq_context* CUDA_CC, int* sh_icol, int* sh_irow, double* sh_big, int icol, double pivinv)
int gauss_errc(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC)
{
	//__shared__ int icol;
	//__shared__ double pivinv;
	//__shared__ int sh_icol[CUDA_BLOCK_DIM];
	//__shared__ int sh_irow[CUDA_BLOCK_DIM];
	//__shared__ double sh_big[CUDA_BLOCK_DIM];

	double big, dum, temp;
	double tmpSwap;
	int i, licol = 0, irow = 0, j, k, l, ll;
	int n = (*CUDA_CC).Mfit; // 54
	int m = (*CUDA_CC).ma;   // 57

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = n / BLOCK_DIM;
	if (n % BLOCK_DIM) brtmph++;		// 1 (thr 1)
	brtmpl = threadIdx.x * brtmph;		// 0
	brtmph = brtmpl + brtmph;			// 1
	if (brtmph > n) brtmph = n;			// false | 1
	brtmpl++;							// 1

	// <<< GausErrorCPre
	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++) (*CUDA_LCC).ipiv[j] = 0;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	// >>> GausErrorCPre End

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("brtmpl: %3d, brtmph: %3d\n", brtmpl, brtmph);

	// <<< GausErrorC
	for (i = 1; i <= n; i++)
	{
		big = 0;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)  // 1 to 1 on thread 0 first pass for all "i"
		{
			//if (threadIdx.x == 0 && i == 2)
			//	printf("[%d][%3d] ipiv[%3d]: %5d, covar[%3d]: %10.7f\n",
			//		blockIdx.x, threadIdx.x, j, (*CUDA_LCC).ipiv[j], j * (*CUDA_CC).Mfit1 + 1, (*CUDA_LCC).covar[j * (*CUDA_CC).Mfit1 + 1]);

			if ((*CUDA_LCC).ipiv[j] != 1)
			{
				//if (blockIdx.x == 0)
				//	printf("[%3d] i[%3d] ipiv[%3d]: %10.7f\n", threadIdx.x, i, j, (*CUDA_LCC).ipiv[j]);

				int ixx = j * (*CUDA_CC).Mfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if ((*CUDA_LCC).ipiv[k] == 0)
					{
						double tmpcov = fabs((*CUDA_LCC).covar[ixx]);
						if (tmpcov >= big)
						{
							//if (blockIdx.x == 0)
							//	printf("[%3d] i[%3d] ipiv[%3d]: %3d, ipiv[%3d]: %3d, big: %10.7f, tmpcov: %10.7f, covar[%3d]: %10.7f\n",
							//		threadIdx.x, i, j, (*CUDA_LCC).ipiv[j], k, (*CUDA_LCC).ipiv[k], big, tmpcov, ixx, (*CUDA_LCC).covar[ixx]);

							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if ((*CUDA_LCC).ipiv[k] > 1)
					{
						//printf("-");
						barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
						/*					        deallocate_vector((void *) ipiv);
												deallocate_vector((void *) indxc);
												deallocate_vector((void *) indxr);*/
						return(1);
					}
				}
			}
		}
		(*CUDA_LCC).sh_big[threadIdx.x] = big;
		(*CUDA_LCC).sh_irow[threadIdx.x] = irow;
		(*CUDA_LCC).sh_icol[threadIdx.x] = licol;

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		//int d = (*CUDA_LCC).sh_icol[0];
		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[%3d][%3d] i: %3d, licol: %3d\n", blockIdx.x, threadIdx.x, i, licol);
		//	//printf("[%3d][%3d] i: %3d, sh_col[%3d]: %d, d: %3d\n", blockIdx.x, threadIdx.x, i, threadIdx.x, (*CUDA_LCC).sh_icol[threadIdx.x], d);

		if (threadIdx.x == 0)
		{
			big = (*CUDA_LCC).sh_big[0];				// = 0
			(*CUDA_LCC).icol = (*CUDA_LCC).sh_icol[0];	// = 0
			irow = (*CUDA_LCC).sh_irow[0];				// = 0

			for (j = 1; j < BLOCK_DIM; j++)				// 1..127
			{
				//if (blockIdx.x == 0 && i == 1)
				//	printf("sh_big[%3d]: %10.7f\n", j, (*CUDA_LCC).sh_big[j]);

				if ((*CUDA_LCC).sh_big[j] >= big)
				{
					big = (*CUDA_LCC).sh_big[j];
					irow = (*CUDA_LCC).sh_irow[j];
					(*CUDA_LCC).icol = (*CUDA_LCC).sh_icol[j];
				}
			}

			//(*CUDA_LCC).ipiv[(*CUDA_LCC).icol] = ++(*CUDA_LCC).ipiv[(*CUDA_LCC).icol];
			++(*CUDA_LCC).ipiv[(*CUDA_LCC).icol];

			//if (blockIdx.x == 0)
			//	printf("i: %2d, icol: %3d, irow: %3d, ipiv[%3d]: %3d\n", i, (*CUDA_LCC).icol, irow, (*CUDA_LCC).icol, (*CUDA_LCC).ipiv[(*CUDA_LCC).icol]);


			if (irow != (*CUDA_LCC).icol) // what is going on here ???
			{
				//if (blockIdx.x == 0)
				//	printf("irow: %3d\n", irow);
				for (l = 1; l <= n; l++)
				{
					//SwapDouble((*CUDA_LCC).covar[irow * (*CUDA_CC).Mfit1 + l], (*CUDA_LCC).covar[icol * (*CUDA_CC).Mfit1 + l]);
					tmpSwap = (*CUDA_LCC).covar[irow * (*CUDA_CC).Mfit1 + l];
					(*CUDA_LCC).covar[irow * (*CUDA_CC).Mfit1 + l] = (*CUDA_LCC).covar[(*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + l];
					(*CUDA_LCC).covar[(*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + l] = tmpSwap;

				}

				//SwapDouble((*CUDA_LCC).da[irow], (*CUDA_LCC).da[icol]);
				tmpSwap = (*CUDA_LCC).da[irow];
				(*CUDA_LCC).da[irow] = (*CUDA_LCC).da[(*CUDA_LCC).icol];
				(*CUDA_LCC).da[(*CUDA_LCC).icol] = tmpSwap;

				//SWAP(b[irow],b[icol])
			}

			(*CUDA_LCC).indxr[i] = irow;
			(*CUDA_LCC).indxc[i] = (*CUDA_LCC).icol;

			//if (blockIdx.x == 0)
			//	printf("i: %3d, irow: %3d, icol: %3d\n", i, irow, (*CUDA_LCC).icol);

			int covarIdx = (*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + (*CUDA_LCC).icol;

			if ((*CUDA_LCC).covar[covarIdx] == 0.0)
			{
				j = 0;
				for (int l = 1; l <= (*CUDA_CC).ma; l++)
				{
					if ((*CUDA_CC).ia[l])
					{
						j++;
						(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
					}
				}

				return(2);
			}

			//<<<<<<<<<<  (*CUDA_LCC).
			(*CUDA_LCC).pivinv = 1.0 / (*CUDA_LCC).covar[covarIdx];
			(*CUDA_LCC).covar[covarIdx] = 1.0;


			(*CUDA_LCC).da[(*CUDA_LCC).icol] = (*CUDA_LCC).da[(*CUDA_LCC).icol] * (*CUDA_LCC).pivinv;
			//b[icol] *= pivinv;

			//if(blockIdx.x == 0)
			//	printf("[%d] i[%2d] da[%4d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).icol, (*CUDA_LCC).da[(*CUDA_LCC).icol]); // da - OK

			//if (blockIdx.x == 0)
			//	printf("[%d] i[%2d] pivinv: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).pivinv); // pivinv - OK

		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[%d] icol: %5d, mfit1: %3d, l: %3d\n", blockIdx.x, icol, (*CUDA_CC).Mfit1, l);

		for (l = brtmpl; l <= brtmph; l++)
		{
			int qq = (*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + l;
			double covar1 = (*CUDA_LCC).covar[qq] * (*CUDA_LCC).pivinv;
			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("[%d][%3d] i[%3d] l[%3d] icol: %3d, pivinv: %10.7f, covar[%4d]: %10.7f, covar: %10.7f\n",
			//		blockIdx.x, threadIdx.x, i, l, (*CUDA_LCC).icol, (*CUDA_LCC).pivinv, qq, (*CUDA_LCC).covar[qq], covar1);

			//barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);// | CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

			//covar[qq] = 1.0;
			//(*CUDA_LCC).covar[qq] = (*CUDA_LCC).covar[qq] * pivinv;
			(*CUDA_LCC).covar[qq] = covar1;
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		for (ll = brtmpl; ll <= brtmph; ll++)
		{
			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("i[%d%3d] ll: %4d, brtmpl: %3d, brtmph; %3d\n", i, ll, brtmpl, brtmph);

			if (ll != (*CUDA_LCC).icol)
			{
				int ixx = ll * (*CUDA_CC).Mfit1;
				int jxx = (*CUDA_LCC).icol * (*CUDA_CC).Mfit1;
				dum = (*CUDA_LCC).covar[ixx + (*CUDA_LCC).icol];
				(*CUDA_LCC).covar[ixx + (*CUDA_LCC).icol] = 0.0;
				ixx++;
				jxx++;
				for (l = 1; l <= n; l++, ixx++, jxx++)
				{
					(*CUDA_LCC).covar[ixx] -= (*CUDA_LCC).covar[jxx] * dum;
				}

				(*CUDA_LCC).da[ll] -= (*CUDA_LCC).da[(*CUDA_LCC).icol] * dum;
				//b[ll] -= b[icol]*dum;

			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	// << GausErrorCPost
	if (threadIdx.x == 0)
	{
		for (l = n; l >= 1; l--)
		{
			if ((*CUDA_LCC).indxr[l] != (*CUDA_LCC).indxc[l])
			{
				for (k = 1; k <= n; k++)
				{
					//SwapDouble((*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxr[l]], (*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxc[l]]);
					tmpSwap = (*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxr[l]];
					(*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxr[l]] = (*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxc[l]];
					(*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxc[l]] = tmpSwap;
				}
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return(0);
	// >>> GaussErrorCPost END
}
// #undef SWAP
 //from Numerical Recipes

//N.B. The foll. L-M routines are modified versions of Press et al.
//  converted from Mikko's fortran code

//  8.11.2006


int mrqmin_1_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC)
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
	// >>> Iter1Mrqmin1EndPre1 END

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	// <<< Iter1Mrqmin1EndPre2
	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * (*CUDA_CC).Mfit1 + 1;
		for (int k = 1; k <= (*CUDA_CC).Mfit; k++, ixx++)
		{
			(*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];
		}

		int qq = j * (*CUDA_CC).Mfit1 + j;
		(*CUDA_LCC).covar[qq] = (*CUDA_LCC).alpha[qq] * (1 + (*CUDA_LCC).Alamda);
		(*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	// >>> Iter1Mrqmin1EndPre2 END

	// <<< gauss_errc    ---- GAUS ERROR CODE ----
	int err_code = gauss_errc(CUDA_LCC, CUDA_CC);
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

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	// <<< Iter1Mrqmin1EndPost END

	return err_code;
}

void mrqmin_2_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC) //, int* ia, int ma)
{
	int j, k, l;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		(*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / (*CUDA_CC).Alamda_incr;
		for (j = 1; j <= (*CUDA_CC).Mfit; j++)
		{
			for (k = 1; k <= (*CUDA_CC).Mfit; k++)
			{
				(*CUDA_LCC).alpha[j * (*CUDA_CC).Mfit1 + k] = (*CUDA_LCC).covar[j * (*CUDA_CC).Mfit1 + k];

				//if (blockIdx.x == 0)
				//	printf("alpha[%3d]: %10.7f\n", (*CUDA_LCC).alpha[j * (*CUDA_CC).Mfit1 + k]);
			}

			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		}
		for (l = 1; l <= (*CUDA_CC).ma; l++)
		{
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
		}
	}
	else
	{
		(*CUDA_LCC).Alamda = (*CUDA_CC).Alamda_incr * (*CUDA_LCC).Alamda;
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}


}
kernel void k(){}
kernel void ClCheckEnd(
    __global int* CUDA_End,
    int theEnd)
{
    int3 blockIdx;
    blockIdx.x = get_group_id(0);

    if (blockIdx.x == 0)
        *CUDA_End = theEnd;

    //if (blockIdx.x == 0)
        //printf("CheckEnd CUDA_End: %2d\n", *CUDA_End);

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

    int n = n_start + blockIdx.x;


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

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    //if (blockIdx.x == 0)
        //printf("Prepare CUDA_End: %2d\n", *CUDA_End);
}

__kernel void ClCalculatePreparePole(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global struct freq_result* CUDA_FR,
    __global double* CUDA_cg_first,
    __global int* CUDA_End,
    __global struct freq_context* CUDA_CC2,
    //double CUDA_cl,
    int m)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

    //int t = *CUDA_End;
    //*CUDA_End = 13;
    //printf("[%d] PreparePole t: %d, CUDA_End: %d\n", x, t, *CUDA_End);


    if ((*CUDA_LCC).isInvalid)
    {
        //atomic_add(CUDA_End, 1);
        atomic_inc(CUDA_End);
        //printf("prepare pole %d ", (*CUDA_End));

        (*CUDA_FR).isReported = 0; //signal not to read result

        //printf("[%d] isReported: %d \n", blockIdx.x, (*CUDA_FR).isReported);

        return;
    }

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("[Device] PreparePole > ma: %d\n", (*CUDA_CC).ma);

    double period = 1.0 / (*CUDA_LCC).freq;

    //* starts from the initial ellipsoid */
    for (int i = 1; i <= (*CUDA_CC).Ncoef; i++)
    {
        (*CUDA_LCC).cg[i] = CUDA_cg_first[i];
        //if(blockIdx.x == 0)
        //	printf("cg[%3d]: %10.7f\n", i, CUDA_cg_first[i]);
    }
    //printf("Idx: %d | m: %d | Ncoef: %d\n", x, m, (*CUDA_CC).Ncoef);
    //printf("cg[%d]: %.7f\n", x, CUDA_CC[x].cg[CUDA_CC[x].Ncoef + 1]);
    //printf("Idx: %d | beta_pole[%d]: %.7f\n", x, m, CUDA_CC[x].beta_pole[m]);

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
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3] = 24.0 * 2.0 * PI / period;

    //if (threadIdx.x == 0)
    //{
    //	printf("[%3d] cg[%3d]: %10.7f, period: %10.7f\n", blockIdx.x, (*CUDA_CC).Ncoef + 3, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3], period);
    //}

    for (int i = 1; i <= (*CUDA_CC).Nphpar; i++)
    {
        (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i] = (*CUDA_CC).par[i];
        //              ia[Ncoef+3+i] = ia_par[i]; moved to global
        //if (blockIdx.x == 0)
        //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + i, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i]);

    }

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

    if (blockIdx.x == 0)
    {
        for (int i = 0; i < MAX_N_OBS + 1; i++)
        {
            //printf("[%d] %g", blockIdx.x, (*CUDA_CC).Brightness[i]);
            (*CUDA_CC2).Brightness[i] = (*CUDA_CC).Brightness[i];
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Begin(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_result* CUDA_FR,
    __global int* CUDA_End,
    int CUDA_n_iter_min,
    int CUDA_n_iter_max,
    double CUDA_iter_diff_max,
    double CUDA_Alamda_start)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

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
    //barrier(CLK_GLOBAL_MEM_FENCE); // TEST
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Start(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
    //__global int* CUDA_End)
{
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
    mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Matrix(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int lpoints)
{
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

    mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, lpoints, num);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Curve1(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    __local int num;  // __shared__
    __local double tmave[BLOCK_DIM];

    if (threadIdx.x == 0)
    {
        num = 0;
    }

    mrqcof_curve1(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, tmave, inrel, lpoints, num);

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("[Mrqcof1Curve1] [%d][%3d] alpha[56]: %10.7f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).alpha[56]);

    //if (blockIdx.x == 0)
    //	printf("dytemp[8636]: %10.7f\n", dytemp[8636]);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Curve1Last(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints)
{
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

    mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, res, inrel, lpoints);
    //if (threadIdx.x == 0)
    //{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++) {
    //		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    //}
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Curve2(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    //if (blockIdx.x == 0)
    //printf("[%3d] isInvalid: %3d, isNiter: %3d, isAlamda: %3d\n", threadIdx.x, (*CUDA_LCC).isInvalid, (*CUDA_LCC).isNiter, (*CUDA_LCC).isAlamda);

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    mrqcof_curve2(CUDA_LCC, CUDA_CC, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("[Mrqcof1Curve2] [%d][%3d] alpha[56]: %10.7f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).alpha[56]);

    //if (threadIdx.x == 0)
    //{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++) {
    //	printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    //}
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
{
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


    (*CUDA_LCC).Ochisq = mrqcof_end(CUDA_LCC, CUDA_CC, (*CUDA_LCC).alpha);


    ////if (threadIdx.x == 0)
    ////{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++) {
    //	printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    ////}
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqmin1End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
{
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


    mrqmin_1_end(CUDA_LCC, CUDA_CC);

    //if (blockIdx.x == 0) {
    //	printf("[%3d] sh_icol[%3d]: %3d\n", threadIdx.x, threadIdx.x, sh_icol[threadIdx.x]);
    //}
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Start(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
{
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
    mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("alpha[56]: %10.7f\n", (*CUDA_LCC).alpha[56]);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Matrix(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int lpoints)
{
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
    mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, lpoints, num);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Curve1(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    __local int num;  // __shared__
    __local double tmave[BLOCK_DIM];

    if (threadIdx.x == 0)
    {
        num = 0;
    }

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof2Curve1\n");

    //mrqcof_curve1(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
    mrqcof_curve1(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, tmave, inrel, lpoints, num);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Curve2(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    //__global double* CUDA_Dytemp,
    const int inrel,
    const int lpoints)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof2Curve2\n");

    mrqcof_curve2(CUDA_LCC, CUDA_CC, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Curve1Last(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints)
{
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
    mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, res, inrel, lpoints);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    (*CUDA_LCC).Chisq = mrqcof_end(CUDA_LCC, CUDA_CC, (*CUDA_LCC).covar);

    //if (blockIdx.x == 0)
    //	printf("[%3d] Chisq: %10.7f\n", threadIdx.x, (*CUDA_LCC).Chisq);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqmin2End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
{
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
    mrqmin_2_end(CUDA_LCC, CUDA_CC);

    (*CUDA_LCC).Niter++;

    //if (blockIdx.x == 0)
    //	printf("[%3d] Niter: %d\n", threadIdx.x, (*CUDA_LCC).Niter);
    //printf("|");
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
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
        if ((*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
        {
            if (threadIdx.x == 0)
            {
                (*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
            }

            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

            int brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
            if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
            int brtmpl = threadIdx.x * brtmph;
            brtmph = brtmpl + brtmph;
            if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
            brtmpl++;

            curv(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, brtmpl, brtmph);

            if (threadIdx.x == 0)
            {
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

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST

        if (threadIdx.x == 0)
        {
            //if (blockIdx.x == 0)
            //	printf("ndata - 3: %3d\n", (*CUDA_CC).ndata - 3);

            (*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / ((*CUDA_CC).ndata - 3));

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

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
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
    const double period = 2 * PI / (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3];

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
        (*CUDA_LFR).dark_best = dark / totarea * 100;
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
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
