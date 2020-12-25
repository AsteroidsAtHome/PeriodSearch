#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

#include "constants.h"
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
//#include "../../../../../../../Program Files (x86)/Windows Kits/10/Include/10.0.10240.0/ucrt/math.h"
#include <cstdio>

__global__ void CudaCalculatePrepare(int n_start, int n_max, double freq_start, double freq_step)
{
	const auto n = n_start + blockIdx.x;
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	//zero context
	//	CUDA_CC is zeroed itself as global memory but need to reset between freq TODO
	if (n > n_max)
	{
		(*CUDA_LCC).isInvalid = 1;
		return;
	}
	else
	{
		(*CUDA_LCC).isInvalid = 0;
	}

	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;

	/* initial poles */
	(*CUDA_LFR).per_best = 0;
	(*CUDA_LFR).dark_best = 0;
	(*CUDA_LFR).la_best = 0;
	(*CUDA_LFR).be_best = 0;
	(*CUDA_LFR).dev_best = 1e40;
}

__global__ void CudaCalculatePreparePole(int m)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		atomicAdd(&CUDA_End, 1);
		(*CUDA_LFR).isReported = 0; //signal not to read result

		return;
	}

	const auto period = 1 / (*CUDA_LCC).freq;

	/* starts from the initial ellipsoid */
	for (auto i = 1; i <= CUDA_Ncoef; i++)
	{
		(*CUDA_LCC).cg[i] = CUDA_cg_first[i];
	}

	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = CUDA_beta_pole[m];
	(*CUDA_LCC).cg[CUDA_Ncoef + 2] = CUDA_lambda_pole[m];

	/* The formulas use beta measured from the pole */
	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = 90 - (*CUDA_LCC).cg[CUDA_Ncoef + 1];

	/* conversion of lambda, beta to radians */
	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef + 1];
	(*CUDA_LCC).cg[CUDA_Ncoef + 2] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef + 2];

	/* Use omega instead of period */
	(*CUDA_LCC).cg[CUDA_Ncoef + 3] = 24 * 2 * PI / period;

	for (auto i = 1; i <= CUDA_Nphpar; i++)
	{
		(*CUDA_LCC).cg[CUDA_Ncoef + 3 + i] = CUDA_par[i];
		//              ia[Ncoef+3+i] = ia_par[i]; moved to global
	}

	/* Lommel-Seeliger part */
	(*CUDA_LCC).cg[CUDA_Ncoef + 3 + CUDA_Nphpar + 2] = 1;
	/* Use logarithmic formulation for Lambert to keep it positive */
	(*CUDA_LCC).cg[CUDA_Ncoef + 3 + CUDA_Nphpar + 1] = log(CUDA_cl);

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

__global__ void CudaCalculateIter1Begin(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	(*CUDA_LCC).isNiter = (((*CUDA_LCC).Niter < CUDA_n_iter_max) && ((*CUDA_LCC).iter_diff > CUDA_iter_diff_max)) || ((*CUDA_LCC).Niter < CUDA_n_iter_min);

	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Alamda < 0)
		{
			(*CUDA_LCC).isAlamda = 1;
			(*CUDA_LCC).Alamda = CUDA_Alamda_start; /* initial alambda */
		}
		else
			(*CUDA_LCC).isAlamda = 0;
	}
	else
	{
		if (!(*CUDA_LFR).isReported)
		{
			atomicAdd(&CUDA_End, 1);
#ifdef _DEBUG
			/*const int is_precalc = CUDA_Is_Precalc;
			if(is_precalc)
			{
				printf("%d ", CUDA_End);
			}*/
#endif
			(*CUDA_LFR).isReported = 1;
		}
	}

}

__global__ void CudaCalculateIter1Mrqmin1End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	int block = CUDA_BLOCK_DIM;
	/*gauss_err=*/mrqmin_1_end(CUDA_LCC, CUDA_ma, CUDA_mfit, CUDA_mfit1, block);
}

__global__ void CudaCalculateIter1Mrqmin2End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqmin_2_end(CUDA_LCC, CUDA_ia, CUDA_ma);
	(*CUDA_LCC).Niter++;
}

__global__ void CudaCalculateIter1Mrqcof1Start(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_start(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
}

__global__ void CudaCalculateIter1Mrqcof1Matrix(const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).cg, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof1Curve1(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof1Curve1Last(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1_last(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof1End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	(*CUDA_LCC).Ochisq = mrqcof_end(CUDA_LCC, (*CUDA_LCC).alpha);
}

__global__ void CudaCalculateIter1Mrqcof2Start(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_start(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
}

__global__ void CudaCalculateIter1Mrqcof2Matrix(const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).atry, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof2Curve1(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve1(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof2Curve1Last(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve1_last(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof2End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	(*CUDA_LCC).Chisq = mrqcof_end(CUDA_LCC, (*CUDA_LCC).covar);
}

__global__ void CudaCalculateIter2(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
		{
			if (threadIdx.x == 0)
			{
				(*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
			}
			__syncthreads();

			auto brtmph = CUDA_Numfac / CUDA_BLOCK_DIM;
			if (CUDA_Numfac % CUDA_BLOCK_DIM) brtmph++;
			int brtmpl = threadIdx.x * brtmph;
			brtmph = brtmpl + brtmph;
			if (brtmph > CUDA_Numfac) brtmph = CUDA_Numfac;
			brtmpl++;

			curv(CUDA_LCC, (*CUDA_LCC).cg, brtmpl, brtmph);

			if (threadIdx.x == 0)
			{
				for (auto i = 1; i <= 3; i++)
				{
					(*CUDA_LCC).chck[i] = 0;
					for (auto j = 1; j <= CUDA_Numfac; j++)
					{
						(*CUDA_LCC).chck[i] = (*CUDA_LCC).chck[i] + (*CUDA_LCC).Area[j] * CUDA_Nor[j][i - 1];
					}
				}

				//(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - (pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0)) * pow(CUDA_conw_r, 2.0);
				(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1] + (*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2]+ (*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]) * (CUDA_conw_r * CUDA_conw_r);
			}
		}
		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / (CUDA_ndata - 3));
			/* only if this step is better than the previous,
				1e-10 is for numeric errors */
			if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
			{
				(*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
				(*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
			}
			//		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
		}
	}
}

__global__ void CudaCalculateFinishPole(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	double totarea = 0;
	for (auto i = 1; i <= CUDA_Numfac; i++)
	{
		totarea = totarea + (*CUDA_LCC).Area[i];
	}

	//const auto sum = pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0);
	const auto sum = ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1]) + ((*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2]) + ((*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]);
	const auto dark = sqrt(sum);

	/* period solution */
	const auto period = 2 * PI / (*CUDA_LCC).cg[CUDA_Ncoef + 3];

	/* pole solution */
	const auto la_tmp = RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef + 2];
	const auto be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef + 1];

	if ((*CUDA_LCC).dev_new < (*CUDA_LFR).dev_best)
	{
		(*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
		(*CUDA_LFR).per_best = period;
		(*CUDA_LFR).dark_best = dark / totarea * 100;
		(*CUDA_LFR).la_best = la_tmp;
		(*CUDA_LFR).be_best = be_tmp;
	}
	//debug
	/*	(*CUDA_LFR).dark=dark;
	(*CUDA_LFR).totarea=totarea;
	(*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
	(*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
	(*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
}

__global__ void CudaCalculateFinish(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if ((*CUDA_LFR).la_best < 0)
		(*CUDA_LFR).la_best += 360;

	if (isnan((*CUDA_LFR).dark_best) == 1)
		(*CUDA_LFR).dark_best = 1.0;
}