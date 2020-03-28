/* N.B. The foll. L-M routines are modified versions of Press et al.
   converted from Mikko's fortran code

   8.11.2006
*/

#include <cuda.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <device_launch_parameters.h>
#include <stdio.h>

__device__ int mrqmin_1_end(freq_context* CUDA_LCC, const int ma, const int mfit, const int mfit1, const int block)
{
	int j;
	//precalc thread boundaries
	int tmph, tmpl;
	tmph = ma / block;
	if (ma % block) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > ma) tmph = ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = mfit / block;
	if (mfit % block) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > mfit) brtmph = mfit;
	brtmpl++;

	if ((*CUDA_LCC).isAlamda)
	{
		for (j = tmpl; j <= tmph; j++)
		{
			(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
		}

		__syncthreads();
	}

	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * mfit1 + 1;
		for (int k = 1; k <= CUDA_mfit; k++, ixx++)
		{
			(*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];
		}

		(*CUDA_LCC).covar[j * mfit1 + j] = (*CUDA_LCC).alpha[j * mfit1 + j] * (1 + (*CUDA_LCC).Alamda);
		(*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];
	}
	__syncthreads();

	int err_code = gauss_errc(CUDA_LCC, ma);
	if(err_code)
	{
		return;
	}

	//err_code = gauss_errc(CUDA_LCC, CUDA_mfit, (*CUDA_LCC).da);

	//     __syncthreads(); inside gauss

	if (threadIdx.x == 0)
	{

		//		if (err_code != 0) return(err_code); bacha na sync threads

		j = 0;
		for (int l = 1; l <= ma; l++)
			if (CUDA_ia[l])
			{
				j++;
				(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
			}
	}
	__syncthreads();

	return(err_code);
}

__device__ void mrqmin_2_end(freq_context* CUDA_LCC, int ia[], int ma)
{
	int j, k, l;

	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		(*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / CUDA_Alamda_incr;
		for (j = 1; j <= CUDA_mfit; j++)
		{
			for (k = 1; k <= CUDA_mfit; k++)
				(*CUDA_LCC).alpha[j * CUDA_mfit1 + k] = (*CUDA_LCC).covar[j * CUDA_mfit1 + k];
			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		}
		for (l = 1; l <= ma; l++)
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
	}
	else
	{
		(*CUDA_LCC).Alamda = CUDA_Alamda_incr * (*CUDA_LCC).Alamda;
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}

	return;
}

