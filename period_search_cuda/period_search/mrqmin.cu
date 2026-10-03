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

	}
	/* no barrier needed: the solver does not touch atry, and its first
	   (unconditional) barrier orders this copy before thread 0's atry
	   update below */

	/* the damped matrix is staged straight from alpha into shared memory by
	   the solver; covar is not touched (it is rezeroed by mrqcof_start before
	   mrqcof2 accumulates into it) */
	int err_code = gauss_errc_shared(CUDA_LCC, ma);
	if(err_code)
	{
		return err_code;
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

	return err_code;
}

__device__ void mrqmin_2_end(freq_context* CUDA_LCC, int ia[], int ma)
{
	const int tid = threadIdx.x;

	/* The threads of the block split the copies; the scalar updates are
	   done by thread 0 only. Its Chisq = Ochisq in the else branch cannot
	   send a late reader down the other branch (Chisq < Ochisq stays false). */
	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		if (tid == 0)
			(*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / CUDA_Alamda_incr;

		for (int e = tid; e < CUDA_mfit * CUDA_mfit; e += CUDA_BLOCK_DIM)
		{
			const int j = e / CUDA_mfit + 1;
			const int k = e - (j - 1) * CUDA_mfit + 1;
			(*CUDA_LCC).alpha[j * CUDA_mfit1 + k] = (*CUDA_LCC).covar[j * CUDA_mfit1 + k];
		}
		for (int j = tid + 1; j <= CUDA_mfit; j += CUDA_BLOCK_DIM)
			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		for (int l = tid + 1; l <= ma; l += CUDA_BLOCK_DIM)
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
	}
	else if (tid == 0)
	{
		(*CUDA_LCC).Alamda = CUDA_Alamda_incr * (*CUDA_LCC).Alamda;
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}

	return;
}