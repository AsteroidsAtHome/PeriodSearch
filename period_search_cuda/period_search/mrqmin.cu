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
	/* no __syncthreads() here: the solver does not touch atry, and its own
	   syncs order this copy before thread 0 rewrites atry below */

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
	/* no trailing __syncthreads(): this is the last statement of
	   CudaCalculateIter1Mrqmin1End, the kernel boundary orders it */

	return err_code;
}

__device__ void mrqmin_2_end(freq_context* CUDA_LCC, int ia[], int ma)
{
	/* the threads of the block (any block size) split the copies; the scalar
	   updates are done by thread 0 only. Its Chisq = Ochisq in the else branch
	   cannot send a late reader down the other branch (Chisq < Ochisq stays
	   false), and every copied value is the same as in the serial loop. */
	const int tid = threadIdx.x;
	const int mfit = CUDA_mfit, mfit1 = CUDA_mfit1;

	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		if (tid == 0)
			(*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / CUDA_Alamda_incr;
		for (int e = tid; e < mfit * mfit; e += blockDim.x)
		{
			const int j = e / mfit + 1;
			const int k = e - (j - 1) * mfit + 1;
			(*CUDA_LCC).alpha[j * mfit1 + k] = (*CUDA_LCC).covar[j * mfit1 + k];
		}
		for (int j = tid + 1; j <= mfit; j += blockDim.x)
			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		for (int l = tid + 1; l <= ma; l += blockDim.x)
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
	}
	else if (tid == 0)
	{
		(*CUDA_LCC).Alamda = CUDA_Alamda_incr * (*CUDA_LCC).Alamda;
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}
}