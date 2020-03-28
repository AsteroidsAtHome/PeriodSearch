#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//__device__ int gauss_errc(freq_context *CUDA_LCC,int n, double b[])
__device__ int gauss_errc(freq_context* CUDA_LCC, const int ma)
{
	__shared__ int icol;
	__shared__ double pivinv;
	__shared__ int sh_icol[CUDA_BLOCK_DIM];
	__shared__ int sh_irow[CUDA_BLOCK_DIM];
	__shared__ double sh_big[CUDA_BLOCK_DIM];

	//	__shared__ int indxc[MAX_N_PAR+1],indxr[MAX_N_PAR+1],ipiv[MAX_N_PAR+1];
	int i, licol = 0, irow = 0, j, k, l, ll;
	double big, dum, temp;
	int n = CUDA_mfit;

	int brtmph, brtmpl;
	brtmph = n / CUDA_BLOCK_DIM;
	if (n % CUDA_BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > n) brtmph = n;
	brtmpl++;

	/*        indxc=vector_int(n+1);
		indxr=vector_int(n+1);
		ipiv=vector_int(n+1);*/

	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++) (*CUDA_LCC).ipiv[j] = 0;
	}
	__syncthreads();

	for (i = 1; i <= n; i++)
	{
		big = 0.0;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)
			if ((*CUDA_LCC).ipiv[j] != 1)
			{
				int ixx = j * (CUDA_mfit1)+1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if ((*CUDA_LCC).ipiv[k] == 0)
					{
						double tmpcov = fabs((*CUDA_LCC).covar[ixx]);
						if (tmpcov >= big)
						{
							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if ((*CUDA_LCC).ipiv[k] > 1)
					{
						//printf("-");
						__syncthreads();
						/*					        deallocate_vector((void *) ipiv);
												deallocate_vector((void *) indxc);
												deallocate_vector((void *) indxr);*/
						return(1);
					}
				}
			}
		sh_big[threadIdx.x] = big;
		sh_irow[threadIdx.x] = irow;
		sh_icol[threadIdx.x] = licol;
		__syncthreads();
		if (threadIdx.x == 0)
		{
			big = sh_big[0];
			icol = sh_icol[0];
			irow = sh_irow[0];
			for (j = 1; j < CUDA_BLOCK_DIM; j++)
			{
				if (sh_big[j] >= big)
				{
					big = sh_big[j];
					irow = sh_irow[j];
					icol = sh_icol[j];
				}
			}
			++((*CUDA_LCC).ipiv[icol]);
			if (irow != icol)
			{
				for (l = 1; l <= n; l++)
				{
					SWAP((*CUDA_LCC).covar[irow * (CUDA_mfit1)+l], (*CUDA_LCC).covar[icol * (CUDA_mfit1)+l])
				}

				SWAP((*CUDA_LCC).da[irow], (*CUDA_LCC).da[icol])
					//SWAP(b[irow],b[icol])
			}
			(*CUDA_LCC).indxr[i] = irow;
			(*CUDA_LCC).indxc[i] = icol;
			if ((*CUDA_LCC).covar[icol * (CUDA_mfit1)+icol] == 0.0)
			{
				j = 0;
				for (int l = 1; l <= ma; l++)
				{
					if (CUDA_ia[l])
					{
						j++;
						(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
					}
				}
				//printf("+");
				/*					    deallocate_vector((void *) ipiv);
												deallocate_vector((void *) indxc);
												deallocate_vector((void *) indxr);*/
				return(2);
			}
			pivinv = 1.0 / (*CUDA_LCC).covar[icol * (CUDA_mfit1)+icol];
			(*CUDA_LCC).covar[icol * (CUDA_mfit1)+icol] = 1.0;
			(*CUDA_LCC).da[icol] *= pivinv;
			//b[icol] *= pivinv;
		}
		__syncthreads();

		for (l = brtmpl; l <= brtmph; l++)
		{
			(*CUDA_LCC).covar[icol * (CUDA_mfit1)+l] *= pivinv;
		}
		__syncthreads();

		for (ll = brtmpl; ll <= brtmph; ll++)
			if (ll != icol)
			{
				int ixx = ll * (CUDA_mfit1), jxx = icol * (CUDA_mfit1);
				dum = (*CUDA_LCC).covar[ixx + icol];
				(*CUDA_LCC).covar[ixx + icol] = 0.0;
				ixx++;
				jxx++;
				for (l = 1; l <= n; l++, ixx++, jxx++) (*CUDA_LCC).covar[ixx] -= (*CUDA_LCC).covar[jxx] * dum;
				(*CUDA_LCC).da[ll] -= (*CUDA_LCC).da[icol] * dum;
				//b[ll] -= b[icol]*dum;
			}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		for (l = n; l >= 1; l--)
		{
			if ((*CUDA_LCC).indxr[l] != (*CUDA_LCC).indxc[l])
				for (k = 1; k <= n; k++)
					SWAP((*CUDA_LCC).covar[k * (CUDA_mfit1)+(*CUDA_LCC).indxr[l]], (*CUDA_LCC).covar[k * (CUDA_mfit1)+(*CUDA_LCC).indxc[l]]);
		}
	}
	__syncthreads();
	/*        deallocate_vector((void *) ipiv);
		deallocate_vector((void *) indxc);
		deallocate_vector((void *) indxr);*/

	return(0);
}
#undef SWAP
/* from Numerical Recipes */
