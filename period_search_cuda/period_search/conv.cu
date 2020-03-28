/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <device_launch_parameters.h>

//#define __device

__device__ double conv(freq_context *CUDA_LCC,int nc,int tmpl,int tmph,int brtmpl,int brtmph)
{
   int i, j,k;
   __shared__ double res[CUDA_BLOCK_DIM];
   double tmp,dtmp;

   tmp=0;
	j=blockIdx.x*(CUDA_Numfac1)+brtmpl;
   for (i = brtmpl; i <= brtmph; i++,j++)
   {
	   int2 bfr;
		bfr=tex1Dfetch(texArea,j);
		tmp += __hiloint2double(bfr.y,bfr.x) * CUDA_Nor[i][nc];
   }
   res[threadIdx.x]=tmp;
   __syncthreads();
//parallel reduction
	k=CUDA_BLOCK_DIM>>1;
	while (k>1)
	{
		if (threadIdx.x<k) res[threadIdx.x]+=res[threadIdx.x+k];
		k=k>>1;
		__syncthreads();
	}
	if (threadIdx.x==0)
	{
		tmp = res[0]+res[1];
	}
//parallel reduction end
	__syncthreads();

	int m=blockIdx.x*CUDA_Dg_block+tmpl*(CUDA_Numfac1);
	for (j = tmpl; j <= tmph; j++,m+=(CUDA_Numfac1))
	{
			dtmp=0;
			if (j<=CUDA_Ncoef)
			{
				int mm=m+1;
				for (i = 1; i <= CUDA_Numfac; i++,mm++)
				{
					int2 xx;
					xx=tex1Dfetch(texDg,mm);
					dtmp += CUDA_Darea[i] * __hiloint2double(xx.y,xx.x) * CUDA_Nor[i][nc];
				}
			}
			(*CUDA_LCC).dyda[j]=dtmp;
	}
	__syncthreads();

   return (tmp);
}
