/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <device_launch_parameters.h>


/* comment the following line if no YORP */
/*#define YORP*/

__device__ void mrqcof_start(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[])
{
   int j,k;
//
    int brtmph,brtmpl;
	brtmph=CUDA_Numfac/CUDA_BLOCK_DIM;
	if(CUDA_Numfac%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>CUDA_Numfac) brtmph=CUDA_Numfac;
	brtmpl++;

   /* N.B. curv and blmatrix called outside bright
      because output same for all points */
   curv(CUDA_LCC,a,brtmpl,brtmph);

   if (threadIdx.x==0)
   {
//   #ifdef YORP
//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
  // #else
      blmatrix(CUDA_LCC,a[CUDA_ma-4-CUDA_Nphpar],a[CUDA_ma-3-CUDA_Nphpar]);
//   #endif
	   (*CUDA_LCC).trial_chisq = 0;
	   (*CUDA_LCC).np = 0;
	   (*CUDA_LCC).np1 = 0;
	   (*CUDA_LCC).np2 = 0;
	   (*CUDA_LCC).ave = 0;
   }

    brtmph=CUDA_mfit/CUDA_BLOCK_DIM;
	if(CUDA_mfit%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>CUDA_mfit) brtmph=CUDA_mfit;
	brtmpl++;

   for(j = brtmpl; j <= brtmph; j++)
   {
      for (k = 1; k <= j; k++)
         alpha[j*(CUDA_mfit1)+k]=0;
      beta[j]=0;
   }
}

__device__ double mrqcof_end(freq_context *CUDA_LCC,double *alpha)
{
   /* mirror the lower triangle into the upper one: alpha[k][j] = alpha[j][k]
      for k < j. The threads of the block split the (j, k) pairs; reads
      (row > col) and writes (row < col) never overlap, so no ordering is
      needed and every entry gets exactly the value it got before. */
   const int total = CUDA_mfit * (CUDA_mfit - 1) / 2;

   /* pair e -> row j = 2.., column k = 1..j-1 (row-major lower triangle) */
   int j = 2, k = threadIdx.x + 1;
   while (k > j - 1) { k -= j - 1; j++; }
   for (int e = threadIdx.x; e < total; e += CUDA_BLOCK_DIM)
   {
      alpha[k*(CUDA_mfit1)+j] = alpha[j*(CUDA_mfit1)+k];

      k += CUDA_BLOCK_DIM;
      while (k > j - 1) { k -= j - 1; j++; }
   }

   return (*CUDA_LCC).trial_chisq;
}

__device__ void mrqcof_matrix(freq_context *CUDA_LCC, double a[], int Lpoints)
{
   /* geometry is computed inside bright_curve1() since the 2026 rewrite */
}

__device__ void mrqcof_curve1(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
   /* geometry, brightness, derivatives, and the dave/ave sums are all
	  produced by bright_curve1() (see bright.cu), one thread per point.
	  alpha/beta are untouched here - they are accumulated in MrqcofCurve2. */
   bright_curve1(CUDA_LCC, a, Inrel, Lpoints);
}

__device__ void mrqcof_curve1_last(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
	/* the last "lightcurve" is the convexity regularization: brightness and
	   derivatives depend only on Area and Dsph (all rotation/phase columns
	   are zero, as in the old conv()). One warp per block; the Dg fold
	   applies here too: Dg[i][l]*Darea[i]*Nor = Dsph[i][l]*(Area[i]*Nor). */
	const int tid = threadIdx.x;
	brightshare* __restrict__ shw = &mrq_share_block()->b;
	double* __restrict__ ww = shw->wcA;

	const int ma = CUDA_ma, nco = CUDA_Ncoef, nf = CUDA_Numfac;
	double* __restrict__ dytemp = (*CUDA_LCC).dytemp;
	double* __restrict__ ytemp = (*CUDA_LCC).ytemp;
	double const* __restrict__ areap = &CUDA_Area[blockIdx.x * CUDA_Numfac1];
	int lnp = (*CUDA_LCC).np;
	double lave = (Inrel == 1) ? 0 : (*CUDA_LCC).ave;

	const int c1 = 1 + tid, c2 = 33 + tid;
	double dave1 = 0, dave2 = 0;

#pragma unroll 1
	for (int jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;
		double ym = 0, a1 = 0, a2 = 0;
#pragma unroll 1
		for (int f0 = 1; f0 <= nf; f0 += 32)
		{
			const int i = f0 + tid;
			double w = 0.0;
			if (i <= nf)
			{
				w = areap[i] * CUDA_Nor[i][jp - 1];
				ym += w;
			}
			ww[tid] = w;
			__syncwarp();
			int kend = nf - f0 + 1;
			if (kend > 32) kend = 32;
#pragma unroll 4
			for (int k = 0; k < kend; k++)
			{
				double w2 = ww[k];
				double const* __restrict__ row = CUDA_Dsph[f0 + k];
				a1 += w2 * row[c1];
				a2 += w2 * row[c2];
			}
			__syncwarp();
		}
#pragma unroll
		for (int off = 16; off > 0; off >>= 1)
			ym += __shfl_xor_sync(0xffffffff, ym, off);

		double v1 = (c1 <= nco) ? a1 : 0.0;
		double v2 = (c2 <= nco) ? a2 : 0.0;
		double* __restrict__ row = dytemp + (size_t)(jp - 1) * DYT_STRIDE;
		if (c1 <= ma) { row[c1] = v1; dave1 += v1; }
		if (c2 <= ma) { row[c2] = v2; dave2 += v2; }
		if (tid == 0) ytemp[jp] = ym;
		if (Inrel == 1) lave += ym;
		__syncwarp();
	}

	if (Inrel == 1)
	{
		/* the old code reset dave[] and accumulated the 3 points into it;
		   the per-lane column sums are exactly that */
		if (c1 <= ma) (*CUDA_LCC).dave[c1] = dave1;
		if (c2 <= ma) (*CUDA_LCC).dave[c2] = dave2;
	}
	if (tid == 0)
	{
		(*CUDA_LCC).np = lnp;
		(*CUDA_LCC).ave = lave;
	}
}
