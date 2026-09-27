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
   int j,k;

   /* mirror the lower triangle; each row is split over the block's threads
      (source and destination never overlap) */
   for (j = 2; j <= CUDA_mfit; j++)
      for (k = 1 + threadIdx.x; k <= j-1; k += blockDim.x)
         alpha[k*(CUDA_mfit1)+j] = alpha[j*(CUDA_mfit1)+k];

   return (*CUDA_LCC).trial_chisq;
}

__device__ void mrqcof_matrix(freq_context *CUDA_LCC, double a[], int Lpoints)
{
   /* geometry is computed inside bright_curve1_warp() since the 2026 rewrite */
}

__device__ void mrqcof_curve1(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
   /* warp-cooperative rewrite: geometry, brightness, derivatives, and the
	  dave/ave sums are all produced by one warp in bright_curve1_warp()
	  (see bright.cu). alpha/beta are untouched here - they are accumulated
	  in MrqcofCurve2. */
   bright_curve1_warp(CUDA_LCC, a, Inrel, Lpoints);
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

	const int ma = CUDA_ma, nco = CUDA_Ncoef, nf = CUDA_Numfac;
	double* __restrict__ dytemp = (*CUDA_LCC).dytemp;
	double* __restrict__ ytemp = (*CUDA_LCC).ytemp;
	double const* __restrict__ areap = &CUDA_Area[blockIdx.x * CUDA_Numfac1];
	int lnp = (*CUDA_LCC).np;
	double lave = (Inrel == 1) ? 0 : (*CUDA_LCC).ave;

	const int c1 = 1 + tid, c2 = 33 + tid;
	double dave1 = 0, dave2 = 0;

	/* the points (the 3 convexity pseudo-points) are processed together, up to
	   three per pass over the facets - it used to be one pass per point. Each
	   CUDA_Dsph row element is loaded once for all of them and the per-point
	   sums are independent chains; every sum keeps its facet order, and
	   dave/lave still accumulate the points in ascending order. */
	double* __restrict__ ww3 = &shw->geo[0][0];   /* 3 x 32 staged weights */
#pragma unroll 1
	for (int jb = 1; jb <= Lpoints; jb += 3)
	{
		const int npts = (Lpoints - jb + 1 < 3) ? (Lpoints - jb + 1) : 3;
		double ym[3] = { 0, 0, 0 }, a1[3] = { 0, 0, 0 }, a2[3] = { 0, 0, 0 };
#pragma unroll 1
		for (int f0 = 1; f0 <= nf; f0 += 32)
		{
			const int i = f0 + tid;
#pragma unroll
			for (int q = 0; q < 3; q++)
			{
				double w = 0.0;
				if (i <= nf && q < npts)
				{
					w = areap[i] * CUDA_Nor[i][jb + q - 1];
					ym[q] += w;
				}
				ww3[q * 32 + tid] = w;
			}
			__syncwarp();
			int kend = nf - f0 + 1;
			if (kend > 32) kend = 32;
#pragma unroll 4
			for (int k = 0; k < kend; k++)
			{
				double const* __restrict__ row = CUDA_Dsph[f0 + k];
				double r1 = row[c1], r2 = row[c2];
#pragma unroll
				for (int q = 0; q < 3; q++)
				{
					double w2 = ww3[q * 32 + k];
					a1[q] += w2 * r1;
					a2[q] += w2 * r2;
				}
			}
			__syncwarp();
		}

		for (int q = 0; q < npts; q++)
		{
			const int jp = jb + q;
			lnp++;
#pragma unroll
			for (int off = 16; off > 0; off >>= 1)
				ym[q] += __shfl_xor_sync(0xffffffff, ym[q], off);

			double v1 = (c1 <= nco) ? a1[q] : 0.0;
			double v2 = (c2 <= nco) ? a2[q] : 0.0;
			double* __restrict__ row = dytemp + (size_t)(jp - 1) * DYT_STRIDE;
			if (c1 <= ma) { row[c1] = v1; dave1 += v1; }
			if (c2 <= ma) { row[c2] = v2; dave2 += v2; }
			if (tid == 0) ytemp[jp] = ym[q];
			if (Inrel == 1) lave += ym[q];
		}
		/* ww3 is re-written by the next pass */
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
	__syncwarp();
}
