//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 2026 rewrite: the normal equations are accumulated once per CURVE2_K-point
   tile (a rank-K update computed from shared memory) instead of once per data
   point. The old code did, for every point, a read-modify-write sweep of the
   whole triangular alpha matrix in global memory - by far the largest memory
   stream of the application after Dg. Staging reads dytemp coalesced (it is
   stored transposed, see bright.cu) and the relative-lightcurve
   renormalization is folded into the staging, which removes one more full
   read+write pass over dytemp.

   One block of CUDA_BLOCK_DIM threads per context, laid out as in the
   OpenCL build (flattened triangle, W = T * s2w staged once per tile, beta
   rows spread over the threads). The original MrqcofCurve2's absolute
   (ia[1]!=0) and relative (ia[1]==0) address arithmetic is reproduced
   element for element; within a tile only the summation order over
   the K points changes (a+b+c+d... becomes one fused sum), which is the usual
   reordering freedom.

   T[p][l] holds the staged dyda of point p, 1-based parameter row l. Tiles
   past the end of the lightcurve are zero-filled so they add exact zeros. */

__device__ void MrqcofCurve2(freq_context* CUDA_LCC, double* alpha, double beta[], int inrel, int lpoints)
{
  const int tid = threadIdx.x;
  curve2share* __restrict__ shw = curve2_share_block();
  double (* __restrict__ T)[DYT_STRIDE] = shw->T;
  double (* __restrict__ W)[DYT_STRIDE] = shw->W;
  double* __restrict__ s2w = shw->s2w;
  double* __restrict__ dws = shw->dws;
  double* __restrict__ dys = shw->dy;

  const int ma = CUDA_ma;
  const int mfit1 = CUDA_mfit1;
  const int lastone = CUDA_lastone, lastma = CUDA_lastma;
  double* __restrict__ dytemp = (*CUDA_LCC).dytemp;
  double* __restrict__ ytemp = (*CUDA_LCC).ytemp;

  const int lnp1base = (*CUDA_LCC).np1;
  const int lnp2base = (*CUDA_LCC).np2;
  const double ave = (*CUDA_LCC).ave;
  double ltrial_chisq = (*CUDA_LCC).trial_chisq;

  /* both original index variants at once: row L = l - o, column M = m - o,
     1 <= M <= L <= n, with o = 1 for relative curves (ia[1] == 0: the frozen
     first parameter is skipped and everything shifts by one) and o = 0
     otherwise; alpha[L][M] and beta[L] are exactly the entries the old
     row-by-row loops updated */
  const int o = CUDA_ia[1] ? 0 : 1;
  const int n = lastone - o;
  const int E = n * (n + 1) / 2;

  /* this thread's first (L, M) pair of the flattened triangle */
  int L0 = 1, M0 = tid + 1;
  while (M0 > L0) { M0 -= L0; L0++; }

#pragma unroll 1
  for (int jp0 = 1; jp0 <= lpoints; jp0 += CURVE2_K)
    {
      int P = lpoints - jp0 + 1;
      if (P > CURVE2_K) P = CURVE2_K;

      /* ---- stage the tile (consecutive threads = consecutive parameters,
	 coalesced reads); tiles past the end of the lightcurve and the
	 unused column 0 are zero-filled so they add exact zeros ---- */
#pragma unroll 1
      for (int e = tid; e < CURVE2_K * DYT_STRIDE; e += CUDA_BLOCK_DIM)
	{
	  const int p = e / DYT_STRIDE;
	  const int c = e - p * DYT_STRIDE;
	  double r = 0.0;
	  if (p < P)
	    {
	      const int jp = jp0 + p;
	      double const* __restrict__ row = dytemp + (size_t)(jp - 1) * DYT_STRIDE;
	      if (inrel)
		{
		  /* renormalization for relative lightcurves, folded in;
		     same arithmetic as the old in-place pass */
		  if (c >= 2 && c <= ma)
		    {
		      double yytmp = ytemp[jp];
		      double coef = CUDA_sig[lnp1base + jp] * lpoints / ave;
		      double coef1 = yytmp / ave;
		      r = coef * (row[c] - coef1 * (*CUDA_LCC).dave[c]);
		    }
		  /* c == 1: the size-scale derivative is explicitly zero */
		}
	      else if (c >= 1 && c <= ma)
		r = row[c];
	    }
	  T[p][c] = r;
	}

      /* ---- per-point scalars ---- */
      if (tid < CURVE2_K)
	{
	  const int p = tid;
	  double s2wv = 0.0, dyv = 0.0;
	  if (p < P)
	    {
	      const int jp = jp0 + p;
	      const int lnp2 = lnp2base + jp;
	      double ymod;
	      if (inrel)
		{
		  double coef = CUDA_sig[lnp1base + jp] * lpoints / ave;
		  ymod = coef * ytemp[jp];
		}
	      else
		ymod = ytemp[jp];
	      double sig2i = 1 / (CUDA_sig[lnp2] * CUDA_sig[lnp2]);
	      double wght = CUDA_Weight[lnp2];
	      dyv = CUDA_brightness[lnp2] - ymod;
	      s2wv = sig2i * wght;
	    }
	  s2w[p] = s2wv;
	  dws[p] = dyv * s2wv;
	  dys[p] = dyv;
	}
      __syncthreads();

      /* W[p][l] = T[p][l] * s2w[p], computed once per tile instead of once
	 per row by every thread (same product, same rounding) */
#pragma unroll 1
      for (int e = tid; e < CURVE2_K * DYT_STRIDE; e += CUDA_BLOCK_DIM)
	{
	  const int p = e / DYT_STRIDE;
	  const int c = e - p * DYT_STRIDE;
	  W[p][c] = T[p][c] * s2w[p];
	}
      __syncthreads();

      /* ---- rank-K update of the main triangle: flattened and dealt
	 round-robin to all threads (the old per-row split left most lanes
	 idle); every entry has exactly one writer and gets the same
	 alpha + (sum over the tile's points in ascending order) ---- */
      {
	int Lr = L0, Mr = M0;
#pragma unroll 1
	for (int e = tid; e < E; e += CUDA_BLOCK_DIM)
	  {
	    double acc = 0.0;
#pragma unroll
	    for (int p = 0; p < CURVE2_K; p++)
	      acc += W[p][Lr + o] * T[p][Mr + o];
	    alpha[Lr * mfit1 + Mr] = alpha[Lr * mfit1 + Mr] + acc;

	    Mr += CUDA_BLOCK_DIM;
	    while (Mr > Lr) { Mr -= Lr; Lr++; }
	  }
      }
      /* the beta rows, spread out instead of all running on thread 0 */
#pragma unroll 1
      for (int Lb = tid + 1; Lb <= n; Lb += CUDA_BLOCK_DIM)
	{
	  double b = 0.0;
#pragma unroll
	  for (int p = 0; p < CURVE2_K; p++)
	    b += dws[p] * T[p][Lb + o];
	  beta[Lb] = beta[Lb] + b;
	}

      /* ---- gated tail rows (lastone < l <= lastma), j counts gated rows
	 from n; columns 1..n as above, then the gated columns ---- */
      {
	int j = n;
#pragma unroll 1
	for (int l = lastone + 1; l <= lastma; l++)
	  {
	    if (!CUDA_ia[l]) continue;
	    j++;
	    double* __restrict__ alphrow = alpha + j * mfit1;
#pragma unroll 1
	    for (int Mt = 1 + tid; Mt <= n; Mt += CUDA_BLOCK_DIM)
	      {
		double acc = 0.0;
#pragma unroll
		for (int p = 0; p < CURVE2_K; p++)
		  acc += W[p][l] * T[p][Mt + o];
		alphrow[Mt] = alphrow[Mt] + acc;
	      }
	    if (tid == 0)
	      {
		int k = n;
		for (int m = lastone + 1; m <= l; m++)
		  {
		    if (CUDA_ia[m])
		      {
			k++;
			double acc = 0.0;
#pragma unroll
			for (int p = 0; p < CURVE2_K; p++)
			  acc += W[p][l] * T[p][m];
			alphrow[k] = alphrow[k] + acc;
		      }
		  }
		double b = 0.0;
#pragma unroll
		for (int p = 0; p < CURVE2_K; p++)
		  b += dws[p] * T[p][l];
		beta[j] = beta[j] + b;
	      }
	  }
      }

      /* chi-square: same per-point terms in the same ascending order */
      if (tid == 0)
	{
	  for (int p = 0; p < P; p++)
	    ltrial_chisq = ltrial_chisq + dys[p] * dys[p] * s2w[p];
	}

      /* everyone must finish reading the tile before the next one is staged
	 (np1/np2 below were read by every thread before the first barrier) */
      if (jp0 + CURVE2_K <= lpoints)
	__syncthreads();
    } /* jp0 */

  if (tid == 0)
    {
      (*CUDA_LCC).np1 = lnp1base + lpoints;
      (*CUDA_LCC).np2 = lnp2base + lpoints;
      (*CUDA_LCC).trial_chisq = ltrial_chisq;
    }
}


__global__ void CudaCalculateIter1Mrqcof1Curve2(const int inrel, const int lpoints)
{
  const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

  if ((*CUDA_LCC).isInvalid) return;

  if (!(*CUDA_LCC).isNiter) return;

  if (!(*CUDA_LCC).isAlamda) return;

  MrqcofCurve2(CUDA_LCC, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof2Curve2(const int inrel, const int lpoints)
{
  const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

  if ((*CUDA_LCC).isInvalid) return;

  if (!(*CUDA_LCC).isNiter) return;

  MrqcofCurve2(CUDA_LCC, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}
