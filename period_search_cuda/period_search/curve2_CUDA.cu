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

   One warp per block. The two branches below reproduce the original
   MrqcofCurve2's absolute (ia[1]!=0) and relative (ia[1]==0) address
   arithmetic element for element; within a tile only the summation order over
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
  double* __restrict__ dws = shw->dws;

  const int ma = CUDA_ma;
  const int mfit1 = CUDA_mfit1;
  const int lastone = CUDA_lastone, lastma = CUDA_lastma;
  double* __restrict__ dytemp = (*CUDA_LCC).dytemp;
  double* __restrict__ ytemp = (*CUDA_LCC).ytemp;

  const int lnp1base = (*CUDA_LCC).np1;
  const int lnp2base = (*CUDA_LCC).np2;
  const double ave = (*CUDA_LCC).ave;
  double ltrial_chisq = (*CUDA_LCC).trial_chisq;

  const int j1 = 1 + tid;   /* parameter rows owned by this lane */
  const int j2 = 33 + tid;

#pragma unroll 1
  for (int jp0 = 1; jp0 <= lpoints; jp0 += CURVE2_K)
    {
      int P = lpoints - jp0 + 1;
      if (P > CURVE2_K) P = CURVE2_K;

      /* ---- stage the tile (lanes = parameters, coalesced reads) and the
	 per-point scalars, ascending jp to keep the chisq order ----
	 Every lane computes every point's scalars anyway (lane 0 alone stores
	 dws), so each lane also stores its own columns of the weighted tile
	 W[p][l] = T[p][l] * s2w[p]: formed once per tile (as the OpenCL wpL)
	 instead of once per triangle entry - same product, same rounding. */
#pragma unroll 1
      for (int p = 0; p < CURVE2_K; p++)
	{
	  double r1 = 0.0, r2 = 0.0, s2wv = 0.0, dyv = 0.0;
	  if (p < P)
	    {
	      const int jp = jp0 + p;
	      const int lnp2 = lnp2base + jp;
	      double const* __restrict__ row = dytemp + (size_t)(jp - 1) * DYT_STRIDE;
	      double ymod;
	      if (inrel)
		{
		  /* renormalization for relative lightcurves, folded in;
		     same arithmetic as the old in-place pass */
		  double yytmp = ytemp[jp];
		  double coef = CUDA_sig[lnp1base + jp] * lpoints / ave;
		  double coef1 = yytmp / ave;
		  if (j1 >= 2 && j1 <= ma)
		    r1 = coef * (row[j1] - coef1 * (*CUDA_LCC).dave[j1]);
		  if (j2 <= ma)
		    r2 = coef * (row[j2] - coef1 * (*CUDA_LCC).dave[j2]);
		  /* j1 == 1: the size-scale derivative is explicitly zero */
		  ymod = coef * yytmp;
		}
	      else
		{
		  if (j1 <= ma) r1 = row[j1];
		  if (j2 <= ma) r2 = row[j2];
		  ymod = ytemp[jp];
		}
	      double sig2i = 1 / (CUDA_sig[lnp2] * CUDA_sig[lnp2]);
	      double wght = CUDA_Weight[lnp2];
	      dyv = CUDA_brightness[lnp2] - ymod;
	      s2wv = sig2i * wght;
	      ltrial_chisq = ltrial_chisq + dyv * dyv * s2wv;
	    }
	  T[p][j1] = r1;
	  W[p][j1] = r1 * s2wv;
	  /* column 64 is past the row (ma <= DYT_STRIDE-1) and never read */
	  if (j2 < DYT_STRIDE)
	    {
	      T[p][j2] = r2;
	      W[p][j2] = r2 * s2wv;
	    }
	  if (tid == 0)
	    dws[p] = dyv * s2wv;
	}
      __syncwarp();

      /* ---- rank-K triangular update, both original variants ----
	 Main triangle, both index variants at once: row L = l - o, column
	 M = m - o, 1 <= M <= L <= n, with o = 1 for relative curves (ia[1]==0:
	 the frozen first parameter is skipped and everything shifts by one) and
	 o = 0 otherwise; alpha[L*mfit1 + M] and beta[L] are exactly the entries
	 the old row loops updated.
	 Rows are processed in pairs (L, n+1-L): a pair has n+1 <= DYT_STRIDE
	 entries, so every lane has one or two of them, where the old one-row-at-
	 a-time loop left lanes m > l idle on the short rows. Each entry
	 computes acc = sum_p W[p][L+o] * T[p][M+o] in the same order, and the
	 beta rows (done by lane 0 alone before) are spread over the lanes -
	 bit-identical, with CURVE2_K fewer FP64 multiplies per entry now that
	 the weighted row comes precomputed from W. */
      {
	const int o = CUDA_ia[1] ? 0 : 1;
	const int n = lastone - o;

#pragma unroll 1
	for (int Lb = tid + 1; Lb <= n; Lb += 32)
	  {
	    double b = 0.0;
#pragma unroll
	    for (int p = 0; p < CURVE2_K; p++)
	      b += dws[p] * T[p][Lb + o];
	    beta[Lb] = beta[Lb] + b;
	  }

#pragma unroll 1
	for (int L1 = 1, L2 = n; L1 <= L2; L1++, L2--)
	  {
	    const int cnt = (L1 == L2) ? L1 : L1 + L2;
#pragma unroll 1
	    for (int q = tid; q < cnt; q += 32)
	      {
		int Lr, M;
		if (q < L1) { Lr = L1; M = q + 1; }
		else        { Lr = L2; M = q - L1 + 1; }

		double acc = 0.0;
#pragma unroll
		for (int p = 0; p < CURVE2_K; p++)
		  acc += W[p][Lr + o] * T[p][M + o];
		double* __restrict__ alphrow = alpha + Lr * mfit1;
		alphrow[M] = alphrow[M] + acc;
	      }
	  }
      }

      if (CUDA_ia[1]) /* absolute: gated tail rows unchanged */
	{
	  /* gated tail rows (lastone < l <= lastma), j counts gated rows */
	  int j = lastone;
#pragma unroll 1
	  for (int l = lastone + 1; l <= lastma; l++)
	    {
	      if (!CUDA_ia[l]) continue;
	      j++;
	      double* __restrict__ alphrow = alpha + j * mfit1;
#pragma unroll 1
	      for (int m = 1 + tid; m <= lastone; m += 32)
		{
		  double acc = 0.0;
#pragma unroll
		  for (int p = 0; p < CURVE2_K; p++)
		    acc += W[p][l] * T[p][m];
		  alphrow[m] = alphrow[m] + acc;
		}
	      if (tid == 0)
		{
		  int k = lastone;
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
      else /* relative (ia[1]==0): gated tail rows unchanged */
	{
	  int j = lastone - 1;
#pragma unroll 1
	  for (int l = lastone + 1; l <= lastma; l++)
	    {
	      if (!CUDA_ia[l]) continue;
	      j++;
	      double* __restrict__ alphrow = alpha + j * mfit1;
#pragma unroll 1
	      for (int m = 2 + tid; m <= lastone; m += 32)
		{
		  double acc = 0.0;
#pragma unroll
		  for (int p = 0; p < CURVE2_K; p++)
		    acc += W[p][l] * T[p][m];
		  alphrow[m - 1] = alphrow[m - 1] + acc;
		}
	      if (tid == 0)
		{
		  int k = lastone - 1;
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
      __syncwarp();
    } /* jp0 */

  if (tid == 0)
    {
      (*CUDA_LCC).np1 = lnp1base + lpoints;
      (*CUDA_LCC).np2 = lnp2base + lpoints;
      (*CUDA_LCC).trial_chisq = ltrial_chisq;
    }
  __syncwarp();
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
