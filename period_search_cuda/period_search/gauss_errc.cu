/* from Numerical Recipes

   8.11.2006

   2026: the damped normal matrix is staged into dynamic shared memory and the
   Gauss-Jordan elimination runs entirely there, one block (CUDA_BLOCK_DIM
   threads) per frequency-pole pair. The old version worked on the matrix in
   global memory: ~mfit full read+write sweeps of a ~25 KB matrix per solve
   made it DRAM-bound. Two consequences of the caller's structure are used:

   * the inverted matrix itself is dead - mrqcof_start rezeroes covar before
	 mrqcof2 accumulates into it, and mrqmin_2_end copies that fresh
	 accumulation - so neither the solved matrix nor the final
	 column-unscramble pass of the classic routine is needed; only da (the
	 step) and the return code leave this function;

   * pivot selection order matches the original except for exact |value| ties
	 across the different scan partitioning.

   The caller passes the shared-memory size:
   (mfit1*(mfit1|1) + mfit1 + 1) * sizeof(double). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "globals_CUDA.h"
#include <device_launch_parameters.h>

__device__ int gauss_errc_shared(freq_context* CUDA_LCC, const int ma)
{
	const int mf = CUDA_mfit;
	const int mf1 = CUDA_mfit1;
	const int tid = threadIdx.x;
	const int nthreads = blockDim.x;
	const int stride = mf1 | 1; /* odd stride: conflict-free column walks */

	extern __shared__ double sh[];
	double* __restrict__ cov = sh;                        /* [mf1][stride], row 0 unused */
	double* __restrict__ das = sh + (size_t)mf1 * stride; /* [mf1] */

	__shared__ double sh_big[CUDA_BLOCK_DIM];
	__shared__ short sh_irow[CUDA_BLOCK_DIM], sh_icol[CUDA_BLOCK_DIM];
	__shared__ short ipiv[DYT_STRIDE];
	__shared__ double pivinv_s;
	__shared__ int icol_s, irow_s, err_s;

	const double damp = 1 + (*CUDA_LCC).Alamda;

	/* stage the damped normal matrix; covar never touches global memory */
	for (int x = mf1 + 1 + tid; x < mf1 * mf1; x += nthreads)
	{
		const int j = x / mf1, k = x - j * mf1;
		if (k == 0) continue; /* column 0 is never read */
		double v = (*CUDA_LCC).alpha[x];
		if (j == k) v *= damp;
		cov[j * stride + k] = v;
	}
	for (int x = 1 + tid; x <= mf; x += nthreads)
	{
		das[x] = (*CUDA_LCC).beta[x];
		ipiv[x] = 0;
	}
	if (tid == 0) err_s = 0;
	__syncthreads();

	for (int i = 1; i <= mf; i++)
	{
		/* full-pivot search: thread j scans row j */
		double big = 0.0;
		int irow = 0, licol = 0;
		const int j = 1 + tid;
		if (j <= mf && ipiv[j] != 1)
		{
			double const* __restrict__ rowp = cov + j * stride;
			for (int k = 1; k <= mf; k++)
			{
				const int ii = ipiv[k];
				if (ii == 0)
				{
					const double t = fabs(rowp[k]);
					if (t >= big)
					{
						big = t;
						irow = j;
						licol = k;
					}
				}
				else if (ii > 1)
					err_s = 1; /* all writers store the same value */
			}
		}
		sh_big[tid] = big;
		sh_irow[tid] = (short)irow;
		sh_icol[tid] = (short)licol;
		__syncthreads();

		if (err_s)
			break;

		if (tid == 0)
		{
			double b = sh_big[0];
			int ir = sh_irow[0], ic = sh_icol[0];
			for (int t = 1; t < nthreads; t++)
				if (sh_big[t] >= b)
				{
					b = sh_big[t];
					ir = sh_irow[t];
					ic = sh_icol[t];
				}
			ipiv[ic] += 1;
			icol_s = ic;
			irow_s = ir;
		}
		__syncthreads();

		const int icol = icol_s;
		const int irowg = irow_s;

		if (irowg != icol)
		{
			for (int l = 1 + tid; l <= mf; l += nthreads)
			{
				const double t = cov[irowg * stride + l];
				cov[irowg * stride + l] = cov[icol * stride + l];
				cov[icol * stride + l] = t;
			}
			if (tid == 0)
			{
				const double t = das[irowg];
				das[irowg] = das[icol];
				das[icol] = t;
			}
		}
		__syncthreads();

		const double piv = cov[icol * stride + icol];
		if (piv == 0.0)
		{
			if (tid == 0) err_s = 2;
			__syncthreads();
			break;
		}

		if (tid == 0)
		{
			const double pv = 1.0 / piv;
			pivinv_s = pv;
			cov[icol * stride + icol] = 1.0;
			das[icol] *= pv;
		}
		__syncthreads();
		const double pivinv = pivinv_s;

		for (int x = 1 + tid; x <= mf; x += nthreads)
			cov[icol * stride + x] *= pivinv;
		__syncthreads();

		/* eliminate all other rows: warp per row, lanes over columns */
		const int wid = tid >> 5, lane = tid & 31, nwarps = nthreads >> 5;
		for (int ll = 1 + wid; ll <= mf; ll += nwarps)
		{
			if (ll == icol) continue;
			const double dum = cov[ll * stride + icol];
			__syncwarp();
			double const* __restrict__ prow = cov + icol * stride;
			double* __restrict__ lrow = cov + ll * stride;
			for (int c = 1 + lane; c <= mf; c += 32)
			{
				const double base = (c == icol) ? 0.0 : lrow[c];
				lrow[c] = base - prow[c] * dum;
			}
			if (lane == 0)
				das[ll] -= das[icol] * dum;
		}
		__syncthreads();
	}

	/* the step vector goes back to global (mrqmin uses it for atry, and
	   mrqmin_2_end copies it into beta on success) */
	for (int x = 1 + tid; x <= mf; x += nthreads)
		(*CUDA_LCC).da[x] = das[x];
	__syncthreads();

	return err_s;
}
