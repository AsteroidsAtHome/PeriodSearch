#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdlib>
#include <vector>
#include <immintrin.h>
#include "declarations.h"
#include "CalcStrategyAvx512.hpp"

#if defined(__GNUC__)
__attribute__((target("avx512f,avx512dq")))
#endif

/**
* @brief Solves a linear system of equations using Gaussian elimination with partial pivoting.
*
* This function implements the Gaussian elimination algorithm with partial pivoting to solve a
* linear system of equations. It rearranges the covariance matrix and the right-hand side vector
* to find the solution.
*
* @param gl A reference to a globals structure containing the covariance matrix and other global data.
* @param n The dimension of the system (number of equations/variables).
* @param b A vector of doubles representing the right-hand side vector of the system.
* @param error An integer reference to store error codes:
*              - 0: No error
*              - 1: Singular matrix
*              - 2: Zero pivot element
*
* @note The function modifies the covariance matrix `covar` in place.
*
* @source Numerical Recipes
*
* @date 8.11.2006
*/
void CalcStrategyAvx512::gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int& error)
{
	//int * indxc, * indxr, * ipiv;
	int i, icol = 0, irow = 0, j, k, l, ll;
	double big, dum, pivinv, temp;

	auto& a = gl.covar;

	//indxc = vector_int(n + 1);
	std::vector<int> indxc(n + 1 + 1, 0);
	//indxr = vector_int(n + 1);
	std::vector<int> indxr(n + 1 + 1, 0);
	//ipiv = vector_int(n + 1);
	//memset(ipiv, 0, n * sizeof(int));
	std::vector<int> ipiv(n + 1 + 1, 0);

	/* Pivot search. The ipiv state changes only once per pivot step, so the column
	   bookkeeping is hoisted out of the row loop (O(n) per step instead of O(n*n))
	   and the inner loop is left branch-free: a per-lane running maximum plus the
	   index that produced it, reduced once per step. */
	const int P = (n + 7) & ~7;						// columns rounded up to a whole vector
	std::vector<unsigned char> colmask(P >> 3, 0);	// one bit per column, set = still free
	alignas(64) double max_val[8], max_idx[8];
	double best;

	__m512d sign_mask = _mm512_set1_pd(-0.0);
	__m512d avx_step = _mm512_set1_pd(8.0);
	__m512d avx_lane = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

	for (i = 1; i <= n; i++)
	{
		for (k = 0; k < P; k += 8)
		{
			__mmask8 m = 0;

			for (l = 0; l < 8 && k + l < n; l++)		// columns past n stay masked off
			{
				if (ipiv[k + l] > 1)
				{
					error = 1;
					return;
				}

				if (ipiv[k + l] == 0) m |= static_cast<__mmask8>(1u << l);
			}

			colmask[k >> 3] = m;
		}

		__m512d avx_big = _mm512_setzero_pd();	// per-lane running 'big', starts at 0.0
		__m512d avx_idx = _mm512_set1_pd(-1.0);	// per-lane winning index j*P+k, -1 = none

		for (j = 0; j < n; j++)
		{
			if (ipiv[j] == 1) continue;

			__m512d avx_cur = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j) * P), avx_lane);

			for (k = 0; k < P; k += 8)
			{
				const __mmask8 cmk = colmask[k >> 3];
				// maskz load: used columns and the k >= n padding are never touched
				__m512d avx_abs = _mm512_andnot_pd(sign_mask, _mm512_maskz_loadu_pd(cmk, &a[j][k]));

				__mmask8 avx_ge = _mm512_mask_cmp_pd_mask(cmk, avx_abs, avx_big, _CMP_GE_OQ);
				avx_big = _mm512_mask_mov_pd(avx_big, avx_ge, avx_abs);
				avx_idx = _mm512_mask_mov_pd(avx_idx, avx_ge, avx_cur);
				avx_cur = _mm512_add_pd(avx_cur, avx_step);
			}
		}

		/* Largest value wins; ties go to the largest index, which reproduces the
		   scalar '>=' ("the last one seen wins") exactly. */
		_mm512_store_pd(max_val, avx_big);
		_mm512_store_pd(max_idx, avx_idx);
		big = max_val[0];
		best = max_idx[0];

		for (l = 1; l < 8; l++)
		{
			if (max_val[l] > big || (max_val[l] == big && max_idx[l] > best))
			{
				big = max_val[l];
				best = max_idx[l];
			}
		}

		if (best >= 0.0)
		{
			const int idx = static_cast<int>(best);
			irow = idx / P;
			icol = idx - irow * P;
		}

		++(ipiv[icol]);
		if (irow != icol)
		{
			for (l = 0; l < n; l++) SWAP(a[irow][l], a[icol][l])
				SWAP(b[irow], b[icol])
		}

		indxr[i] = irow;
		indxc[i] = icol;

		if (a[icol][icol] == 0.0)
		{
			//deallocate_vector((void*)indxc);
			//deallocate_vector((void*)ipiv);
			//deallocate_vector((void*)indxr);
			error = 2;

			return;
		}

		pivinv = 1.0 / a[icol][icol];
		__m512d avx_pivinv = _mm512_set1_pd(pivinv);
		a[icol][icol] = 1.0;

		for (l = 0; l + 7 < n; l += 8)
		{
			__m512d avx_a1 = _mm512_load_pd(&a[icol][l]);
			avx_a1 = _mm512_mul_pd(avx_a1, avx_pivinv);
			_mm512_store_pd(&a[icol][l], avx_a1);
		}

		int rem = n - (l - 1);
		if (rem > 0) {
			int rem = n - l;
			__mmask8 mask = (__mmask8)((1 << rem) - 1);
			__m512d avx_a1 = _mm512_maskz_loadu_pd(mask, &a[icol][l]);
			avx_a1 = _mm512_mask_mul_pd(avx_a1, mask, avx_a1, avx_pivinv);
			_mm512_mask_storeu_pd(&a[icol][l], mask, avx_a1);
		}

		b[icol] *= pivinv;
		for (ll = 0; ll < n; ll++)
		{
			if (ll != icol)
			{
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				__m512d avx_dum = _mm512_set1_pd(dum);
				for (l = 0; l + 7 < n; l += 8)
				{
					__m512d avx_a = _mm512_load_pd(&a[ll][l]);
					__m512d avx_aa = _mm512_load_pd(&a[icol][l]);
					avx_a = _mm512_fnmadd_pd(avx_aa, avx_dum, avx_a);
					_mm512_store_pd(&a[ll][l], avx_a);
				}

				int rem = n - (l - 1);
				if (rem > 0) {
					int rem = n - l;
					__mmask8 mask = (__mmask8)((1 << rem) - 1);
					__m512d avx_a = _mm512_maskz_loadu_pd(mask, &a[ll][l]);
					__m512d avx_aa = _mm512_maskz_loadu_pd(mask, &a[icol][l]);
					avx_a = _mm512_mask_fnmadd_pd(avx_aa, mask, avx_dum, avx_a);
					_mm512_mask_store_pd(&a[ll][l], mask, avx_a);
				}

				b[ll] -= b[icol] * dum;
			}
		}
	}

	for (l = n; l >= 1; l--)
	{
		if (indxr[l] != indxc[l])
			for (k = 0; k < n; k++)
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
	}

	//deallocate_vector((void*)indxc);
	//deallocate_vector((void*)ipiv);
	//deallocate_vector((void*)indxr);
	error = 0;
	return;
}
#undef SWAP
