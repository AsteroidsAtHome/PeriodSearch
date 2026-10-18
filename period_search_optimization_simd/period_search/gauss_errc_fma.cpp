/* from Numerical Recipes */

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdlib>
#include <vector>
#include <immintrin.h>
#include "declarations.h"
#include "CalcStrategyFma.hpp"

#if defined(__GNUC__)
__attribute__((target("avx,fma")))
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
void CalcStrategyFma::gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int& error)
{
	//int * indxc,  * indxr, * ipiv;
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
	const int P = (n + 3) & ~3;				// columns rounded up to a whole vector
	std::vector<double> colmask(P, 0.0);	// negative = column still free, 0.0 = taken
	alignas(32) double max_val[4], max_idx[4];
	double best;

	__m256d sign_mask = _mm256_set1_pd(-0.0);
	__m256d avx_reject = _mm256_set1_pd(-1.0);	// |a| >= 0, so a negative lane never wins
	__m256d avx_step = _mm256_set1_pd(4.0);
	__m256d avx_lane = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);

	for (i = 1; i <= n; i++)
	{
		for (k = 0; k < n; k++)
		{
			if (ipiv[k] > 1)
			{
				error = 1;
				return;
			}

			colmask[k] = ipiv[k] ? 0.0 : -1.0;	// blendv only looks at the sign bit
		}

		for (k = n; k < P; k++)
			colmask[k] = 0.0;					// padding lanes can never win

		__m256d avx_big = _mm256_setzero_pd();	// per-lane running 'big', starts at 0.0
		__m256d avx_idx = _mm256_set1_pd(-1.0);	// per-lane winning index j*P+k, -1 = none

		for (j = 0; j < n; j++)
		{
			if (ipiv[j] == 1) continue;

			__m256d avx_cur = _mm256_add_pd(_mm256_set1_pd(static_cast<double>(j) * P), avx_lane);

			for (k = 0; k < P; k += 4)
			{
				__m256d avx_abs = _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(&a[j][k])); // abs
				avx_abs = _mm256_blendv_pd(avx_reject, avx_abs, _mm256_loadu_pd(&colmask[k]));

				__m256d avx_ge = _mm256_cmp_pd(avx_abs, avx_big, _CMP_GE_OQ);
				avx_big = _mm256_blendv_pd(avx_big, avx_abs, avx_ge);
				avx_idx = _mm256_blendv_pd(avx_idx, avx_cur, avx_ge);
				avx_cur = _mm256_add_pd(avx_cur, avx_step);
			}
		}

		/* Largest value wins; ties go to the largest index, which reproduces the
		   scalar '>=' ("the last one seen wins") exactly. */
		_mm256_store_pd(max_val, avx_big);
		_mm256_store_pd(max_idx, avx_idx);
		big = max_val[0];
		best = max_idx[0];

		for (l = 1; l < 4; l++)
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

		if (a[icol][icol] == 0.0) {
			//deallocate_vector((void*)indxc);
			//deallocate_vector((void*)indxr);
			//deallocate_vector((void*)ipiv);
			error = 2;

			return;
		}

		pivinv = 1.0 / a[icol][icol];
		__m256d avx_pivinv = _mm256_set1_pd(pivinv);
		a[icol][icol] = 1.0;
		int cyklus = (n >> 2) << 2;

		for (l = 0; l < cyklus; l += 4)
		{
			__m256d avx_a1 = _mm256_load_pd(&a[icol][l]);
			avx_a1 = _mm256_mul_pd(avx_a1, avx_pivinv);
			_mm256_store_pd(&a[icol][l], avx_a1);
		}

		if (l < n) a[icol][l] *= pivinv; //last odd value
		if (l + 1 < n) a[icol][l + 1] *= pivinv; //last odd value
		if (l + 2 < n) a[icol][l + 2] *= pivinv; //last odd value

		b[icol] *= pivinv;

		for (ll = 0; ll < n; ll++)
		{
			if (ll != icol)
			{
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				__m256d avx_dum = _mm256_set1_pd(dum);

				for (l = 0; l < cyklus; l += 4)
				{
					__m256d avx_a = _mm256_load_pd(&a[ll][l]);
					__m256d avx_aa = _mm256_load_pd(&a[icol][l]);
                    avx_a = _mm256_fnmadd_pd(avx_aa, avx_dum, avx_a);
					_mm256_store_pd(&a[ll][l], avx_a);
				}

				if (l < n) a[ll][l] -= a[icol][l] * dum; //last odd value
				if (l + 1 < n) a[ll][l + 1] -= a[icol][l + 1] * dum; //last odd value
				if (l + 2 < n) a[ll][l + 2] -= a[icol][l + 2] * dum; //last odd value

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
	//deallocate_vector((void*)indxr);
	//deallocate_vector((void*)ipiv);
	error = 0;

	return;
}
#undef SWAP
