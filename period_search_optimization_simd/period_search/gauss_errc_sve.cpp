/* from Numerical Recipes */

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string.h>
#include "declarations.h"
#include "CalcStrategySve.hpp"

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
* @note The function modifies the covariance matrix covar in place.
*
* @source Numerical Recipes
*
* @date 8.11.2006
*/
#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int &error)
{
	int i, icol = 0, irow = 0, j, k, l, ll;
	double big, dum, pivinv, temp;

	auto& a = gl.covar;

	std::vector<int> indxc(n + 1 + 1, 0);
	std::vector<int> indxr(n + 1 + 1, 0);
	std::vector<int> ipiv(n + 1 + 1, 0);

	const svbool_t pt = svptrue_b64();
	const int cnt = static_cast<int>(svcntd());

	/* Pivot search. The ipiv state changes only once per pivot step, so the column
	   bookkeeping is hoisted out of the row loop (O(n) per step instead of O(n*n))
	   and the inner loop is left branch-free: a per-lane running maximum plus the
	   index that produced it, reduced once per step. Unlike the fixed-width kernels
	   the columns need no padding - the governing predicate keeps the tail lanes out
	   of the load, and the column mask rejects them from the comparison. */
	std::vector<double> colmask(n > 0 ? n : 1, 0.0);	// negative = column still free, 0.0 = taken
	double max_val[SVE_MAX_LANES] = {};
	double max_idx[SVE_MAX_LANES] = {};
	double lane_init[SVE_MAX_LANES] = {};
	double best;

	for (k = 0; k < cnt; k++)
		lane_init[k] = static_cast<double>(k);

	const svfloat64_t avx_zero = svdup_n_f64(0.0);
	const svfloat64_t avx_reject = svdup_n_f64(-1.0);	// |a| >= 0, so a negative lane never wins
	const svfloat64_t avx_step = svdup_n_f64(static_cast<double>(cnt));
	const svfloat64_t avx_lane = svld1_f64(pt, lane_init);

	for (i = 1; i <= n; i++)
	{
		for (k = 0; k < n; k++)
		{
			if (ipiv[k] > 1)
			{
				error = 1;

				return;
			}

			colmask[k] = ipiv[k] ? 0.0 : -1.0;
		}

		svfloat64_t avx_big = svdup_n_f64(0.0);		// per-lane running 'big', starts at 0.0
		svfloat64_t avx_idx = svdup_n_f64(-1.0);	// per-lane winning index j*n+k, -1 = none

		for (j = 0; j < n; j++)
		{
			if (ipiv[j] == 1) continue;

			svfloat64_t avx_cur = svadd_f64_x(pt, svdup_n_f64(static_cast<double>(j) * n), avx_lane);

			for (k = 0; k < n; k += cnt)
			{
				const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(k), static_cast<int64_t>(n));
				const svbool_t free_col = svcmplt_f64(pg, svld1_f64(pg, &colmask[k]), avx_zero);

				svfloat64_t avx_abs = svabs_f64_x(pt, svld1_f64(pg, &a[j][k]));
				avx_abs = svsel_f64(free_col, avx_abs, avx_reject);

				const svbool_t avx_ge = svcmpge_f64(pt, avx_abs, avx_big);
				avx_big = svsel_f64(avx_ge, avx_abs, avx_big);
				avx_idx = svsel_f64(avx_ge, avx_cur, avx_idx);
				avx_cur = svadd_f64_x(pt, avx_cur, avx_step);
			}
		}

		/* Largest value wins; ties go to the largest index, which reproduces the
		   scalar '>=' ("the last one seen wins") exactly. */
		svst1_f64(pt, max_val, avx_big);
		svst1_f64(pt, max_idx, avx_idx);
		big = max_val[0];
		best = max_idx[0];

		for (l = 1; l < cnt; l++)
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
			irow = idx / n;
			icol = idx - irow * n;
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
			error = 2;

			return;
		}

		pivinv = 1.0 / a[icol][icol];
		const svfloat64_t avx_pivinv = svdup_n_f64(pivinv);
		a[icol][icol] = 1.0;

		for (l = 0; l < n; l += cnt)
		{
			const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(l), static_cast<int64_t>(n));
			svfloat64_t avx_a1 = svld1_f64(pg, &a[icol][l]);
			avx_a1 = svmul_f64_x(pg, avx_a1, avx_pivinv);
			svst1_f64(pg, &a[icol][l], avx_a1);
		}

		b[icol] *= pivinv;

		for (ll = 0; ll < n; ll++)
		{
			if (ll != icol)
			{
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				const svfloat64_t avx_dum = svdup_n_f64(dum);

				for (l = 0; l < n; l += cnt)
				{
					const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(l), static_cast<int64_t>(n));
					svfloat64_t avx_a = svld1_f64(pg, &a[ll][l]);
					svfloat64_t avx_aa = svld1_f64(pg, &a[icol][l]);
					svfloat64_t avx_result = svmls_f64_x(pg, avx_a, avx_aa, avx_dum);
					svst1_f64(pg, &a[ll][l], avx_result);
				}

				b[ll] -= b[icol] * dum;
			}
		}
	}

	for (l = n; l >= 1; l--)
	{
		if (indxr[l] != indxc[l])
			for (k = 0; k < n; k++)
			{
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
			}
	}

	error = 0;

	return;
}
#undef SWAP
