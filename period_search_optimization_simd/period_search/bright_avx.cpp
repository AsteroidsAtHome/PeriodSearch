#include <cmath>
#include <cstdio>
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <immintrin.h>
#include "CalcStrategyAvx.hpp"

/**
 * @brief Pairwise horizontal sums: lane 0 = (a0 + a1) + (a2 + a3), lane 1 = (b0 + b1) + (b2 + b3).
 */
#if defined(__GNUC__)
__attribute__((target("avx")))
#endif
static inline __m256d hsum2_pd(const __m256d a, const __m256d b)
{
	const __m256d s = _mm256_hadd_pd(a, b);
	return _mm256_add_pd(s, _mm256_permute2f128_pd(s, s, 1));
}

/**
 * @brief dyda[4v .. 4v + 4W) = Scale * sum_j dbr[j] * Dg_row[j][v .. v + W)
 *
 * W accumulators stay in registers, so each visible Dg row is streamed only once per chunk. Facets are summed
 * sequentially (including the zero-weight pair padding), i.e. in the same order as the original pairwise loop.
 * Relies on the padding entry at dbr[incl_count] and on valid row pointers up to Dg_row[incl_count + DG_PREFETCH_ROWS].
 */
template <int W>
#if defined(__GNUC__)
__attribute__((target("avx"), always_inline))
#endif
static inline void dg_accumulate(__m256d* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda, const __m256d avx_Scale)
{
	// Named accumulators (not an array) so that gcc keeps all of them in registers; unused ones are optimized away.
	__m256d a0 = _mm256_setzero_pd(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0;
	__m256d a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0, a12 = a0, a13 = a0;

#define DG_MAC(k) if (W > k) a##k = _mm256_add_pd(a##k, _mm256_mul_pd(pdbr, _mm256_load_pd(p + 4 * k)))
	const int n = (incl_count + 1) & ~1;
	for (int j = 0; j < n; j++)
	{
		// Dg rows exceed L1 in total and are visited in a data dependent order, so fetch a few rows ahead
		const char* pf = reinterpret_cast<const char*>(Dg_row[j + DG_PREFETCH_ROWS] + v);
		for (int b = 0; b < W * 32; b += 64)
			_mm_prefetch(pf + b, _MM_HINT_T0);

		const double* p = reinterpret_cast<const double*>(Dg_row[j] + v);
		const __m256d pdbr = _mm256_broadcast_sd(&dbr[j]);
		DG_MAC(0); DG_MAC(1); DG_MAC(2); DG_MAC(3); DG_MAC(4); DG_MAC(5); DG_MAC(6);
		DG_MAC(7); DG_MAC(8); DG_MAC(9); DG_MAC(10); DG_MAC(11); DG_MAC(12); DG_MAC(13);
	}
#undef DG_MAC

#define DG_STORE(k) if (W > k) _mm256_store_pd(&dyda[4 * (v + k)], _mm256_mul_pd(a##k, avx_Scale))
	DG_STORE(0); DG_STORE(1); DG_STORE(2); DG_STORE(3); DG_STORE(4); DG_STORE(5); DG_STORE(6);
	DG_STORE(7); DG_STORE(8); DG_STORE(9); DG_STORE(10); DG_STORE(11); DG_STORE(12); DG_STORE(13);
#undef DG_STORE
}

#if defined(__GNUC__)
__attribute__((target("avx")))
#endif
static void dg_chunk(const int w, __m256d* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda, const __m256d avx_Scale)
{
	switch (w)
	{
		case 14: dg_accumulate<14>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 13: dg_accumulate<13>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 12: dg_accumulate<12>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 11: dg_accumulate<11>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 10: dg_accumulate<10>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 9: dg_accumulate<9>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 8: dg_accumulate<8>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 7: dg_accumulate<7>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 6: dg_accumulate<6>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 5: dg_accumulate<5>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 4: dg_accumulate<4>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 3: dg_accumulate<3>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 2: dg_accumulate<2>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 1: dg_accumulate<1>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		default: break;
	}
}

#if defined(__GNUC__)
__attribute__((target("avx")))
#endif

/**
 * @brief Computes integrated brightness of all visible and illuminated areas and its derivatives.
 *
 * This function calculates the integrated brightness of all visible and illuminated areas based on the provided time `t`,
 * coefficient vector `cg`, and global data. It also computes the derivatives of the brightness with respect to the coefficients.
 *
 * @param t The time at which the brightness is evaluated.
 * @param cg A reference to a vector of doubles containing the coefficients for the brightness calculation.
 * @param ncoef An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables `ymod` and `dyda`.
 *
 * @date 8.11.2006
 * @author Josef Durec
 *
 * @date 29.2.2024 modified by Georgi Vidinski
 */
void CalcStrategyAvx::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j, k;
	incl_count = 0;
	double *ee = gl.xx1;
	double *ee0 = gl.xx2;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);			/* Lambert */
	cls = cg[ncoef];					/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php);			/* computes also Scale */

	matrix(cg[ncoef0], t, tmat, dtm);

	/* Directions (and derivatives) in the rotating system */
	for (i = 1; i <= 3; i++)
	{
		e[i] = 0;
		e0[i] = 0;
		for (j = 1; j <= 3; j++)
		{
			e[i] += tmat[i][j] * ee[j];
			e0[i] += tmat[i][j] * ee0[j];
			de[i][j] = 0;
			de0[i][j] = 0;
			for (k = 1; k <= 3; k++)
			{
				de[i][j] += dtm[j][i][k] * ee[k];
				de0[i][j] += dtm[j][i][k] * ee0[k];
			}
		}
	}

	/*Integrated brightness (phase coefficients used later) */
	const __m256d avx_e1 = _mm256_broadcast_sd(&e[1]);
	const __m256d avx_e2 = _mm256_broadcast_sd(&e[2]);
	const __m256d avx_e3 = _mm256_broadcast_sd(&e[3]);
	const __m256d avx_e01 = _mm256_broadcast_sd(&e0[1]);
	const __m256d avx_e02 = _mm256_broadcast_sd(&e0[2]);
	const __m256d avx_e03 = _mm256_broadcast_sd(&e0[3]);

	const __m256d avx_tiny = _mm256_set1_pd(TINY);
	const __m256d avx_cl = _mm256_set1_pd(cl), avx_cl1 = _mm256_set_pd(0, 0, 1, cl), avx_cls = _mm256_set1_pd(cls), avx_11 = _mm256_set1_pd(1.0);
	const __m256d avx_Scale = _mm256_broadcast_sd(&Scale);
	__m256d res_br = _mm256_setzero_pd();
	__m256d avx_dyda1 = _mm256_setzero_pd();
	__m256d avx_dyda2 = _mm256_setzero_pd();
	__m256d avx_dyda3 = _mm256_setzero_pd();
	__m256d avx_d = _mm256_setzero_pd();
	__m256d avx_d1 = _mm256_setzero_pd();

#ifdef __GNUC__
	double g[4] __attribute__((aligned(64)));
#else
	alignas(64) double g[4];
#endif

	for (i = 0; i < Numfac; i += 4)
	{
		const __m256d avx_Nor1 = _mm256_load_pd(&gl.Nor[0][i]);
		const __m256d avx_Nor2 = _mm256_load_pd(&gl.Nor[1][i]);
		const __m256d avx_Nor3 = _mm256_load_pd(&gl.Nor[2][i]);

		__m256d avx_lmu = _mm256_mul_pd(avx_e1, avx_Nor1);
		avx_lmu = _mm256_add_pd(avx_lmu, _mm256_mul_pd(avx_e2, avx_Nor2));
		avx_lmu = _mm256_add_pd(avx_lmu, _mm256_mul_pd(avx_e3, avx_Nor3));
		__m256d avx_lmu0 = _mm256_mul_pd(avx_e01, avx_Nor1);
		avx_lmu0 = _mm256_add_pd(avx_lmu0, _mm256_mul_pd(avx_e02, avx_Nor2));
		avx_lmu0 = _mm256_add_pd(avx_lmu0, _mm256_mul_pd(avx_e03, avx_Nor3));

		const __m256d cmp = _mm256_and_pd(_mm256_cmp_pd(avx_lmu, avx_tiny, _CMP_GT_OS), _mm256_cmp_pd(avx_lmu0, avx_tiny, _CMP_GT_OS));
		const int icmp = _mm256_movemask_pd(cmp);
		if (!icmp)
			continue;

		// Hidden lanes get a unit denominator (keeps everything finite) and zero area (contributes nothing);
		// visible lanes are evaluated with exactly the same operations as before.
		const __m256d avx_Area = _mm256_and_pd(_mm256_load_pd(&gl.Area[i]), cmp);
		const __m256d avx_dnom = _mm256_blendv_pd(avx_11, _mm256_add_pd(avx_lmu, avx_lmu0), cmp);
		const __m256d avx_q = _mm256_mul_pd(avx_lmu, avx_lmu0);
		const __m256d avx_s = _mm256_mul_pd(avx_q, _mm256_add_pd(avx_cl, _mm256_div_pd(avx_cls, avx_dnom)));

		// dbr = Darea * s,  s = mu * mu0 * (cl + cls / (mu + mu0))
		_mm256_store_pd(g, _mm256_mul_pd(_mm256_load_pd(&gl.Darea[i]), avx_s));
		res_br = _mm256_add_pd(res_br, _mm256_mul_pd(avx_Area, avx_s));

		// dsmu = cls * (mu0 / (mu + mu0))^2 + cl * mu0,  dsmu0 = cls * (mu / (mu + mu0))^2 + cl * mu
		__m256d avx_powdnom = _mm256_div_pd(avx_lmu0, avx_dnom);
		avx_powdnom = _mm256_mul_pd(avx_powdnom, avx_powdnom);
		const __m256d avx_dsmu = _mm256_add_pd(_mm256_mul_pd(avx_cls, avx_powdnom), _mm256_mul_pd(avx_cl, avx_lmu0));
		avx_powdnom = _mm256_div_pd(avx_lmu, avx_dnom);
		avx_powdnom = _mm256_mul_pd(avx_powdnom, avx_powdnom);
		const __m256d avx_dsmu0 = _mm256_add_pd(_mm256_mul_pd(avx_cls, avx_powdnom), _mm256_mul_pd(avx_cl, avx_lmu));

		// rotation derivatives: (Nor . de[:, j]) * dsmu + (Nor . de0[:, j]) * dsmu0
		__m256d avx_sum1 = _mm256_mul_pd(avx_Nor1, _mm256_broadcast_sd(&de[1][1]));
		avx_sum1 = _mm256_add_pd(avx_sum1, _mm256_mul_pd(avx_Nor2, _mm256_broadcast_sd(&de[2][1])));
		avx_sum1 = _mm256_add_pd(avx_sum1, _mm256_mul_pd(avx_Nor3, _mm256_broadcast_sd(&de[3][1])));
		__m256d avx_sum10 = _mm256_mul_pd(avx_Nor1, _mm256_broadcast_sd(&de0[1][1]));
		avx_sum10 = _mm256_add_pd(avx_sum10, _mm256_mul_pd(avx_Nor2, _mm256_broadcast_sd(&de0[2][1])));
		avx_sum10 = _mm256_add_pd(avx_sum10, _mm256_mul_pd(avx_Nor3, _mm256_broadcast_sd(&de0[3][1])));
		__m256d avx_sum2 = _mm256_mul_pd(avx_Nor1, _mm256_broadcast_sd(&de[1][2]));
		avx_sum2 = _mm256_add_pd(avx_sum2, _mm256_mul_pd(avx_Nor2, _mm256_broadcast_sd(&de[2][2])));
		avx_sum2 = _mm256_add_pd(avx_sum2, _mm256_mul_pd(avx_Nor3, _mm256_broadcast_sd(&de[3][2])));
		__m256d avx_sum20 = _mm256_mul_pd(avx_Nor1, _mm256_broadcast_sd(&de0[1][2]));
		avx_sum20 = _mm256_add_pd(avx_sum20, _mm256_mul_pd(avx_Nor2, _mm256_broadcast_sd(&de0[2][2])));
		avx_sum20 = _mm256_add_pd(avx_sum20, _mm256_mul_pd(avx_Nor3, _mm256_broadcast_sd(&de0[3][2])));
		__m256d avx_sum3 = _mm256_mul_pd(avx_Nor1, _mm256_broadcast_sd(&de[1][3]));
		avx_sum3 = _mm256_add_pd(avx_sum3, _mm256_mul_pd(avx_Nor2, _mm256_broadcast_sd(&de[2][3])));
		avx_sum3 = _mm256_add_pd(avx_sum3, _mm256_mul_pd(avx_Nor3, _mm256_broadcast_sd(&de[3][3])));
		__m256d avx_sum30 = _mm256_mul_pd(avx_Nor1, _mm256_broadcast_sd(&de0[1][3]));
		avx_sum30 = _mm256_add_pd(avx_sum30, _mm256_mul_pd(avx_Nor2, _mm256_broadcast_sd(&de0[2][3])));
		avx_sum30 = _mm256_add_pd(avx_sum30, _mm256_mul_pd(avx_Nor3, _mm256_broadcast_sd(&de0[3][3])));

		avx_dyda1 = _mm256_add_pd(avx_dyda1, _mm256_mul_pd(avx_Area, _mm256_add_pd(_mm256_mul_pd(avx_sum1, avx_dsmu), _mm256_mul_pd(avx_sum10, avx_dsmu0))));
		avx_dyda2 = _mm256_add_pd(avx_dyda2, _mm256_mul_pd(avx_Area, _mm256_add_pd(_mm256_mul_pd(avx_sum2, avx_dsmu), _mm256_mul_pd(avx_sum20, avx_dsmu0))));
		avx_dyda3 = _mm256_add_pd(avx_dyda3, _mm256_mul_pd(avx_Area, _mm256_add_pd(_mm256_mul_pd(avx_sum3, avx_dsmu), _mm256_mul_pd(avx_sum30, avx_dsmu0))));

		// derivatives w.r.t. cl, cls
		avx_d = _mm256_add_pd(avx_d, _mm256_mul_pd(avx_q, avx_Area));
		avx_d1 = _mm256_add_pd(avx_d1, _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(avx_Area, avx_lmu), avx_lmu0), avx_dnom));

		// Branchless compaction of the visible facets: always write the slot, advance only when the lane is visible.
		for (int l = 0; l < 4; l++)
		{
			Dg_row[incl_count] = reinterpret_cast<__m256d*>(gl.Dg[i + l]);
			dbr[incl_count] = g[l];
			incl_count += (icmp >> l) & 1;
		}
	}

	// zero-weight padding entry for the pairwise order, valid (unused) rows for the prefetch look-ahead
	dbr[incl_count] = 0.0;
	for (j = 0; j <= DG_PREFETCH_ROWS; j++)
		Dg_row[incl_count + j] = reinterpret_cast<__m256d*>(gl.Dg[0]);

	_mm256_store_pd(g, hsum2_pd(res_br, res_br));
	gl.ymod = g[0];

	/* Derivatives of brightness w.r.t. g-coefficients, in balanced chunks of at most 14 vectors (accumulators in registers) */
	const int nvec = (ncoef0 - 3 + 3) / 4;
	const int nchunks = (nvec + 13) / 14;
	const int wchunk = nchunks > 0 ? (nvec + nchunks - 1) / nchunks : 1;
	for (int v = 0; v < nvec; v += wchunk)
		dg_chunk(nvec - v < wchunk ? nvec - v : wchunk, Dg_row, dbr, incl_count, v, gl.dyda, avx_Scale);

	/* Derivatives of brightness w.r.t. rotation parameters */
	_mm256_store_pd(g, _mm256_mul_pd(hsum2_pd(avx_dyda1, avx_dyda2), avx_Scale));
	gl.dyda[ncoef0 - 3 + 1 - 1] = g[0];
	gl.dyda[ncoef0 - 3 + 2 - 1] = g[1];
	_mm256_store_pd(g, _mm256_mul_pd(hsum2_pd(avx_dyda3, avx_dyda3), avx_Scale));
	gl.dyda[ncoef0 - 3 + 3 - 1] = g[0];

	/* Derivatives of br. w.r.t. cl, cls */
	_mm256_store_pd(g, _mm256_mul_pd(_mm256_mul_pd(hsum2_pd(avx_d, avx_d1), avx_Scale), avx_cl1));
	gl.dyda[ncoef - 1 - 1] = g[0];
	gl.dyda[ncoef - 1] = g[1];

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
	{
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];
	}

	/* Scaled brightness */
	gl.ymod *= Scale;
}
