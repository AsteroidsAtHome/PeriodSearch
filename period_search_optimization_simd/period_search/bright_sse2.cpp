#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <emmintrin.h>  // SSE2
#include "CalcStrategySse2.hpp"

/**
 * @brief Lane-wise select (SSE2 has no blendv): mask ? b : a.
 */
#if defined(__GNUC__)
__attribute__((target("sse2")))
#endif
static inline __m128d select_pd(const __m128d a, const __m128d b, const __m128d mask)
{
	return _mm_or_pd(_mm_and_pd(mask, b), _mm_andnot_pd(mask, a));
}

/**
 * @brief dyda[2v .. 2v + 2W) = Scale * sum_j dbr[j] * Dg_row[j][v .. v + W)
 *
 * W accumulators stay in registers, so each visible Dg row is streamed only once per chunk. Facets are summed
 * sequentially (including the zero-weight pair padding), i.e. in the same order as the original loop.
 * Relies on the padding entry at dbr[incl_count] and on valid row pointers up to Dg_row[incl_count + DG_PREFETCH_ROWS].
 */
template <int W>
#if defined(__GNUC__)
__attribute__((target("sse2"), always_inline))
#endif
static inline void dg_accumulate(__m128d* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda, const __m128d avx_Scale)
{
	// Named accumulators (not an array) so that gcc keeps all of them in registers; unused ones are optimized away.
	__m128d a0 = _mm_setzero_pd(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0;
	__m128d a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0, a12 = a0, a13 = a0;

#define DG_MAC(k) if (W > k) a##k = _mm_add_pd(a##k, _mm_mul_pd(pdbr, _mm_load_pd(p + 2 * k)))
	const int n = (incl_count + 1) & ~1;
	for (int j = 0; j < n; j++)
	{
		// Dg rows exceed L1 in total and are visited in a data dependent order, so fetch a few rows ahead
		const char* pf = reinterpret_cast<const char*>(Dg_row[j + DG_PREFETCH_ROWS] + v);
		for (int b = 0; b < W * 16; b += 64)
			_mm_prefetch(pf + b, _MM_HINT_T0);

		const double* p = reinterpret_cast<const double*>(Dg_row[j] + v);
		const __m128d pdbr = _mm_load1_pd(&dbr[j]);
		DG_MAC(0); DG_MAC(1); DG_MAC(2); DG_MAC(3); DG_MAC(4); DG_MAC(5); DG_MAC(6);
		DG_MAC(7); DG_MAC(8); DG_MAC(9); DG_MAC(10); DG_MAC(11); DG_MAC(12); DG_MAC(13);
	}
#undef DG_MAC

#define DG_STORE(k) if (W > k) _mm_store_pd(&dyda[2 * (v + k)], _mm_mul_pd(a##k, avx_Scale))
	DG_STORE(0); DG_STORE(1); DG_STORE(2); DG_STORE(3); DG_STORE(4); DG_STORE(5); DG_STORE(6);
	DG_STORE(7); DG_STORE(8); DG_STORE(9); DG_STORE(10); DG_STORE(11); DG_STORE(12); DG_STORE(13);
#undef DG_STORE
}

#if defined(__GNUC__)
__attribute__((target("sse2")))
#endif
static void dg_chunk(const int w, __m128d* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda, const __m128d avx_Scale)
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
__attribute__((target("sse2")))
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
void CalcStrategySse2::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j, k;
	incl_count = 0;
	double *ee = gl.xx1;
	double *ee0 = gl.xx2;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);				/* Lambert */
	cls = cg[ncoef];						/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php);				/* computes also Scale */

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
	const __m128d avx_e1 = _mm_load1_pd(&e[1]);
	const __m128d avx_e2 = _mm_load1_pd(&e[2]);
	const __m128d avx_e3 = _mm_load1_pd(&e[3]);
	const __m128d avx_e01 = _mm_load1_pd(&e0[1]);
	const __m128d avx_e02 = _mm_load1_pd(&e0[2]);
	const __m128d avx_e03 = _mm_load1_pd(&e0[3]);
	const __m128d avx_de11 = _mm_load1_pd(&de[1][1]);
	const __m128d avx_de12 = _mm_load1_pd(&de[1][2]);
	const __m128d avx_de13 = _mm_load1_pd(&de[1][3]);
	const __m128d avx_de21 = _mm_load1_pd(&de[2][1]);
	const __m128d avx_de22 = _mm_load1_pd(&de[2][2]);
	const __m128d avx_de23 = _mm_load1_pd(&de[2][3]);
	const __m128d avx_de31 = _mm_load1_pd(&de[3][1]);
	const __m128d avx_de32 = _mm_load1_pd(&de[3][2]);
	const __m128d avx_de33 = _mm_load1_pd(&de[3][3]);
	const __m128d avx_de011 = _mm_load1_pd(&de0[1][1]);
	const __m128d avx_de012 = _mm_load1_pd(&de0[1][2]);
	const __m128d avx_de013 = _mm_load1_pd(&de0[1][3]);
	const __m128d avx_de021 = _mm_load1_pd(&de0[2][1]);
	const __m128d avx_de022 = _mm_load1_pd(&de0[2][2]);
	const __m128d avx_de023 = _mm_load1_pd(&de0[2][3]);
	const __m128d avx_de031 = _mm_load1_pd(&de0[3][1]);
	const __m128d avx_de032 = _mm_load1_pd(&de0[3][2]);
	const __m128d avx_de033 = _mm_load1_pd(&de0[3][3]);
	const __m128d avx_Scale = _mm_load1_pd(&Scale);

	const __m128d avx_tiny = _mm_set1_pd(TINY);
	const __m128d avx_cl = _mm_set1_pd(cl), avx_cl1 = _mm_set_pd(1, cl), avx_cls = _mm_set1_pd(cls), avx_11 = _mm_set1_pd(1.0);
	__m128d res_br = _mm_setzero_pd();
	__m128d avx_dyda1 = _mm_setzero_pd();
	__m128d avx_dyda2 = _mm_setzero_pd();
	__m128d avx_dyda3 = _mm_setzero_pd();
	__m128d avx_d = _mm_setzero_pd();
	__m128d avx_d1 = _mm_setzero_pd();

#ifdef __GNUC__
	double g[2] __attribute__((aligned(16)));
#else
	alignas(16) double g[2];
#endif

	for (i = 0; i < Numfac; i += 2)
	{
		const __m128d avx_Nor1 = _mm_load_pd(&gl.Nor[0][i]);
		const __m128d avx_Nor2 = _mm_load_pd(&gl.Nor[1][i]);
		const __m128d avx_Nor3 = _mm_load_pd(&gl.Nor[2][i]);

		__m128d avx_lmu = _mm_mul_pd(avx_e1, avx_Nor1);
		avx_lmu = _mm_add_pd(avx_lmu, _mm_mul_pd(avx_e2, avx_Nor2));
		avx_lmu = _mm_add_pd(avx_lmu, _mm_mul_pd(avx_e3, avx_Nor3));
		__m128d avx_lmu0 = _mm_mul_pd(avx_e01, avx_Nor1);
		avx_lmu0 = _mm_add_pd(avx_lmu0, _mm_mul_pd(avx_e02, avx_Nor2));
		avx_lmu0 = _mm_add_pd(avx_lmu0, _mm_mul_pd(avx_e03, avx_Nor3));

		const __m128d cmp = _mm_and_pd(_mm_cmpgt_pd(avx_lmu, avx_tiny), _mm_cmpgt_pd(avx_lmu0, avx_tiny));
		const int icmp = _mm_movemask_pd(cmp);
		if (!icmp)
			continue;

		// Hidden lanes get a unit denominator (keeps everything finite) and zero area (contributes nothing);
		// visible lanes are evaluated with exactly the same operations as before.
		const __m128d avx_Area = _mm_and_pd(_mm_load_pd(&gl.Area[i]), cmp);
		const __m128d avx_dnom = select_pd(avx_11, _mm_add_pd(avx_lmu, avx_lmu0), cmp);
		const __m128d avx_q = _mm_mul_pd(avx_lmu, avx_lmu0);
		const __m128d avx_s = _mm_mul_pd(avx_q, _mm_add_pd(avx_cl, _mm_div_pd(avx_cls, avx_dnom)));

		// dbr = Darea * s,  s = mu * mu0 * (cl + cls / (mu + mu0))
		_mm_store_pd(g, _mm_mul_pd(_mm_load_pd(&gl.Darea[i]), avx_s));
		res_br = _mm_add_pd(res_br, _mm_mul_pd(avx_Area, avx_s));

		// dsmu = cls * (mu0 / (mu + mu0))^2 + cl * mu0,  dsmu0 = cls * (mu / (mu + mu0))^2 + cl * mu
		__m128d avx_powdnom = _mm_div_pd(avx_lmu0, avx_dnom);
		avx_powdnom = _mm_mul_pd(avx_powdnom, avx_powdnom);
		const __m128d avx_dsmu = _mm_add_pd(_mm_mul_pd(avx_cls, avx_powdnom), _mm_mul_pd(avx_cl, avx_lmu0));
		avx_powdnom = _mm_div_pd(avx_lmu, avx_dnom);
		avx_powdnom = _mm_mul_pd(avx_powdnom, avx_powdnom);
		const __m128d avx_dsmu0 = _mm_add_pd(_mm_mul_pd(avx_cls, avx_powdnom), _mm_mul_pd(avx_cl, avx_lmu));

		// rotation derivatives: (Nor . de[:, j]) * dsmu + (Nor . de0[:, j]) * dsmu0
		__m128d avx_sum1 = _mm_mul_pd(avx_Nor1, avx_de11);
		avx_sum1 = _mm_add_pd(avx_sum1, _mm_mul_pd(avx_Nor2, avx_de21));
		avx_sum1 = _mm_add_pd(avx_sum1, _mm_mul_pd(avx_Nor3, avx_de31));
		__m128d avx_sum10 = _mm_mul_pd(avx_Nor1, avx_de011);
		avx_sum10 = _mm_add_pd(avx_sum10, _mm_mul_pd(avx_Nor2, avx_de021));
		avx_sum10 = _mm_add_pd(avx_sum10, _mm_mul_pd(avx_Nor3, avx_de031));
		__m128d avx_sum2 = _mm_mul_pd(avx_Nor1, avx_de12);
		avx_sum2 = _mm_add_pd(avx_sum2, _mm_mul_pd(avx_Nor2, avx_de22));
		avx_sum2 = _mm_add_pd(avx_sum2, _mm_mul_pd(avx_Nor3, avx_de32));
		__m128d avx_sum20 = _mm_mul_pd(avx_Nor1, avx_de012);
		avx_sum20 = _mm_add_pd(avx_sum20, _mm_mul_pd(avx_Nor2, avx_de022));
		avx_sum20 = _mm_add_pd(avx_sum20, _mm_mul_pd(avx_Nor3, avx_de032));
		__m128d avx_sum3 = _mm_mul_pd(avx_Nor1, avx_de13);
		avx_sum3 = _mm_add_pd(avx_sum3, _mm_mul_pd(avx_Nor2, avx_de23));
		avx_sum3 = _mm_add_pd(avx_sum3, _mm_mul_pd(avx_Nor3, avx_de33));
		__m128d avx_sum30 = _mm_mul_pd(avx_Nor1, avx_de013);
		avx_sum30 = _mm_add_pd(avx_sum30, _mm_mul_pd(avx_Nor2, avx_de023));
		avx_sum30 = _mm_add_pd(avx_sum30, _mm_mul_pd(avx_Nor3, avx_de033));

		avx_dyda1 = _mm_add_pd(avx_dyda1, _mm_mul_pd(avx_Area, _mm_add_pd(_mm_mul_pd(avx_sum1, avx_dsmu), _mm_mul_pd(avx_sum10, avx_dsmu0))));
		avx_dyda2 = _mm_add_pd(avx_dyda2, _mm_mul_pd(avx_Area, _mm_add_pd(_mm_mul_pd(avx_sum2, avx_dsmu), _mm_mul_pd(avx_sum20, avx_dsmu0))));
		avx_dyda3 = _mm_add_pd(avx_dyda3, _mm_mul_pd(avx_Area, _mm_add_pd(_mm_mul_pd(avx_sum3, avx_dsmu), _mm_mul_pd(avx_sum30, avx_dsmu0))));

		// derivatives w.r.t. cl, cls
		avx_d = _mm_add_pd(avx_d, _mm_mul_pd(avx_q, avx_Area));
		avx_d1 = _mm_add_pd(avx_d1, _mm_div_pd(_mm_mul_pd(_mm_mul_pd(avx_Area, avx_lmu), avx_lmu0), avx_dnom));

		// Branchless compaction of the visible facets: always write the slot, advance only when the lane is visible.
		for (int l = 0; l < 2; l++)
		{
			Dg_row[incl_count] = reinterpret_cast<__m128d*>(gl.Dg[i + l]);
			dbr[incl_count] = g[l];
			incl_count += (icmp >> l) & 1;
		}
	}

	// zero-weight padding entry for the pairwise order, valid (unused) rows for the prefetch look-ahead
	dbr[incl_count] = 0.0;
	for (j = 0; j <= DG_PREFETCH_ROWS; j++)
		Dg_row[incl_count + j] = reinterpret_cast<__m128d*>(gl.Dg[0]);

	res_br = _mm_add_pd(res_br, _mm_shuffle_pd(res_br, _mm_setzero_pd(), 1));
	gl.ymod = _mm_cvtsd_f64(res_br);

	/* Derivatives of brightness w.r.t. g-coefficients, in balanced chunks of at most 14 vectors (accumulators in registers) */
	const int nvec = (ncoef0 - 3 + 1) / 2;
	const int nchunks = (nvec + 13) / 14;
	const int wchunk = nchunks > 0 ? (nvec + nchunks - 1) / nchunks : 1;
	for (int v = 0; v < nvec; v += wchunk)
		dg_chunk(nvec - v < wchunk ? nvec - v : wchunk, Dg_row, dbr, incl_count, v, gl.dyda, avx_Scale);

	/* Derivatives of brightness w.r.t. rotation parameters */
	avx_dyda1 = _mm_shuffle_pd(
		  _mm_add_pd(avx_dyda1, _mm_shuffle_pd(avx_dyda1, _mm_setzero_pd(), 1)),
		  _mm_add_pd(avx_dyda2, _mm_shuffle_pd(avx_dyda2, _mm_setzero_pd(), 1)),
		  0);

	avx_dyda1 = _mm_mul_pd(avx_dyda1, avx_Scale);
	_mm_storeu_pd(&gl.dyda[ncoef0 - 3 + 1 - 1], avx_dyda1); //unaligned memory because of odd index
	avx_dyda3 = _mm_add_pd(avx_dyda3, _mm_shuffle_pd(avx_dyda3, _mm_setzero_pd(), 1));
	avx_dyda3 = _mm_mul_pd(avx_dyda3, avx_Scale);
	gl.dyda[ncoef0 - 3 + 3 - 1] = _mm_cvtsd_f64(avx_dyda3);

	/* Derivatives of br. w.r.t. cl, cls */
	avx_d = _mm_shuffle_pd(
		  _mm_add_pd(avx_d, _mm_shuffle_pd(avx_d, _mm_setzero_pd(), 1)),
		  _mm_add_pd(avx_d1, _mm_shuffle_pd(avx_d1, _mm_setzero_pd(), 1)),
		  0);

	avx_d = _mm_mul_pd(avx_d, avx_Scale);
	avx_d = _mm_mul_pd(avx_d, avx_cl1);
	_mm_storeu_pd(&gl.dyda[ncoef - 1 - 1], avx_d); //unaligned memory because of odd index

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
	{
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];
	}

	/* Scaled brightness */
	gl.ymod *= Scale;
}
