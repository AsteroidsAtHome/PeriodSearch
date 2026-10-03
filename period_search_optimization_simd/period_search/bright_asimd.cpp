/* computes integrated brightness of all visible and illuminated areas
   and its derivatives

   8.11.2006 - Josef Durec
   25.3.2024 - Pavel Rosicky
*/

#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "CalcStrategyAsimd.hpp"
#include <arm_neon.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define DG_PREFETCH(p) __prefetch(p)
#else
#define DG_PREFETCH(p) __builtin_prefetch(p, 0, 3)
#endif

/**
 * @brief dyda[2v .. 2v + 2W) = Scale * sum_j dbr[j] * Dg_row[j][v .. v + W)
 *
 * W accumulators stay in registers, so each visible Dg row is streamed only once per chunk. Facets are summed
 * sequentially (including the zero-weight pair padding).
 * Relies on the padding entry at dbr[incl_count] and on valid row pointers up to Dg_row[incl_count + DG_PREFETCH_ROWS].
 * With 32 NEON registers up to 26 accumulators (+ weight + loads) fit, i.e. 49 Dg columns (Lmax = Mmax = 6) in one pass.
 */
template <int W>
#if defined(__GNUC__)
__attribute__((always_inline))
#endif
static inline void dg_accumulate(float64x2_t* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda, const float64x2_t avx_Scale)
{
	// Named accumulators (not an array) so that the compiler keeps all of them in registers; unused ones are optimized away.
	float64x2_t a0 = vdupq_n_f64(0.0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0;
	float64x2_t a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0, a12 = a0, a13 = a0;
	float64x2_t a14 = a0, a15 = a0, a16 = a0, a17 = a0, a18 = a0, a19 = a0;
	float64x2_t a20 = a0, a21 = a0, a22 = a0, a23 = a0, a24 = a0, a25 = a0;

#define DG_FMA(k) if (W > k) a##k = vfmaq_f64(a##k, pdbr, vld1q_f64(p + 2 * k))
	const int n = (incl_count + 1) & ~1;
	for (int j = 0; j < n; j++)
	{
		// Dg rows exceed L1 in total and are visited in a data dependent order, so fetch a few rows ahead
		const char* pf = reinterpret_cast<const char*>(Dg_row[j + DG_PREFETCH_ROWS] + v);
		for (int b = 0; b < W * 16; b += 64)
			DG_PREFETCH(pf + b);

		const double* p = reinterpret_cast<const double*>(Dg_row[j] + v);
		const float64x2_t pdbr = vld1q_dup_f64(&dbr[j]);
		DG_FMA(0); DG_FMA(1); DG_FMA(2); DG_FMA(3); DG_FMA(4); DG_FMA(5); DG_FMA(6);
		DG_FMA(7); DG_FMA(8); DG_FMA(9); DG_FMA(10); DG_FMA(11); DG_FMA(12); DG_FMA(13);
		DG_FMA(14); DG_FMA(15); DG_FMA(16); DG_FMA(17); DG_FMA(18); DG_FMA(19);
		DG_FMA(20); DG_FMA(21); DG_FMA(22); DG_FMA(23); DG_FMA(24); DG_FMA(25);
	}
#undef DG_FMA

#define DG_STORE(k) if (W > k) vst1q_f64(&dyda[2 * (v + k)], vmulq_f64(a##k, avx_Scale))
	DG_STORE(0); DG_STORE(1); DG_STORE(2); DG_STORE(3); DG_STORE(4); DG_STORE(5); DG_STORE(6);
	DG_STORE(7); DG_STORE(8); DG_STORE(9); DG_STORE(10); DG_STORE(11); DG_STORE(12); DG_STORE(13);
	DG_STORE(14); DG_STORE(15); DG_STORE(16); DG_STORE(17); DG_STORE(18); DG_STORE(19);
	DG_STORE(20); DG_STORE(21); DG_STORE(22); DG_STORE(23); DG_STORE(24); DG_STORE(25);
#undef DG_STORE
}

static void dg_chunk(const int w, float64x2_t* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda, const float64x2_t avx_Scale)
{
	switch (w)
	{
		case 26: dg_accumulate<26>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 25: dg_accumulate<25>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 24: dg_accumulate<24>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 23: dg_accumulate<23>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 22: dg_accumulate<22>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 21: dg_accumulate<21>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 20: dg_accumulate<20>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 19: dg_accumulate<19>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 18: dg_accumulate<18>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 17: dg_accumulate<17>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 16: dg_accumulate<16>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
		case 15: dg_accumulate<15>(Dg_row, dbr, incl_count, v, dyda, avx_Scale); break;
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

#if defined __GNUG__ && !defined __clang__
__attribute__((__target__("arch=armv8-a+simd")))
#elif defined __GNUG__ && __clang__
// NOTE: The following generates warning: unsupported architecture 'armv8-a+simd' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// __attribute__((target("arch=armv8-a+simd")))
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
 * @date 25.3.2024 modified by Pavel Rosicky
 */
void CalcStrategyAsimd::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j, k;
	incl_count = 0;
	double *ee = gl.xx1;
	double *ee0 = gl.xx2;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);	/* Lambert */
	cls = cg[ncoef];			/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php);	/* computes also Scale */

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
	const float64x2_t avx_e1 = vdupq_n_f64(e[1]);
	const float64x2_t avx_e2 = vdupq_n_f64(e[2]);
	const float64x2_t avx_e3 = vdupq_n_f64(e[3]);
	const float64x2_t avx_e01 = vdupq_n_f64(e0[1]);
	const float64x2_t avx_e02 = vdupq_n_f64(e0[2]);
	const float64x2_t avx_e03 = vdupq_n_f64(e0[3]);
	const float64x2_t avx_de11 = vdupq_n_f64(de[1][1]);
	const float64x2_t avx_de12 = vdupq_n_f64(de[1][2]);
	const float64x2_t avx_de13 = vdupq_n_f64(de[1][3]);
	const float64x2_t avx_de21 = vdupq_n_f64(de[2][1]);
	const float64x2_t avx_de22 = vdupq_n_f64(de[2][2]);
	const float64x2_t avx_de23 = vdupq_n_f64(de[2][3]);
	const float64x2_t avx_de31 = vdupq_n_f64(de[3][1]);
	const float64x2_t avx_de32 = vdupq_n_f64(de[3][2]);
	const float64x2_t avx_de33 = vdupq_n_f64(de[3][3]);
	const float64x2_t avx_de011 = vdupq_n_f64(de0[1][1]);
	const float64x2_t avx_de012 = vdupq_n_f64(de0[1][2]);
	const float64x2_t avx_de013 = vdupq_n_f64(de0[1][3]);
	const float64x2_t avx_de021 = vdupq_n_f64(de0[2][1]);
	const float64x2_t avx_de022 = vdupq_n_f64(de0[2][2]);
	const float64x2_t avx_de023 = vdupq_n_f64(de0[2][3]);
	const float64x2_t avx_de031 = vdupq_n_f64(de0[3][1]);
	const float64x2_t avx_de032 = vdupq_n_f64(de0[3][2]);
	const float64x2_t avx_de033 = vdupq_n_f64(de0[3][3]);
	const float64x2_t avx_Scale = vdupq_n_f64(Scale);

	const float64x2_t avx_tiny = vdupq_n_f64(TINY);
	const float64x2_t avx_cl = vdupq_n_f64(cl);
	const float64x2_t avx_cl1 = vsetq_lane_f64(cl, vdupq_n_f64(1.0), 0);
	const float64x2_t avx_cls = vdupq_n_f64(cls);
	const float64x2_t avx_11 = vdupq_n_f64(1.0);
	float64x2_t res_br = vdupq_n_f64(0.0);
	float64x2_t avx_dyda1 = vdupq_n_f64(0.0);
	float64x2_t avx_dyda2 = vdupq_n_f64(0.0);
	float64x2_t avx_dyda3 = vdupq_n_f64(0.0);
	float64x2_t avx_d = vdupq_n_f64(0.0);
	float64x2_t avx_d1 = vdupq_n_f64(0.0);

	double g[2];

	for (i = 0; i < Numfac; i += 2)
	{
		const float64x2_t avx_Nor1 = vld1q_f64(&gl.Nor[0][i]);
		const float64x2_t avx_Nor2 = vld1q_f64(&gl.Nor[1][i]);
		const float64x2_t avx_Nor3 = vld1q_f64(&gl.Nor[2][i]);

		float64x2_t avx_lmu = vmulq_f64(avx_e1, avx_Nor1);
		avx_lmu = vfmaq_f64(avx_lmu, avx_e2, avx_Nor2);
		avx_lmu = vfmaq_f64(avx_lmu, avx_e3, avx_Nor3);
		float64x2_t avx_lmu0 = vmulq_f64(avx_e01, avx_Nor1);
		avx_lmu0 = vfmaq_f64(avx_lmu0, avx_e02, avx_Nor2);
		avx_lmu0 = vfmaq_f64(avx_lmu0, avx_e03, avx_Nor3);

		const uint64x2_t cmp = vandq_u64(vcgtq_f64(avx_lmu, avx_tiny), vcgtq_f64(avx_lmu0, avx_tiny));
		const int icmp = static_cast<int>((vgetq_lane_u64(cmp, 0) & 1) | ((vgetq_lane_u64(cmp, 1) & 1) << 1));
		if (!icmp)
			continue;

		// Hidden lanes get a unit denominator (keeps everything finite) and zero area (contributes nothing);
		// visible lanes are evaluated with exactly the same operations as before.
		const float64x2_t avx_Area = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vld1q_f64(&gl.Area[i])), cmp));
		const float64x2_t avx_dnom = vbslq_f64(cmp, vaddq_f64(avx_lmu, avx_lmu0), avx_11);
		const float64x2_t avx_q = vmulq_f64(avx_lmu, avx_lmu0);
		const float64x2_t avx_s = vmulq_f64(avx_q, vaddq_f64(avx_cl, vdivq_f64(avx_cls, avx_dnom)));

		// dbr = Darea * s,  s = mu * mu0 * (cl + cls / (mu + mu0))
		vst1q_f64(g, vmulq_f64(vld1q_f64(&gl.Darea[i]), avx_s));
		res_br = vaddq_f64(res_br, vmulq_f64(avx_Area, avx_s));

		// dsmu = cls * (mu0 / (mu + mu0))^2 + cl * mu0,  dsmu0 = cls * (mu / (mu + mu0))^2 + cl * mu
		float64x2_t avx_powdnom = vdivq_f64(avx_lmu0, avx_dnom);
		avx_powdnom = vmulq_f64(avx_powdnom, avx_powdnom);
		const float64x2_t avx_dsmu = vfmaq_f64(vmulq_f64(avx_cls, avx_powdnom), avx_cl, avx_lmu0);
		avx_powdnom = vdivq_f64(avx_lmu, avx_dnom);
		avx_powdnom = vmulq_f64(avx_powdnom, avx_powdnom);
		const float64x2_t avx_dsmu0 = vfmaq_f64(vmulq_f64(avx_cls, avx_powdnom), avx_cl, avx_lmu);

		// rotation derivatives: (Nor . de[:, j]) * dsmu + (Nor . de0[:, j]) * dsmu0
		float64x2_t avx_sum1 = vmulq_f64(avx_Nor1, avx_de11);
		avx_sum1 = vfmaq_f64(avx_sum1, avx_Nor2, avx_de21);
		avx_sum1 = vfmaq_f64(avx_sum1, avx_Nor3, avx_de31);
		float64x2_t avx_sum10 = vmulq_f64(avx_Nor1, avx_de011);
		avx_sum10 = vfmaq_f64(avx_sum10, avx_Nor2, avx_de021);
		avx_sum10 = vfmaq_f64(avx_sum10, avx_Nor3, avx_de031);
		float64x2_t avx_sum2 = vmulq_f64(avx_Nor1, avx_de12);
		avx_sum2 = vfmaq_f64(avx_sum2, avx_Nor2, avx_de22);
		avx_sum2 = vfmaq_f64(avx_sum2, avx_Nor3, avx_de32);
		float64x2_t avx_sum20 = vmulq_f64(avx_Nor1, avx_de012);
		avx_sum20 = vfmaq_f64(avx_sum20, avx_Nor2, avx_de022);
		avx_sum20 = vfmaq_f64(avx_sum20, avx_Nor3, avx_de032);
		float64x2_t avx_sum3 = vmulq_f64(avx_Nor1, avx_de13);
		avx_sum3 = vfmaq_f64(avx_sum3, avx_Nor2, avx_de23);
		avx_sum3 = vfmaq_f64(avx_sum3, avx_Nor3, avx_de33);
		float64x2_t avx_sum30 = vmulq_f64(avx_Nor1, avx_de013);
		avx_sum30 = vfmaq_f64(avx_sum30, avx_Nor2, avx_de023);
		avx_sum30 = vfmaq_f64(avx_sum30, avx_Nor3, avx_de033);

		avx_dyda1 = vfmaq_f64(avx_dyda1, vaddq_f64(vmulq_f64(avx_sum1, avx_dsmu), vmulq_f64(avx_sum10, avx_dsmu0)), avx_Area);
		avx_dyda2 = vfmaq_f64(avx_dyda2, vaddq_f64(vmulq_f64(avx_sum2, avx_dsmu), vmulq_f64(avx_sum20, avx_dsmu0)), avx_Area);
		avx_dyda3 = vfmaq_f64(avx_dyda3, vaddq_f64(vmulq_f64(avx_sum3, avx_dsmu), vmulq_f64(avx_sum30, avx_dsmu0)), avx_Area);

		// derivatives w.r.t. cl, cls
		avx_d = vfmaq_f64(avx_d, avx_q, avx_Area);
		avx_d1 = vaddq_f64(avx_d1, vdivq_f64(vmulq_f64(vmulq_f64(avx_Area, avx_lmu), avx_lmu0), avx_dnom));

		// Branchless compaction of the visible facets: always write the slot, advance only when the lane is visible.
		for (int l = 0; l < 2; l++)
		{
			Dg_row[incl_count] = reinterpret_cast<float64x2_t*>(gl.Dg[i + l]);
			dbr[incl_count] = g[l];
			incl_count += (icmp >> l) & 1;
		}
	}

	// zero-weight padding entry for the pairwise order, valid (unused) rows for the prefetch look-ahead
	dbr[incl_count] = 0.0;
	for (j = 0; j <= DG_PREFETCH_ROWS; j++)
		Dg_row[incl_count + j] = reinterpret_cast<float64x2_t*>(gl.Dg[0]);

	res_br = vpaddq_f64(res_br, res_br);
	vst1q_lane_f64(&gl.ymod, res_br, 0);

	/* Derivatives of brightness w.r.t. g-coefficients, in balanced chunks of at most 26 vectors (accumulators in registers) */
	const int nvec = (ncoef0 - 3 + 1) / 2;
	const int nchunks = (nvec + 25) / 26;
	const int wchunk = nchunks > 0 ? (nvec + nchunks - 1) / nchunks : 1;
	for (int v = 0; v < nvec; v += wchunk)
		dg_chunk(nvec - v < wchunk ? nvec - v : wchunk, Dg_row, dbr, incl_count, v, gl.dyda, avx_Scale);

	/* Derivatives of brightness w.r.t. rotation parameters */
	avx_dyda1 = vpaddq_f64(avx_dyda1, avx_dyda2);
	avx_dyda1 = vmulq_f64(avx_dyda1, avx_Scale);
	vst1q_f64(&gl.dyda[ncoef0 - 3 + 1 - 1], avx_dyda1);	//unaligned memory because of odd index

	avx_dyda3 = vpaddq_f64(avx_dyda3, avx_dyda3);
	avx_dyda3 = vmulq_f64(avx_dyda3, avx_Scale);
	vst1q_f64(&gl.dyda[ncoef0 - 3 + 3 - 1], avx_dyda3);	//unaligned memory because of odd index

	/* Derivatives of br. w.r.t. cl, cls */
	avx_d = vpaddq_f64(avx_d, avx_d1);
	avx_d = vmulq_f64(avx_d, avx_Scale);
	avx_d = vmulq_f64(avx_d, avx_cl1);
	vst1q_f64(&gl.dyda[ncoef - 1 - 1], avx_d);	//unaligned memory because of odd index

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];

	/* Scaled brightness */
	gl.ymod *= Scale;
}
