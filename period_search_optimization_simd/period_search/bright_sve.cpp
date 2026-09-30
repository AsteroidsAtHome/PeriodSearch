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
#include "CalcStrategySve.hpp"
#if defined(__aarch64__) || defined(_M_ARM64)
  #include <arm_neon.h>
  #define DG_HAVE_NEON 1
#endif

#if defined(_MSC_VER) && !defined(__clang__)
  #include <intrin.h>
  #if defined(_M_ARM64)
    #define DG_PREFETCH(p) __prefetch(p)
  #else
    #define DG_PREFETCH(p) _mm_prefetch(p, _MM_HINT_T0)
  #endif
#else
  #define DG_PREFETCH(p) __builtin_prefetch(p, 0, 3)
#endif

#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
  #define SVE_TARGET __attribute__((__target__("+sve")))
  #define SVE_TARGET_INLINE __attribute__((__target__("+sve"), always_inline))
#elif defined(__GNUC__)
  #define SVE_TARGET
  #define SVE_TARGET_INLINE __attribute__((always_inline))
#else
  #define SVE_TARGET
  #define SVE_TARGET_INLINE
#endif

// doubles per row of gl.Dg
constexpr int DG_STRIDE = MAX_N_PAR + 8;

/**
 * @brief dyda[cnt * v .. cnt * (v + W)) = Scale * sum_j dbr[j] * Dg[Dg_idx[j]][cnt * v .. cnt * (v + W))
 *
 * W accumulators stay in registers, so each visible Dg row is streamed only once per chunk (with 32 vector
 * registers up to 26 accumulators fit: 49 Dg columns are a single pass already with 128-bit vectors).
 * Only the last vector of the chunk can be partial, it is governed by pl (all-true when the chunk ends inside the row).
 * Relies on the padding entry at dbr[incl_count] and on valid indices up to Dg_idx[incl_count + DG_PREFETCH_ROWS].
 */
template <int W>
SVE_TARGET_INLINE
static inline void dg_accumulate(const double* Dg, const int64_t* Dg_idx, const double* dbr, const int incl_count, const int v, double* dyda,
								 const svbool_t pt, const svbool_t pl, const svfloat64_t avx_Scale)
{
	const int cnt = static_cast<int>(svcntd());
	const int off = cnt * v;

	// Named accumulators (not an array) so that the compiler keeps all of them in registers; unused ones are optimized away.
	svfloat64_t a0 = svdup_n_f64(0.0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0;
	svfloat64_t a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0, a12 = a0, a13 = a0;
	svfloat64_t a14 = a0, a15 = a0, a16 = a0, a17 = a0, a18 = a0, a19 = a0;
	svfloat64_t a20 = a0, a21 = a0, a22 = a0, a23 = a0, a24 = a0, a25 = a0;

// vector k is loaded as [p_(k / 8), #(k % 8), MUL VL]: the immediate offset only reaches 8 vectors, so there are 4 bases
#define DG_MLA(k) if (W > k) a##k = svmla_n_f64_x(pt, a##k, svld1_vnum_f64(k == W - 1 ? pl : pt, (k) < 8 ? p0 : (k) < 16 ? p1 : (k) < 24 ? p2 : p3, (k) % 8), pdbr)
	const int n = (incl_count + 1) & ~1;
	for (int j = 0; j < n; j++)
	{
		// Dg rows exceed L1 in total and are visited in a data dependent order, so fetch a few rows ahead
		const char* pf = reinterpret_cast<const char*>(Dg + Dg_idx[j + DG_PREFETCH_ROWS] * DG_STRIDE + off);
		for (int b = 0; b < W * cnt * 8; b += 64)
			DG_PREFETCH(pf + b);

		const double* p0 = Dg + Dg_idx[j] * DG_STRIDE + off;
		const double* p1 = p0 + 8 * cnt;
		const double* p2 = p0 + 16 * cnt;
		const double* p3 = p0 + 24 * cnt;
		const double pdbr = dbr[j];
		DG_MLA(0); DG_MLA(1); DG_MLA(2); DG_MLA(3); DG_MLA(4); DG_MLA(5); DG_MLA(6);
		DG_MLA(7); DG_MLA(8); DG_MLA(9); DG_MLA(10); DG_MLA(11); DG_MLA(12); DG_MLA(13);
		DG_MLA(14); DG_MLA(15); DG_MLA(16); DG_MLA(17); DG_MLA(18); DG_MLA(19);
		DG_MLA(20); DG_MLA(21); DG_MLA(22); DG_MLA(23); DG_MLA(24); DG_MLA(25);
	}
#undef DG_MLA

#define DG_STORE(k) if (W > k) svst1_f64(k == W - 1 ? pl : pt, &dyda[off + cnt * k], svmul_f64_x(pt, a##k, avx_Scale))
	DG_STORE(0); DG_STORE(1); DG_STORE(2); DG_STORE(3); DG_STORE(4); DG_STORE(5); DG_STORE(6);
	DG_STORE(7); DG_STORE(8); DG_STORE(9); DG_STORE(10); DG_STORE(11); DG_STORE(12); DG_STORE(13);
	DG_STORE(14); DG_STORE(15); DG_STORE(16); DG_STORE(17); DG_STORE(18); DG_STORE(19);
	DG_STORE(20); DG_STORE(21); DG_STORE(22); DG_STORE(23); DG_STORE(24); DG_STORE(25);
#undef DG_STORE
}

SVE_TARGET
static void dg_chunk(const int w, const double* Dg, const int64_t* Dg_idx, const double* dbr, const int incl_count, const int v, double* dyda,
					 const svbool_t pt, const svbool_t pl, const svfloat64_t avx_Scale)
{
	switch (w)
	{
		case 26: dg_accumulate<26>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 25: dg_accumulate<25>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 24: dg_accumulate<24>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 23: dg_accumulate<23>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 22: dg_accumulate<22>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 21: dg_accumulate<21>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 20: dg_accumulate<20>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 19: dg_accumulate<19>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 18: dg_accumulate<18>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 17: dg_accumulate<17>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 16: dg_accumulate<16>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 15: dg_accumulate<15>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 14: dg_accumulate<14>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 13: dg_accumulate<13>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 12: dg_accumulate<12>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 11: dg_accumulate<11>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 10: dg_accumulate<10>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 9: dg_accumulate<9>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 8: dg_accumulate<8>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 7: dg_accumulate<7>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 6: dg_accumulate<6>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 5: dg_accumulate<5>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 4: dg_accumulate<4>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 3: dg_accumulate<3>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 2: dg_accumulate<2>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 1: dg_accumulate<1>(Dg, Dg_idx, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		default: break;
	}
}

#if defined(DG_HAVE_NEON)
/**
 * @brief NEON variant of dg_accumulate for 128-bit SVE.
 *
 * With 128-bit vectors SVE brings no extra width, and the Dg product is load bound (one load per FMA): SVE loads one
 * vector per LD1D, while consecutive NEON loads pair into LDP Q (two vectors per instruction). The registers are
 * shared, so both can be mixed freely. The summation order is the same as in dg_accumulate.
 */
template <int W>
#if defined(__GNUC__)
__attribute__((always_inline))
#endif
static inline void dg_accumulate_neon(const double* Dg, const int64_t* Dg_idx, const double* dbr, const int incl_count, const int v, double* dyda,
									  const float64x2_t avx_Scale)
{
	const int off = 2 * v;

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
		const char* pf = reinterpret_cast<const char*>(Dg + Dg_idx[j + DG_PREFETCH_ROWS] * DG_STRIDE + off);
		for (int b = 0; b < W * 16; b += 64)
			DG_PREFETCH(pf + b);

		const double* p = Dg + Dg_idx[j] * DG_STRIDE + off;
		const float64x2_t pdbr = vld1q_dup_f64(&dbr[j]);
		DG_FMA(0); DG_FMA(1); DG_FMA(2); DG_FMA(3); DG_FMA(4); DG_FMA(5); DG_FMA(6);
		DG_FMA(7); DG_FMA(8); DG_FMA(9); DG_FMA(10); DG_FMA(11); DG_FMA(12); DG_FMA(13);
		DG_FMA(14); DG_FMA(15); DG_FMA(16); DG_FMA(17); DG_FMA(18); DG_FMA(19);
		DG_FMA(20); DG_FMA(21); DG_FMA(22); DG_FMA(23); DG_FMA(24); DG_FMA(25);
	}
#undef DG_FMA

	// whole vectors: the (odd) last column spills into dyda[ncoef0 - 3], which is written afterwards
#define DG_STORE(k) if (W > k) vst1q_f64(&dyda[off + 2 * k], vmulq_f64(a##k, avx_Scale))
	DG_STORE(0); DG_STORE(1); DG_STORE(2); DG_STORE(3); DG_STORE(4); DG_STORE(5); DG_STORE(6);
	DG_STORE(7); DG_STORE(8); DG_STORE(9); DG_STORE(10); DG_STORE(11); DG_STORE(12); DG_STORE(13);
	DG_STORE(14); DG_STORE(15); DG_STORE(16); DG_STORE(17); DG_STORE(18); DG_STORE(19);
	DG_STORE(20); DG_STORE(21); DG_STORE(22); DG_STORE(23); DG_STORE(24); DG_STORE(25);
#undef DG_STORE
}

static void dg_chunk_neon(const int w, const double* Dg, const int64_t* Dg_idx, const double* dbr, const int incl_count, const int v, double* dyda,
						  const float64x2_t avx_Scale)
{
	switch (w)
	{
		case 26: dg_accumulate_neon<26>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 25: dg_accumulate_neon<25>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 24: dg_accumulate_neon<24>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 23: dg_accumulate_neon<23>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 22: dg_accumulate_neon<22>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 21: dg_accumulate_neon<21>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 20: dg_accumulate_neon<20>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 19: dg_accumulate_neon<19>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 18: dg_accumulate_neon<18>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 17: dg_accumulate_neon<17>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 16: dg_accumulate_neon<16>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 15: dg_accumulate_neon<15>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 14: dg_accumulate_neon<14>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 13: dg_accumulate_neon<13>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 12: dg_accumulate_neon<12>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 11: dg_accumulate_neon<11>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 10: dg_accumulate_neon<10>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 9: dg_accumulate_neon<9>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 8: dg_accumulate_neon<8>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 7: dg_accumulate_neon<7>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 6: dg_accumulate_neon<6>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 5: dg_accumulate_neon<5>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 4: dg_accumulate_neon<4>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 3: dg_accumulate_neon<3>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 2: dg_accumulate_neon<2>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		case 1: dg_accumulate_neon<1>(Dg, Dg_idx, dbr, incl_count, v, dyda, avx_Scale); break;
		default: break;
	}
}
#endif

/**
 * @brief Computes integrated brightness of all visible and illuminated areas and its derivatives.
 *
 * This function calculates the integrated brightness of all visible and illuminated areas based on
 * the provided time t, coefficient vector cg, and global data. It also computes the derivatives of
 * the brightness with respect to the coefficients.
 *
 * Unlike the fixed width ports, the facet loop is designed for SVE:
 *  - the derivatives w.r.t. the rotation parameters are contracted with de / de0 only once at the end:
 *    sum_f Area * (dsmu * (Nor . de[:, j]) + dsmu0 * (Nor . de0[:, j]))
 *      = sum_c de[c][j] * sum_f Area * dsmu * Nor_c + de0[c][j] * sum_f Area * dsmu0 * Nor_c,
 *    so per facet only six accumulators are updated and no de / de0 constants occupy vector registers,
 *  - the visible facets are compacted with COMPACT / CNTP instead of a scalar loop over the lanes,
 *  - the brightness needs no accumulator of its own: sum_f Area * mu * mu0 * (cl + cls / (mu + mu0))
 *    = cl * d + cls * d1 with the sums d, d1 that the derivatives w.r.t. cl, cls need anyway,
 *  - rejected lanes (and the tail) are simply inactive: zero area from the predicated loads and 1 / dnom = 1.
 *
 * @param t The time at which the brightness is evaluated.
 * @param cg A reference to a vector of doubles containing the coefficients for the brightness calculation.
 * @param ncoef An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables ymod and dyda.
 *
 * @date 8.11.2006
 * @author Josef Durec
 */
SVE_TARGET
void CalcStrategySve::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j, k;
	int nincl = 0;	// local copy of incl_count (a member would be written back to memory in every iteration)
	double *ee = gl.xx1;
	double *ee0 = gl.xx2;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);		/* Lambert */
	cls = cg[ncoef];				/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php);		/* computes also Scale */

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

	/* Integrated brightness (phase coefficients used later) */
	const svbool_t pt = svptrue_b64();
	const int cnt = static_cast<int>(svcntd());

	const svfloat64_t avx_e1 = svdup_n_f64(e[1]);
	const svfloat64_t avx_e2 = svdup_n_f64(e[2]);
	const svfloat64_t avx_e3 = svdup_n_f64(e[3]);
	const svfloat64_t avx_e01 = svdup_n_f64(e0[1]);
	const svfloat64_t avx_e02 = svdup_n_f64(e0[2]);
	const svfloat64_t avx_e03 = svdup_n_f64(e0[3]);
	const svfloat64_t avx_tiny = svdup_n_f64(TINY);
	const svfloat64_t avx_cl = svdup_n_f64(cl);
	const svfloat64_t avx_cls = svdup_n_f64(cls);
	const svfloat64_t avx_11 = svdup_n_f64(1.0);

	// d = sum_f Area * mu * mu0, d1 = sum_f Area * mu * mu0 / (mu + mu0)
	svfloat64_t avx_d = svdup_n_f64(0.0), avx_d1 = avx_d;
	// sum_f Area * dsmu * Nor_c and sum_f Area * dsmu0 * Nor_c, c = 1..3
	svfloat64_t avx_t1 = avx_d, avx_t2 = avx_d, avx_t3 = avx_d;
	svfloat64_t avx_u1 = avx_d, avx_u2 = avx_d, avx_u3 = avx_d;

	for (i = 0; i < Numfac; i += cnt)
	{
		const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(i), static_cast<int64_t>(Numfac));
		const svfloat64_t avx_Nor1 = svld1_f64(pg, &gl.Nor[0][i]);
		const svfloat64_t avx_Nor2 = svld1_f64(pg, &gl.Nor[1][i]);
		const svfloat64_t avx_Nor3 = svld1_f64(pg, &gl.Nor[2][i]);

		svfloat64_t avx_lmu = svmul_f64_x(pt, avx_e1, avx_Nor1);
		avx_lmu = svmla_f64_x(pt, avx_lmu, avx_e2, avx_Nor2);
		avx_lmu = svmla_f64_x(pt, avx_lmu, avx_e3, avx_Nor3);
		svfloat64_t avx_lmu0 = svmul_f64_x(pt, avx_e01, avx_Nor1);
		avx_lmu0 = svmla_f64_x(pt, avx_lmu0, avx_e02, avx_Nor2);
		avx_lmu0 = svmla_f64_x(pt, avx_lmu0, avx_e03, avx_Nor3);

		// visible and illuminated; the tail lanes have zero normals and fail the test
		const svbool_t cmp = svand_z(pt, svcmpgt_f64(pt, avx_lmu, avx_tiny), svcmpgt_f64(pt, avx_lmu0, avx_tiny));
		if (!svptest_any(pt, cmp))
			continue;

		// rejected lanes: zero area / darea (predicated loads) and 1 / dnom = 1, so every term stays finite and vanishes
		const svfloat64_t avx_Area = svld1_f64(cmp, &gl.Area[i]);
		const svfloat64_t avx_inv = svdiv_f64_m(cmp, avx_11, svadd_f64_x(pt, avx_lmu, avx_lmu0));
		const svfloat64_t avx_q = svmul_f64_x(pt, avx_lmu, avx_lmu0);
		const svfloat64_t avx_s = svmul_f64_x(pt, avx_q, svmla_f64_x(pt, avx_cl, avx_cls, avx_inv));
		const svfloat64_t avx_dbr = svmul_f64_x(pt, svld1_f64(cmp, &gl.Darea[i]), avx_s);

		// dsmu = cls * (mu0 / (mu + mu0))^2 + cl * mu0,  dsmu0 = cls * (mu / (mu + mu0))^2 + cl * mu
		svfloat64_t avx_pw = svmul_f64_x(pt, avx_lmu0, avx_inv);
		const svfloat64_t avx_dsmu = svmla_f64_x(pt, svmul_f64_x(pt, avx_cl, avx_lmu0), avx_cls, svmul_f64_x(pt, avx_pw, avx_pw));
		avx_pw = svmul_f64_x(pt, avx_lmu, avx_inv);
		const svfloat64_t avx_dsmu0 = svmla_f64_x(pt, svmul_f64_x(pt, avx_cl, avx_lmu), avx_cls, svmul_f64_x(pt, avx_pw, avx_pw));

		// rotation derivatives (contracted with de / de0 after the loop)
		const svfloat64_t avx_ta = svmul_f64_x(pt, avx_Area, avx_dsmu);
		const svfloat64_t avx_ua = svmul_f64_x(pt, avx_Area, avx_dsmu0);
		avx_t1 = svmla_f64_x(pt, avx_t1, avx_ta, avx_Nor1);
		avx_t2 = svmla_f64_x(pt, avx_t2, avx_ta, avx_Nor2);
		avx_t3 = svmla_f64_x(pt, avx_t3, avx_ta, avx_Nor3);
		avx_u1 = svmla_f64_x(pt, avx_u1, avx_ua, avx_Nor1);
		avx_u2 = svmla_f64_x(pt, avx_u2, avx_ua, avx_Nor2);
		avx_u3 = svmla_f64_x(pt, avx_u3, avx_ua, avx_Nor3);

		// derivatives w.r.t. cl, cls
		const svfloat64_t avx_aq = svmul_f64_x(pt, avx_Area, avx_q);
		avx_d = svadd_f64_x(pt, avx_d, avx_aq);
		avx_d1 = svmla_f64_x(pt, avx_d1, avx_aq, avx_inv);

		// compaction of the visible facets: indices and weights packed to the front and stored contiguously
		const int nvis = static_cast<int>(svcntp_b64(pt, cmp));
		const svbool_t pc = svwhilelt_b64(static_cast<int64_t>(0), static_cast<int64_t>(nvis));
		svst1_s64(pc, &Dg_idx[nincl], svcompact_s64(cmp, svindex_s64(i, 1)));
		svst1_f64(pc, &dbr[nincl], svcompact_f64(cmp, avx_dbr));
		nincl += nvis;
	}
	incl_count = nincl;

	// zero-weight padding entry for the pairwise order, valid (unused) rows for the prefetch look-ahead
	dbr[incl_count] = 0.0;
	for (j = 0; j <= DG_PREFETCH_ROWS; j++)
		Dg_idx[incl_count + j] = 0;

	const double d = svaddv_f64(pt, avx_d);
	const double d1 = svaddv_f64(pt, avx_d1);
	gl.ymod = cl * d + cls * d1;

	/* Derivatives of brightness w.r.t. g-coefficients, in balanced chunks of at most 26 vectors (accumulators in registers) */
	const int ncoef03 = ncoef0 - 3;
	const int nvec = (ncoef03 + cnt - 1) / cnt;
	const int nchunks = (nvec + 26 - 1) / 26;
	const int wchunk = nchunks > 0 ? (nvec + nchunks - 1) / nchunks : 1;
#if defined(DG_HAVE_NEON)
	if (cnt == 2)
	{
		const float64x2_t avx_Scale = vdupq_n_f64(Scale);
		for (int v = 0; v < nvec; v += wchunk)
			dg_chunk_neon(nvec - v < wchunk ? nvec - v : wchunk, &gl.Dg[0][0], Dg_idx, dbr, incl_count, v, gl.dyda, avx_Scale);
	}
	else
#endif
	{
		const svfloat64_t avx_Scale = svdup_n_f64(Scale);
		for (int v = 0; v < nvec; v += wchunk)
		{
			const int w = nvec - v < wchunk ? nvec - v : wchunk;
			const svbool_t pl = svwhilelt_b64(static_cast<int64_t>(cnt) * (v + w - 1), static_cast<int64_t>(ncoef03));
			dg_chunk(w, &gl.Dg[0][0], Dg_idx, dbr, incl_count, v, gl.dyda, pt, pl, avx_Scale);
		}
	}

	/* Derivatives of brightness w.r.t. rotation parameters */
	const double ts[4] = { 0.0, svaddv_f64(pt, avx_t1), svaddv_f64(pt, avx_t2), svaddv_f64(pt, avx_t3) };
	const double us[4] = { 0.0, svaddv_f64(pt, avx_u1), svaddv_f64(pt, avx_u2), svaddv_f64(pt, avx_u3) };
	for (j = 1; j <= 3; j++)
	{
		double sum = 0.0;
		for (k = 1; k <= 3; k++)
			sum += de[k][j] * ts[k] + de0[k][j] * us[k];
		gl.dyda[ncoef0 - 3 + j - 1] = sum * Scale;
	}

	/* Derivatives of br. w.r.t. cl, cls */
	gl.dyda[ncoef - 1 - 1] = d * Scale * cl;
	gl.dyda[ncoef - 1] = d1 * Scale;

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];

	/* Scaled brightness */
	gl.ymod *= Scale;
}
