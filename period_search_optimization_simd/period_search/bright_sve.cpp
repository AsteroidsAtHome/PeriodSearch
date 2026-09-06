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

// Everything below runs on the all-true predicate. The tail lanes of the last iteration
// are loaded as zeros, so lmu / lmu0 are zero there, the visibility test rejects them and
// their contribution to every accumulator is zero as well - which is what lets the final
// svaddv_f64 reduce over the whole vector.
#define INNER_CALC \
    res_br = svadd_f64_x(pt, res_br, avx_pbr); \
    svfloat64_t avx_sum1, avx_sum10, avx_sum2, avx_sum20, avx_sum3, avx_sum30; \
    \
    avx_sum1 = svmul_f64_x(pt, avx_Nor1, avx_de11); \
    avx_sum1 = svmla_f64_x(pt, avx_sum1, avx_Nor2, avx_de21); \
    avx_sum1 = svmla_f64_x(pt, avx_sum1, avx_Nor3, avx_de31); \
    \
    avx_sum10 = svmul_f64_x(pt, avx_Nor1, avx_de011); \
    avx_sum10 = svmla_f64_x(pt, avx_sum10, avx_Nor2, avx_de021); \
    avx_sum10 = svmla_f64_x(pt, avx_sum10, avx_Nor3, avx_de031); \
    \
    avx_sum2 = svmul_f64_x(pt, avx_Nor1, avx_de12); \
    avx_sum2 = svmla_f64_x(pt, avx_sum2, avx_Nor2, avx_de22); \
    avx_sum2 = svmla_f64_x(pt, avx_sum2, avx_Nor3, avx_de32); \
    \
    avx_sum20 = svmul_f64_x(pt, avx_Nor1, avx_de012); \
    avx_sum20 = svmla_f64_x(pt, avx_sum20, avx_Nor2, avx_de022); \
    avx_sum20 = svmla_f64_x(pt, avx_sum20, avx_Nor3, avx_de032); \
    \
    avx_sum3 = svmul_f64_x(pt, avx_Nor1, avx_de13); \
    avx_sum3 = svmla_f64_x(pt, avx_sum3, avx_Nor2, avx_de23); \
    avx_sum3 = svmla_f64_x(pt, avx_sum3, avx_Nor3, avx_de33); \
    \
    avx_sum30 = svmul_f64_x(pt, avx_Nor1, avx_de013); \
    avx_sum30 = svmla_f64_x(pt, avx_sum30, avx_Nor2, avx_de023); \
    avx_sum30 = svmla_f64_x(pt, avx_sum30, avx_Nor3, avx_de033); \
    \
    avx_sum1 = svmul_f64_x(pt, avx_sum1, avx_dsmu); \
    avx_sum2 = svmul_f64_x(pt, avx_sum2, avx_dsmu); \
    avx_sum3 = svmul_f64_x(pt, avx_sum3, avx_dsmu); \
    avx_sum10 = svmul_f64_x(pt, avx_sum10, avx_dsmu0); \
    avx_sum20 = svmul_f64_x(pt, avx_sum20, avx_dsmu0); \
    avx_sum30 = svmul_f64_x(pt, avx_sum30, avx_dsmu0); \
    \
    avx_dyda1 = svmla_f64_x(pt, avx_dyda1, avx_Area, svadd_f64_x(pt, avx_sum1, avx_sum10)); \
    avx_dyda2 = svmla_f64_x(pt, avx_dyda2, avx_Area, svadd_f64_x(pt, avx_sum2, avx_sum20)); \
    avx_dyda3 = svmla_f64_x(pt, avx_dyda3, avx_Area, svadd_f64_x(pt, avx_sum3, avx_sum30)); \
    \
    avx_d = svmla_f64_x(pt, avx_d, avx_Area, svmul_f64_x(pt, avx_lmu, avx_lmu0)); \
    avx_d1 = svmla_f64_x(pt, avx_d1, svmul_f64_x(pt, svmul_f64_x(pt, avx_Area, avx_lmu), avx_lmu0), avx_inv);
// end of inner_calc

// 1/dnom is forced to 1 wherever the facet is rejected, so a zero dnom can never turn into
// an inf that a later multiplication by zero would make a NaN.
#define INNER_CALC_DSMU \
    avx_Area = svld1_f64(pg, &gl.Area[i]); \
    avx_dnom = svadd_f64_x(pt, avx_lmu, avx_lmu0); \
    avx_inv = svsel_f64(cmp, svdiv_f64_x(pt, avx_11, avx_dnom), avx_11); \
    avx_s = svmul_f64_x(pt, svmul_f64_x(pt, avx_lmu, avx_lmu0), svmla_f64_x(pt, avx_cl, avx_cls, avx_inv)); \
    avx_pdbr = svmul_f64_x(pt, svld1_f64(pg, &gl.Darea[i]), avx_s); \
    avx_pbr = svmul_f64_x(pt, avx_Area, avx_s); \
    avx_powdnom = svmul_f64_x(pt, avx_lmu0, avx_inv); \
    avx_powdnom = svmul_f64_x(pt, avx_powdnom, avx_powdnom); \
    avx_dsmu = svmla_f64_x(pt, svmul_f64_x(pt, avx_cls, avx_powdnom), avx_cl, avx_lmu0); \
    avx_powdnom = svmul_f64_x(pt, avx_lmu, avx_inv); \
    avx_powdnom = svmul_f64_x(pt, avx_powdnom, avx_powdnom); \
    avx_dsmu0 = svmla_f64_x(pt, svmul_f64_x(pt, avx_cls, avx_powdnom), avx_cl, avx_lmu);
// end of inner_calc_dsmu

/**
 * @brief Computes integrated brightness of all visible and illuminated areas and its derivatives.
 *
 * This function calculates the integrated brightness of all visible and illuminated areas based on
 * the provided time t, coefficient vector cg, and global data. It also computes the derivatives of
 * the brightness with respect to the coefficients.
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
#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j, k;
	incl_count = 0;
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

	svfloat64_t avx_e1 = svdup_n_f64(e[1]);
	svfloat64_t avx_e2 = svdup_n_f64(e[2]);
	svfloat64_t avx_e3 = svdup_n_f64(e[3]);
	svfloat64_t avx_e01 = svdup_n_f64(e0[1]);
	svfloat64_t avx_e02 = svdup_n_f64(e0[2]);
	svfloat64_t avx_e03 = svdup_n_f64(e0[3]);
	svfloat64_t avx_de11 = svdup_n_f64(de[1][1]);
	svfloat64_t avx_de12 = svdup_n_f64(de[1][2]);
	svfloat64_t avx_de13 = svdup_n_f64(de[1][3]);
	svfloat64_t avx_de21 = svdup_n_f64(de[2][1]);
	svfloat64_t avx_de22 = svdup_n_f64(de[2][2]);
	svfloat64_t avx_de23 = svdup_n_f64(de[2][3]);
	svfloat64_t avx_de31 = svdup_n_f64(de[3][1]);
	svfloat64_t avx_de32 = svdup_n_f64(de[3][2]);
	svfloat64_t avx_de33 = svdup_n_f64(de[3][3]);
	svfloat64_t avx_de011 = svdup_n_f64(de0[1][1]);
	svfloat64_t avx_de012 = svdup_n_f64(de0[1][2]);
	svfloat64_t avx_de013 = svdup_n_f64(de0[1][3]);
	svfloat64_t avx_de021 = svdup_n_f64(de0[2][1]);
	svfloat64_t avx_de022 = svdup_n_f64(de0[2][2]);
	svfloat64_t avx_de023 = svdup_n_f64(de0[2][3]);
	svfloat64_t avx_de031 = svdup_n_f64(de0[3][1]);
	svfloat64_t avx_de032 = svdup_n_f64(de0[3][2]);
	svfloat64_t avx_de033 = svdup_n_f64(de0[3][3]);
	svfloat64_t avx_Scale = svdup_n_f64(Scale);

	svfloat64_t avx_tiny = svdup_n_f64(TINY);
	svfloat64_t avx_cl = svdup_n_f64(cl);
	svfloat64_t avx_cls = svdup_n_f64(cls);
	svfloat64_t avx_11 = svdup_n_f64(1.0);
	svfloat64_t avx_zero = svdup_n_f64(0.0);
	svfloat64_t res_br = svdup_n_f64(0.0);
	svfloat64_t avx_dyda1 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda2 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda3 = svdup_n_f64(0.0);
	svfloat64_t avx_d = svdup_n_f64(0.0);
	svfloat64_t avx_d1 = svdup_n_f64(0.0);

	double s_lmu[SVE_MAX_LANES];
	double s_lmu0[SVE_MAX_LANES];
	double s_pdbr[SVE_MAX_LANES];

	for (i = 0; i < Numfac; i += cnt)
	{
		const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(i), static_cast<int64_t>(Numfac));
		const int active = (Numfac - i) < cnt ? (Numfac - i) : cnt;

		svfloat64_t avx_lmu, avx_lmu0;
		svfloat64_t avx_Nor1 = svld1_f64(pg, &gl.Nor[0][i]);
		svfloat64_t avx_Nor2 = svld1_f64(pg, &gl.Nor[1][i]);
		svfloat64_t avx_Nor3 = svld1_f64(pg, &gl.Nor[2][i]);
		svfloat64_t avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr, avx_inv;
		svfloat64_t avx_Area;

		avx_lmu = svmul_f64_x(pt, avx_e1, avx_Nor1);
		avx_lmu = svmla_f64_x(pt, avx_lmu, avx_e2, avx_Nor2);
		avx_lmu = svmla_f64_x(pt, avx_lmu, avx_e3, avx_Nor3);

		avx_lmu0 = svmul_f64_x(pt, avx_e01, avx_Nor1);
		avx_lmu0 = svmla_f64_x(pt, avx_lmu0, avx_e02, avx_Nor2);
		avx_lmu0 = svmla_f64_x(pt, avx_lmu0, avx_e03, avx_Nor3);

		const svbool_t cmp = svand_z(pt, svcmpgt_f64(pt, avx_lmu, avx_tiny),
										 svcmpgt_f64(pt, avx_lmu0, avx_tiny));

		if (svptest_any(pt, cmp))
		{
			INNER_CALC_DSMU

			/* The per-facet bookkeeping stays scalar: the vector length is not a compile
			   time constant, so the predicate cannot be folded into a lane bitmask. */
			svst1_f64(pg, s_lmu, avx_lmu);
			svst1_f64(pg, s_lmu0, avx_lmu0);
			svst1_f64(pg, s_pdbr, avx_pdbr);

			avx_pbr = svsel_f64(cmp, avx_pbr, avx_zero);
			avx_dsmu = svsel_f64(cmp, avx_dsmu, avx_zero);
			avx_dsmu0 = svsel_f64(cmp, avx_dsmu0, avx_zero);
			avx_lmu = svsel_f64(cmp, avx_lmu, avx_zero);
			avx_lmu0 = svsel_f64(cmp, avx_lmu0, avx_zero);

			for (j = 0; j < active; j++)
			{
				if (s_lmu[j] > TINY && s_lmu0[j] > TINY)
				{
					Dg_row[incl_count] = gl.Dg[i + j];
					dbr[incl_count++] = s_pdbr[j];
				}
			}

			INNER_CALC
		}
	}

	/* one padding entry so the loop below may always read a pair */
	dbr[incl_count] = 0.0;
	Dg_row[incl_count] = Dg_row[0];

	gl.ymod = svaddv_f64(pt, res_br);

	/* Derivatives of brightness w.r.t. g-coefficients */
	const int ncoef03 = ncoef0 - 3;
	for (i = 0; i < ncoef03; i += cnt)
	{
		const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(i), static_cast<int64_t>(ncoef03));
		svfloat64_t tmp1 = svdup_n_f64(0.0);

		for (j = 0; j < incl_count; j += 2)
		{
			tmp1 = svmla_n_f64_x(pg, tmp1, svld1_f64(pg, Dg_row[j] + i), dbr[j]);
			tmp1 = svmla_n_f64_x(pg, tmp1, svld1_f64(pg, Dg_row[j + 1] + i), dbr[j + 1]);
		}

		svst1_f64(pg, &gl.dyda[i], svmul_f64_x(pg, tmp1, avx_Scale));
	}

	/* Derivatives of brightness w.r.t. rotation parameters */
	gl.dyda[ncoef0 - 3 + 1 - 1] = svaddv_f64(pt, avx_dyda1) * Scale;
	gl.dyda[ncoef0 - 3 + 2 - 1] = svaddv_f64(pt, avx_dyda2) * Scale;
	gl.dyda[ncoef0 - 3 + 3 - 1] = svaddv_f64(pt, avx_dyda3) * Scale;

	/* Derivatives of br. w.r.t. cl, cls */
	gl.dyda[ncoef - 1 - 1] = svaddv_f64(pt, avx_d) * Scale * cl;
	gl.dyda[ncoef - 1] = svaddv_f64(pt, avx_d1) * Scale;

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];

	/* Scaled brightness */
	gl.ymod *= Scale;
}
