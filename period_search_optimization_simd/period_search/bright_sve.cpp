/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "CalcStrategySve.hpp"

#define INNER_CALC \
    res_br = svadd_f64_x(pg, res_br, avx_pbr); \
    svfloat64_t avx_sum1, avx_sum10, avx_sum2, avx_sum20, avx_sum3, avx_sum30; \
    \
    avx_sum1 = svmul_f64_x(pg, avx_Nor1, avx_de11); \
    avx_sum1 = svmla_f64_x(pg, avx_sum1, avx_Nor2, avx_de21); \
    avx_sum1 = svmla_f64_x(pg, avx_sum1, avx_Nor3, avx_de31); \
    \
    avx_sum10 = svmul_f64_x(pg, avx_Nor1, avx_de011); \
    avx_sum10 = svmla_f64_x(pg, avx_sum10, avx_Nor2, avx_de021); \
    avx_sum10 = svmla_f64_x(pg, avx_sum10, avx_Nor3, avx_de031); \
    \
    avx_sum2 = svmul_f64_x(pg, avx_Nor1, avx_de12); \
    avx_sum2 = svmla_f64_x(pg, avx_sum2, avx_Nor2, avx_de22); \
    avx_sum2 = svmla_f64_x(pg, avx_sum2, avx_Nor3, avx_de32); \
    \
    avx_sum20 = svmul_f64_x(pg, avx_Nor1, avx_de012); \
    avx_sum20 = svmla_f64_x(pg, avx_sum20, avx_Nor2, avx_de022); \
    avx_sum20 = svmla_f64_x(pg, avx_sum20, avx_Nor3, avx_de032); \
    \
    avx_sum3 = svmul_f64_x(pg, avx_Nor1, avx_de13); \
    avx_sum3 = svmla_f64_x(pg, avx_sum3, avx_Nor2, avx_de23); \
    avx_sum3 = svmla_f64_x(pg, avx_sum3, avx_Nor3, avx_de33); \
    \
    avx_sum30 = svmul_f64_x(pg, avx_Nor1, avx_de013); \
    avx_sum30 = svmla_f64_x(pg, avx_sum30, avx_Nor2, avx_de023); \
    avx_sum30 = svmla_f64_x(pg, avx_sum30, avx_Nor3, avx_de033); \
    \
    avx_sum1 = svmul_f64_x(pg, avx_sum1, avx_dsmu); \
    avx_sum2 = svmul_f64_x(pg, avx_sum2, avx_dsmu); \
    avx_sum3 = svmul_f64_x(pg, avx_sum3, avx_dsmu); \
    avx_sum10 = svmul_f64_x(pg, avx_sum10, avx_dsmu0); \
    avx_sum20 = svmul_f64_x(pg, avx_sum20, avx_dsmu0); \
    avx_sum30 = svmul_f64_x(pg, avx_sum30, avx_dsmu0); \
    \
    avx_dyda1 = svmla_f64_x(pg, avx_dyda1, avx_Area, svadd_f64_x(pg, avx_sum1, avx_sum10)); \
    avx_dyda2 = svmla_f64_x(pg, avx_dyda2, avx_Area, svadd_f64_x(pg, avx_sum2, avx_sum20)); \
    avx_dyda3 = svmla_f64_x(pg, avx_dyda3, avx_Area, svadd_f64_x(pg, avx_sum3, avx_sum30)); \
    \
    avx_d = svmla_f64_x(pg, avx_d, avx_Area, svmul_f64_x(pg, avx_lmu, avx_lmu0)); \
    avx_d1 = svadd_f64_x(pg, avx_d1, svdiv_f64_x(pg, svmul_f64_x(pg, svmul_f64_x(pg, avx_Area, avx_lmu), avx_lmu0), svadd_f64_x(pg, avx_lmu, avx_lmu0)));

// end of inner_calc

#define INNER_CALC_DSMU \
    svfloat64_t avx_Area = svld1_f64(pg, &Area[i]); \
    svfloat64_t avx_dnom = svadd_f64_x(pg, avx_lmu, avx_lmu0); \
    svfloat64_t avx_s = svmul_f64_x(pg, svmul_f64_x(pg, avx_lmu, avx_lmu0), svadd_f64_x(pg, avx_cl, svdiv_f64_x(pg, avx_cls, avx_dnom))); \
    svfloat64_t avx_pdbr = svmul_f64_x(pg, svld1_f64(pg, &Darea[i]), avx_s); \
    svfloat64_t avx_pbr = svmul_f64_x(pg, avx_Area, avx_s); \
    svfloat64_t avx_powdnom = svdiv_f64_x(pg, avx_lmu0, avx_dnom); \
    avx_powdnom = svmul_f64_x(pg, avx_powdnom, avx_powdnom); \
    svfloat64_t avx_dsmu = svmla_f64_x(pg, svmul_f64_x(pg, avx_cls, avx_powdnom), avx_cl, avx_lmu0); \
    avx_powdnom = svdiv_f64_x(pg, avx_lmu, avx_dnom); \
    avx_powdnom = svmul_f64_x(pg, avx_powdnom, avx_powdnom); \
    svfloat64_t avx_dsmu0 = svmla_f64_x(pg, svmul_f64_x(pg, avx_cls, avx_powdnom), avx_cl, avx_lmu);
// end of inner_calc_dsmu


//double bright_fma(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef)
#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br)
{
	int i, j, k;
	incl_count = 0;

	//double cos_alpha, cl, cls, alpha, //br
	//	e[4], e0[4],
	//	php[N_PHOT_PAR + 1], dphp[N_PHOT_PAR + 1],
	//	de[4][4], de0[4][4], tmat[4][4],
	//	dtm[4][4][4];

	//svfloat64_t *Dg_row[MAX_N_FAC + 3], dbr[MAX_N_FAC + 3];

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);			/* Lambert */
    cls = cg[ncoef];					/* Lommel-Seeliger */
    //cos_alpha = dot_product(ee, ee0);
    dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php); /* computes also Scale */

	matrix(cg[ncoef0], t, tmat, dtm);

	// br = 0;
    /* Directions (and ders.) in the rotating system */
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

	/*Integrated brightness (phase coeff. used later) */
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
	svfloat64_t avx_tiny = svdup_n_f64(TINY);
	svfloat64_t avx_cl = svdup_n_f64(cl);
	svfloat64_t avx_cls = svdup_n_f64(cls);
	svfloat64_t avx_11 = svdup_n_f64(1.0);
	svfloat64_t avx_Scale = svdup_n_f64(Scale);
	svfloat64_t res_br = svdup_n_f64(0.0);
	svfloat64_t avx_dyda1 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda2 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda3 = svdup_n_f64(0.0);
	svfloat64_t avx_d = svdup_n_f64(0.0);
	svfloat64_t avx_d1 = svdup_n_f64(0.0);
	int cnt = svcntd();
	double g[svcntd()];

	for (i = 0; i < Numfac; i += cnt)
	{
		svbool_t pg = svwhilelt_b64(i, Numfac);

		svfloat64_t avx_lmu, avx_lmu0;
		svfloat64_t avx_Nor1 = svld1_f64(pg, &Nor[0][i]);
		svfloat64_t avx_Nor2 = svld1_f64(pg, &Nor[1][i]);
		svfloat64_t avx_Nor3 = svld1_f64(pg, &Nor[2][i]);
		svfloat64_t avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr;
		svfloat64_t avx_Area;

		avx_lmu = svmul_f64_x(pg, avx_e1, avx_Nor1);
		avx_lmu = svmla_f64_x(pg, avx_lmu, avx_e2, avx_Nor2);
		avx_lmu = svmla_f64_x(pg, avx_lmu, avx_e3, avx_Nor3);

		avx_lmu0 = svmul_f64_x(pg, avx_e01, avx_Nor1);
		avx_lmu0 = svmla_f64_x(pg, avx_lmu0, avx_e02, avx_Nor2);
		avx_lmu0 = svmla_f64_x(pg, avx_lmu0, avx_e03, avx_Nor3);

		svbool_t cmpe = svcmpgt_f64(pg, avx_lmu, avx_tiny);
		svbool_t cmpe0 = svcmpgt_f64(pg, avx_lmu0, avx_tiny);
		svbool_t cmp = svand_z(pg, cmpe, cmpe0);

		/* find a better solution
		svbool_t pred = svptrue_b64();
		int icmp = 0;
    	for (int i = 0; i < cnt; i++) {
        	if (svptest_first(pred) {
			  icmp |= 1 << b;
			}
        	pred = svptest_next(pred);
    	}
		*/

        // workaround
		double x_cmpe[svcntd()];
		double x_cmpe0[svcntd()];
		svst1_f64(pg, x_cmpe, avx_lmu);
		svst1_f64(pg, x_cmpe0, avx_lmu0);
		int icmp = 0;
		for (int b = 0; b < svcntd(); b++) {
		  	if (x_cmpe[b] > TINY && x_cmpe0[b] > TINY) {
                icmp |= 1 << b;
			}
		}
		// TODO workaround

		if (svptest_any(pg, cmp)) //if (icmp)
		{
			INNER_CALC_DSMU

			svfloat64_t avx_zero = svdup_n_f64(0.0);
			avx_pbr = svsel_f64(cmp, avx_pbr, avx_zero);
			avx_dsmu = svsel_f64(cmp, avx_dsmu, avx_zero);
			avx_dsmu0 = svsel_f64(cmp, avx_dsmu0, avx_zero);
			avx_lmu = svsel_f64(cmp, avx_lmu, avx_zero);
			avx_lmu0 = svsel_f64(cmp, avx_lmu0, avx_11); //abychom nedelili nulou

			svst1_f64(pg, g, avx_pdbr);
			for (int j = 0; j < cnt; j++) {
    			if (icmp & (1 << j)) {
        			Dg_row[incl_count] = (svfloat64_t*)&Dg[i + j];
        			dbr[incl_count++] = svdup_n_f64(g[j]);
    			}
			}
			INNER_CALC
		}
	}

	dbr[incl_count] = svdup_n_f64(0.0);
	Dg_row[incl_count] = Dg_row[0];
    br = svaddv_f64(svptrue_b64(), res_br);

	/* Derivatives of brightness w.r.t. g-coeffs */
	int ncoef03 = ncoef0 - 3, dgi = 0;
	for (i = 0; i < ncoef03; i += cnt) //1 * cnt doubles
	{
		svfloat64_t tmp1 = svdup_n_f64(0.0);

		for (j = 0; j < incl_count; j += 2)
		{
			svfloat64_t *Dgrow, *Dgrow1, pdbr, pdbr1;

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];

			tmp1 = svmla_f64_x(svptrue_b64(), svmla_f64_x(svptrue_b64(), tmp1, pdbr, Dgrow[0]), pdbr1, Dgrow1[0]);
		}
		dgi++;
		tmp1 = svmul_f64_x(svptrue_b64(), tmp1, avx_Scale);
		svst1_f64(svptrue_b64(), &dyda[i], tmp1);
	}

	/* Ders. of brightness w.r.t. rotation parameters */
	dyda[ncoef0 - 3 + 1 - 1] = svaddv_f64(svptrue_b64(), avx_dyda1) * Scale;
	dyda[ncoef0 - 3 + 2 - 1] = svaddv_f64(svptrue_b64(), avx_dyda2) * Scale;
	dyda[ncoef0 - 3 + 3 - 1] = svaddv_f64(svptrue_b64(), avx_dyda3) * Scale;
	/* Ders. of br. w.r.t. cl, cls */
	dyda[ncoef - 1 - 1] = svaddv_f64(svptrue_b64(), avx_d) * Scale * cl;
	dyda[ncoef - 1] = svaddv_f64(svptrue_b64(), avx_d1) * Scale;

	/* Ders. of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		dyda[ncoef0 + i - 1] = br * dphp[i];

	/* Scaled brightness */
	br *= Scale;
}
