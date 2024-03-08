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
    svfloat64_t sve_sum1, sve_sum10, sve_sum2, sve_sum20, sve_sum3, sve_sum30; \
    \
    sve_sum1 = svmul_f64_x(pg, avx_Nor1, avx_de11); \
    sve_sum1 = svmla_f64_x(pg, sve_sum1, avx_Nor2, avx_de21); \
    sve_sum1 = svmla_f64_x(pg, sve_sum1, avx_Nor3, avx_de31); \
    \
    sve_sum10 = svmul_f64_x(pg, avx_Nor1, avx_de011); \
    sve_sum10 = svmla_f64_x(pg, sve_sum10, avx_Nor2, avx_de021); \
    sve_sum10 = svmla_f64_x(pg, sve_sum10, avx_Nor3, avx_de031); \
    \
    sve_sum2 = svmul_f64_x(pg, avx_Nor1, avx_de12); \
    sve_sum2 = svmla_f64_x(pg, sve_sum2, avx_Nor2, avx_de22); \
    sve_sum2 = svmla_f64_x(pg, sve_sum2, avx_Nor3, avx_de32); \
    \
    sve_sum20 = svmul_f64_x(pg, avx_Nor1, avx_de012); \
    sve_sum20 = svmla_f64_x(pg, sve_sum20, avx_Nor2, avx_de022); \
    sve_sum20 = svmla_f64_x(pg, sve_sum20, avx_Nor3, avx_de032); \
    \
    sve_sum3 = svmul_f64_x(pg, avx_Nor1, avx_de13); \
    sve_sum3 = svmla_f64_x(pg, sve_sum3, avx_Nor2, avx_de23); \
    sve_sum3 = svmla_f64_x(pg, sve_sum3, avx_Nor3, avx_de33); \
    \
    sve_sum30 = svmul_f64_x(pg, avx_Nor1, avx_de013); \
    sve_sum30 = svmla_f64_x(pg, sve_sum30, avx_Nor2, avx_de023); \
    sve_sum30 = svmla_f64_x(pg, sve_sum30, avx_Nor3, avx_de033); \
    \
    sve_sum1 = svmul_f64_x(pg, sve_sum1, avx_dsmu); \
    sve_sum2 = svmul_f64_x(pg, sve_sum2, avx_dsmu); \
    sve_sum3 = svmul_f64_x(pg, sve_sum3, avx_dsmu); \
    sve_sum10 = svmul_f64_x(pg, sve_sum10, avx_dsmu0); \
    sve_sum20 = svmul_f64_x(pg, sve_sum20, avx_dsmu0); \
    sve_sum30 = svmul_f64_x(pg, sve_sum30, avx_dsmu0); \
    \
    avx_dyda1 = svmla_f64_x(pg, avx_Area, svadd_f64_x(pg, sve_sum1, sve_sum10), avx_dyda1); \
    avx_dyda2 = svmla_f64_x(pg, avx_Area, svadd_f64_x(pg, sve_sum2, sve_sum20), avx_dyda2); \
    avx_dyda3 = svmla_f64_x(pg, avx_Area, svadd_f64_x(pg, sve_sum3, sve_sum30), avx_dyda3); \
    \
    avx_d = svmla_f64_x(pg, avx_Area, svmul_f64_x(pg, avx_lmu, avx_lmu0), avx_d); \
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
    svfloat64_t avx_dsmu = svmla_f64_x(pg, avx_cl, avx_lmu0, svmul_f64_x(pg, avx_cls, avx_powdnom)); \
    avx_powdnom = svdiv_f64_x(pg, avx_lmu, avx_dnom); \
    avx_powdnom = svmul_f64_x(pg, avx_powdnom, avx_powdnom); \
    svfloat64_t avx_dsmu0 = svmla_f64_x(pg, avx_cl, avx_lmu, svmul_f64_x(pg, avx_cls, avx_powdnom));
// end of inner_calc_dsmu


//double bright_fma(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef)
#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br)
{
	int i, j, k; //ncoef0,
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
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php); /* computes also Scale */

	matrix(cg[ncoef0], t, tmat, dtm);

	//   br = 0;
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
	svfloat64_t avx_e1 = svdup_f64(e[1]);
	svfloat64_t avx_e2 = svdup_f64(e[2]);
	svfloat64_t avx_e3 = svdup_f64(e[3]);
	svfloat64_t avx_e01 = svdup_f64(e0[1]);
	svfloat64_t avx_e02 = svdup_f64(e0[2]);
	svfloat64_t avx_e03 = svdup_f64(e0[3]);
	svfloat64_t avx_de11 = svdup_f64(de[1][1]);
	svfloat64_t avx_de12 = svdup_f64(de[1][2]);
	svfloat64_t avx_de13 = svdup_f64(de[1][3]);
	svfloat64_t avx_de21 = svdup_f64(de[2][1]);
	svfloat64_t avx_de22 = svdup_f64(de[2][2]);
	svfloat64_t avx_de23 = svdup_f64(de[2][3]);
	svfloat64_t avx_de31 = svdup_f64(de[3][1]);
	svfloat64_t avx_de32 = svdup_f64(de[3][2]);
	svfloat64_t avx_de33 = svdup_f64(de[3][3]);
	svfloat64_t avx_de011 = svdup_f64(de0[1][1]);
	svfloat64_t avx_de012 = svdup_f64(de0[1][2]);
	svfloat64_t avx_de013 = svdup_f64(de0[1][3]);
	svfloat64_t avx_de021 = svdup_f64(de0[2][1]);
	svfloat64_t avx_de022 = svdup_f64(de0[2][2]);
	svfloat64_t avx_de023 = svdup_f64(de0[2][3]);
	svfloat64_t avx_de031 = svdup_f64(de0[3][1]);
	svfloat64_t avx_de032 = svdup_f64(de0[3][2]);
	svfloat64_t avx_de033 = svdup_f64(de0[3][3]);
	svfloat64_t avx_tiny = svdup_n_f64(TINY);

	svfloat64_t avx_cl = svdup_n_f64(cl);
	svfloat64_t avx_cl1 = svset_f64(svptrue_b64(), 2, cl);
	svfloat64_t avx_cls = svdup_n_f64(cls);
	svfloat64_t avx_11 = svdup_n_f64(1.0);

	svfloat64_t avx_Scale = svdup_f64(Scale);
	svfloat64_t res_br = svdup_n_f64(0.0);
	svfloat64_t avx_dyda1 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda2 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda3 = svdup_n_f64(0.0);
	svfloat64_t avx_d = svdup_n_f64(0.0);
	svfloat64_t avx_d1 = svdup_n_f64(0.0);
	int cnt = svcntd();
	double g[cnt];

	for (i = 0; i < Numfac; i += cnt)
	{
		svbool_t pg = svwhilelt_b64(i, Numfac);

		svfloat64_t avx_lmu, avx_lmu0, cmpe, cmpe0, cmp;
		svfloat64_t avx_Nor1 = svld1_f64(pg, &Nor[0][i]);
		svfloat64_t avx_Nor2 = svld1_f64(pg, &Nor[1][i]);
		svfloat64_t avx_Nor3 = svld1_f64(pg, &Nor[2][i]);
		svfloat64_t avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr;
		svfloat64_t avx_Area;

		avx_lmu = svmul_f64_x(pg, avx_e1, avx_Nor1);
		avx_lmu = svmla_f64_x(pg, avx_e2, avx_Nor2, avx_lmu);
		avx_lmu = svmla_f64_x(pg, avx_e3, avx_Nor3, avx_lmu);

		avx_lmu0 = svmul_f64_x(pg, avx_e01, avx_Nor1);
		avx_lmu0 = svmla_f64_x(pg, avx_lmu0, avx_e02, avx_Nor2);
		avx_lmu0 = svmla_f64_x(pg, avx_lmu0, avx_e03, avx_Nor3);

		cmpe = svcgt_f64(pg, avx_lmu, avx_tiny);
		cmpe0 = svcgt_f64(pg, avx_lmu0, avx_tiny);
		cmp = svand_f64(pg, cmpe, cmpe0);

		int icmp = svcvt_z_p_b64(cmp);

		if (icmp)
		{
			INNER_CALC_DSMU

			svfloat64_t avx_zero = svdup_f64(0.0);
			avx_pbr = svsel_f64(cmp, avx_pbr, avx_zero);
			avx_dsmu = svsel_f64(cmp, avx_dsmu, avx_zero);
			avx_dsmu0 = svsel_f64(cmp, avx_dsmu0, avx_zero);
			avx_lmu = svsel_f64(cmp, avx_lmu, avx_zero);
			avx_lmu0 = svsel_f64(cmp, avx_lmu0, avx_11); //abychom nedelili nulou

			svst1_f64(pg, g, avx_pdbr);
			if (icmp & 1)
			{
				Dg_row[incl_count] = (svfloat64_t*)&Dg[i];
				dbr[incl_count++] = svdup_f64(g[0]);
			}
			if (icmp & 2)
			{
				Dg_row[incl_count] = (svfloat64_t*)&Dg[i + 1];
				dbr[incl_count++] = svdup_f64(g[1]);
			}
			if (icmp & 4)
			{
				Dg_row[incl_count] = (svfloat64_t*)&Dg[i + 2];
				dbr[incl_count++] = svdup_f64(g[2]);
			}
			if (icmp & 8)
			{
				Dg_row[incl_count] = (svfloat64_t*)&Dg[i + 3];
				dbr[incl_count++] = svdup_f64(g[3]);
			}
			INNER_CALC
		}
	}

	dbr[incl_count] = svdup_n_f64(0.0);
	//   dbr[incl_count+1]=svdup_n_f64(0.0);
	Dg_row[incl_count] = Dg_row[0];
	//   Dg_row[incl_count+1] = Dg_row[0];

	res_br = svadd_f64(pg, svget2_f64(res_br, 0), svget2_f64(res_br, 1));
	res_br = svadd_f64(pg, res_br, svext_f64(res_br, svres_f64(), 1));
	svst1_f64(pg, g, res_br);
	br = g[0];

	/* Derivatives of brightness w.r.t. g-coeffs */
	int ncoef03 = ncoef0 - 3, dgi = 0, cyklus1 = (ncoef03 / 12) * 12;

	for (i = 0; i < cyklus1; i += 12) //3 * 4doubles
	{
		svfloat64_t tmp1 = svdup_n_f64(0.0);
		svfloat64_t tmp2 = svdup_n_f64(0.0);
		svfloat64_t tmp3 = svdup_n_f64(0.0);

		for (j = 0; j < incl_count; j += 2)
		{
			svbool_t pg2 = svwhilelt_b64(j, incl_count);

			svfloat64_t *Dgrow, *Dgrow1, pdbr, pdbr1;

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];

			tmp1 = svmla_f64_x(pg2, svmla_f64_x(pg2, tmp1, pdbr, Dgrow[0]), pdbr1, Dgrow1[0]);
			tmp2 = svmla_f64_x(pg2, svmla_f64_x(pg2, tmp2, pdbr, Dgrow[1]), pdbr1, Dgrow1[1]);
			tmp3 = svmla_f64_x(pg2, svmla_f64_x(pg2, tmp3, pdbr, Dgrow[2]), pdbr1, Dgrow1[2]);
		}
		dgi += 3;
		tmp1 = svmul_f64_x(pg, tmp1, avx_Scale)
		svst1_f64(pg, &dyda[i], tmp1);
		tmp2 = svmul_f64_x(pg, tmp2, avx_Scale)
		svst1_f64(pg, &dyda[i + 4], tmp2);
		tmp3 = svmul_f64_x(pg, tmp3, avx_Scale)
		svst1_f64(pg, &dyda[i + 8], tmp3);
	}
	for (; i < ncoef03; i += 4) //1 * 4doubles
	{
		svfloat64_t tmp1 = svdup_n_f64(0.0);

		for (j = 0; j < incl_count; j += 2)
		{
			svfloat64_t *Dgrow, *Dgrow1, pdbr, pdbr1;

			svbool_t pg3 = svwhilelt_b64(j, incl_count);

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];

			tmp1 = svmla_f64_x(pg3, svmla_f64_x(pg3, tmp1, pdbr, Dgrow[0]), pdbr1, Dgrow1[0]);
		}
		dgi++;
		tmp1 = svmul_f64_x(pg, tmp1, avx_Scale)
		svst1_f64(pg, &dyda[i], tmp1);
	}
	/* Ders. of brightness w.r.t. rotation parameters */
	avx_dyda1 = svadd_f64(svget2_f64(avx_dyda1, 0), svget2_f64(avx_dyda1, 1));
	avx_dyda1 = svadd_f64(avx_dyda1, svext_f64(avx_dyda1, svres_f64(), 1));
	avx_dyda1 = svmul_f64(avx_dyda1, avx_Scale);
	svst1_f64(pg, g, avx_dyda1);
	dyda[ncoef0 - 3 + 1 - 1] = g[0];
	dyda[ncoef0 - 3 + 2 - 1] = g[1];
	avx_dyda3 = svadd_f64(svget2_f64(avx_dyda3, 0), svget2_f64(avx_dyda3, 1));
	avx_dyda3 = svadd_f64(avx_dyda3, svext_f64(avx_dyda3, svres_f64(), 1));
	avx_dyda3 = svmul_f64(avx_dyda3, avx_Scale);
	svst1_f64(pg, g, avx_dyda3);
	dyda[ncoef0 - 3 + 3 - 1] = g[0];
	/* Ders. of br. w.r.t. cl, cls */
	avx_d = svadd_f64(svget2_f64(avx_d, 0), svget2_f64(avx_d1, 1));
	avx_d = svadd_f64(avx_d, svext_f64(avx_d, svres_f64(), 1));
	//avx_d = _mm256_add_pd(avx_d, _mm256_permute2f128_pd(avx_d, avx_d, 1));
	avx_d = svmul_f64(avx_d, avx_Scale);
	avx_d = svmul_f64(avx_d, avx_cl1);
	svst1_f64(pg, g, avx_d);
	dyda[ncoef - 1 - 1] = g[0];
	dyda[ncoef - 1] = g[1];

	/* Ders. of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		dyda[ncoef0 + i - 1] = br * dphp[i];
	/*     dyda[ncoef0+1-1] = br * dphp[1];
		 dyda[ncoef0+2-1] = br * dphp[2];
		 dyda[ncoef0+3-1] = br * dphp[3];*/

		 /* Scaled brightness */
	br *= Scale;

	/*printf("\ndyda[208]:\n");
	printf("_dyda[] ={");
	for(int q = 0; q <=207; q++)
	{
		printf("%.30f, ", dyda[q]);
	}
	printf("};\n"); */

	//return(br);
}
