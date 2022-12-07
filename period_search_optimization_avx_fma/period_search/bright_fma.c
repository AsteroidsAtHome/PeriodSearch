/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <immintrin.h>

#define INNER_CALC \
		 res_br=_mm256_add_pd(res_br,avx_pbr);	\
			__m256d avx_sum1,avx_sum10,avx_sum2,avx_sum20,avx_sum3,avx_sum30; \
			\
				avx_sum1=_mm256_mul_pd(avx_Nor1,avx_de11); \
				avx_sum1=_mm256_fmadd_pd(avx_Nor2,avx_de21, avx_sum1); \
				avx_sum1=_mm256_fmadd_pd(avx_Nor3,avx_de31, avx_sum1); \
\
				avx_sum10=_mm256_mul_pd(avx_Nor1,avx_de011); \
				avx_sum10=_mm256_fmadd_pd(avx_Nor2,avx_de021, avx_sum10); \
				avx_sum10=_mm256_fmadd_pd(avx_Nor3,avx_de031, avx_sum10); \
				\
				avx_sum2=_mm256_mul_pd(avx_Nor1,avx_de12); \
				avx_sum2=_mm256_fmadd_pd(avx_Nor2,avx_de22, avx_sum2); \
				avx_sum2=_mm256_fmadd_pd(avx_Nor3,avx_de32, avx_sum2); \
				\
				avx_sum20=_mm256_mul_pd(avx_Nor1,avx_de012); \
				avx_sum20=_mm256_fmadd_pd(avx_Nor2,avx_de022, avx_sum20); \
				avx_sum20=_mm256_fmadd_pd(avx_Nor3,avx_de032, avx_sum20); \
				\
				avx_sum3=_mm256_mul_pd(avx_Nor1,avx_de13); \
				avx_sum3=_mm256_fmadd_pd(avx_Nor2,avx_de23, avx_sum3); \
				avx_sum3=_mm256_fmadd_pd(avx_Nor3,avx_de33, avx_sum3); \
				\
				avx_sum30=_mm256_mul_pd(avx_Nor1,avx_de013); \
				avx_sum30=_mm256_fmadd_pd(avx_Nor2,avx_de023, avx_sum30); \
				avx_sum30=_mm256_fmadd_pd(avx_Nor3,avx_de033, avx_sum30); \
				\
			avx_sum1=_mm256_mul_pd(avx_sum1,avx_dsmu); \
			avx_sum2=_mm256_mul_pd(avx_sum2,avx_dsmu); \
			avx_sum3=_mm256_mul_pd(avx_sum3,avx_dsmu); \
			avx_sum10=_mm256_mul_pd(avx_sum10,avx_dsmu0); \
			avx_sum20=_mm256_mul_pd(avx_sum20,avx_dsmu0); \
			avx_sum30=_mm256_mul_pd(avx_sum30,avx_dsmu0); \
			\
			avx_dyda1=_mm256_fmadd_pd(avx_Area,_mm256_add_pd(avx_sum1,avx_sum10), avx_dyda1); \
			avx_dyda2=_mm256_fmadd_pd(avx_Area,_mm256_add_pd(avx_sum2,avx_sum20), avx_dyda2); \
			avx_dyda3=_mm256_fmadd_pd(avx_Area,_mm256_add_pd(avx_sum3,avx_sum30), avx_dyda3); \
			\
			avx_d=_mm256_fmadd_pd(_mm256_mul_pd(avx_lmu,avx_lmu0),avx_Area, avx_d); \
			avx_d1=_mm256_add_pd(avx_d1,_mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(avx_Area,avx_lmu),avx_lmu0),_mm256_add_pd(avx_lmu,avx_lmu0)));
// end of inner_calc
#define INNER_CALC_DSMU \
	  avx_Area=_mm256_load_pd(&Area[i]); \
	  avx_dnom=_mm256_add_pd(avx_lmu,avx_lmu0); \
	  avx_s=_mm256_mul_pd(_mm256_mul_pd(avx_lmu,avx_lmu0),_mm256_add_pd(avx_cl,_mm256_div_pd(avx_cls,avx_dnom))); \
	  avx_pdbr=_mm256_mul_pd(_mm256_load_pd(&Darea[i]),avx_s); \
	  avx_pbr=_mm256_mul_pd(avx_Area,avx_s); \
	  avx_powdnom=_mm256_div_pd(avx_lmu0,avx_dnom); \
	  avx_powdnom=_mm256_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu=_mm256_fmadd_pd(avx_cl,avx_lmu0, _mm256_mul_pd(avx_cls,avx_powdnom)); \
	  avx_powdnom=_mm256_div_pd(avx_lmu,avx_dnom); \
	  avx_powdnom=_mm256_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu0=_mm256_fmadd_pd(avx_cl,avx_lmu, _mm256_mul_pd(avx_cls,avx_powdnom));
// end of inner_calc_dsmu


double bright_fma(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef)
{
	int ncoef0, i, j, k,
		incl_count = 0;

	double cos_alpha, br, cl, cls, alpha,
		e[4], e0[4],
		php[N_PHOT_PAR + 1], dphp[N_PHOT_PAR + 1],
		de[4][4], de0[4][4], tmat[4][4],
		dtm[4][4][4];

	__m256d *Dg_row[MAX_N_FAC + 3], dbr[MAX_N_FAC + 3];

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */
	cos_alpha = dot_product(ee, ee0);
	alpha = acos(cos_alpha);
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
			e[i] = e[i] + tmat[i][j] * ee[j];
			e0[i] = e0[i] + tmat[i][j] * ee0[j];
			de[i][j] = 0;
			de0[i][j] = 0;
			for (k = 1; k <= 3; k++)
			{
				de[i][j] = de[i][j] + dtm[j][i][k] * ee[k];
				de0[i][j] = de0[i][j] + dtm[j][i][k] * ee0[k];
			}
		}
	}

	/*Integrated brightness (phase coeff. used later) */
	__m256d avx_e1 = _mm256_broadcast_sd(&e[1]);
	__m256d avx_e2 = _mm256_broadcast_sd(&e[2]);
	__m256d avx_e3 = _mm256_broadcast_sd(&e[3]);
	__m256d avx_e01 = _mm256_broadcast_sd(&e0[1]);
	__m256d avx_e02 = _mm256_broadcast_sd(&e0[2]);
	__m256d avx_e03 = _mm256_broadcast_sd(&e0[3]);
	__m256d avx_de11 = _mm256_broadcast_sd(&de[1][1]);
	__m256d avx_de12 = _mm256_broadcast_sd(&de[1][2]);
	__m256d avx_de13 = _mm256_broadcast_sd(&de[1][3]);
	__m256d avx_de21 = _mm256_broadcast_sd(&de[2][1]);
	__m256d avx_de22 = _mm256_broadcast_sd(&de[2][2]);
	__m256d avx_de23 = _mm256_broadcast_sd(&de[2][3]);
	__m256d avx_de31 = _mm256_broadcast_sd(&de[3][1]);
	__m256d avx_de32 = _mm256_broadcast_sd(&de[3][2]);
	__m256d avx_de33 = _mm256_broadcast_sd(&de[3][3]);
	__m256d avx_de011 = _mm256_broadcast_sd(&de0[1][1]);
	__m256d avx_de012 = _mm256_broadcast_sd(&de0[1][2]);
	__m256d avx_de013 = _mm256_broadcast_sd(&de0[1][3]);
	__m256d avx_de021 = _mm256_broadcast_sd(&de0[2][1]);
	__m256d avx_de022 = _mm256_broadcast_sd(&de0[2][2]);
	__m256d avx_de023 = _mm256_broadcast_sd(&de0[2][3]);
	__m256d avx_de031 = _mm256_broadcast_sd(&de0[3][1]);
	__m256d avx_de032 = _mm256_broadcast_sd(&de0[3][2]);
	__m256d avx_de033 = _mm256_broadcast_sd(&de0[3][3]);

	__m256d avx_tiny = _mm256_set1_pd(TINY);
	__m256d avx_cl = _mm256_set1_pd(cl), avx_cl1 = _mm256_set_pd(0, 0, 1, cl), avx_cls = _mm256_set1_pd(cls), avx_11 = _mm256_set1_pd(1.0);
	__m256d avx_Scale = _mm256_broadcast_sd(&Scale);
	__m256d res_br = _mm256_setzero_pd();
	__m256d avx_dyda1 = _mm256_setzero_pd();
	__m256d avx_dyda2 = _mm256_setzero_pd();
	__m256d avx_dyda3 = _mm256_setzero_pd();
	__m256d avx_d = _mm256_setzero_pd();
	__m256d avx_d1 = _mm256_setzero_pd();
	double g[4];

	for (i = 0; i < Numfac; i += 4)
	{
		__m256d avx_lmu, avx_lmu0, cmpe, cmpe0, cmp;
		__m256d avx_Nor1 = _mm256_load_pd(&Nor[0][i]);
		__m256d avx_Nor2 = _mm256_load_pd(&Nor[1][i]);
		__m256d avx_Nor3 = _mm256_load_pd(&Nor[2][i]);
		__m256d avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr;
		__m256d avx_Area;

		avx_lmu = _mm256_mul_pd(avx_e1, avx_Nor1);
		avx_lmu = _mm256_fmadd_pd(avx_e2, avx_Nor2, avx_lmu);
		//avx_lmu = _mm256_add_pd(avx_lmu, _mm256_mul_pd(avx_e2, avx_Nor2));
		avx_lmu = _mm256_fmadd_pd(avx_e3, avx_Nor3, avx_lmu);
		//avx_lmu = _mm256_add_pd(avx_lmu, _mm256_mul_pd(avx_e3, avx_Nor3));
		avx_lmu0 = _mm256_mul_pd(avx_e01, avx_Nor1);
		avx_lmu0 = _mm256_fmadd_pd(avx_e02, avx_Nor2, avx_lmu0);
		//avx_lmu0 = _mm256_add_pd(avx_lmu0, _mm256_mul_pd(avx_e02, avx_Nor2));
		avx_lmu0 = _mm256_fmadd_pd(avx_e03, avx_Nor3, avx_lmu0);
		//avx_lmu0 = _mm256_add_pd(avx_lmu0, _mm256_mul_pd(avx_e03, avx_Nor3));

		cmpe = _mm256_cmp_pd(avx_lmu, avx_tiny, _CMP_GT_OS);
		cmpe0 = _mm256_cmp_pd(avx_lmu0, avx_tiny, _CMP_GT_OS);
		cmp = _mm256_and_pd(cmpe, cmpe0);
		int icmp = _mm256_movemask_pd(cmp);

		if (icmp)
		{
			INNER_CALC_DSMU

				avx_pbr = _mm256_blendv_pd(_mm256_setzero_pd(), avx_pbr, cmp);
			avx_dsmu = _mm256_blendv_pd(_mm256_setzero_pd(), avx_dsmu, cmp);
			avx_dsmu0 = _mm256_blendv_pd(_mm256_setzero_pd(), avx_dsmu0, cmp);
			avx_lmu = _mm256_blendv_pd(_mm256_setzero_pd(), avx_lmu, cmp);
			avx_lmu0 = _mm256_blendv_pd(avx_11, avx_lmu0, cmp); //abychom nedelili nulou

			_mm256_store_pd(g, avx_pdbr);
			if (icmp & 1)
			{
				Dg_row[incl_count] = (__m256d*)&Dg[i];
				dbr[incl_count++] = _mm256_broadcast_sd(&g[0]);
			}
			if (icmp & 2)
			{
				Dg_row[incl_count] = (__m256d*)&Dg[i + 1];
				dbr[incl_count++] = _mm256_broadcast_sd(&g[1]);
			}
			if (icmp & 4)
			{
				Dg_row[incl_count] = (__m256d*)&Dg[i + 2];
				dbr[incl_count++] = _mm256_broadcast_sd(&g[2]);
			}
			if (icmp & 8)
			{
				Dg_row[incl_count] = (__m256d*)&Dg[i + 3];
				dbr[incl_count++] = _mm256_broadcast_sd(&g[3]);
			}
			INNER_CALC
		}
	}

	dbr[incl_count] = _mm256_setzero_pd();
	//   dbr[incl_count+1]=_mm256_setzero_pd();
	Dg_row[incl_count] = Dg_row[0];
	//   Dg_row[incl_count+1] = Dg_row[0];
	res_br = _mm256_hadd_pd(res_br, res_br);
	res_br = _mm256_add_pd(res_br, _mm256_permute2f128_pd(res_br, res_br, 1));
	_mm256_storeu_pd(g, res_br);
	br = g[0];

	/* Derivatives of brightness w.r.t. g-coeffs */
	int ncoef03 = ncoef0 - 3, dgi = 0, cyklus1 = (ncoef03 / 12) * 12;

	for (i = 0; i < cyklus1; i += 12) //3 * 4doubles
	{
		__m256d tmp1 = _mm256_setzero_pd();
		__m256d tmp2 = _mm256_setzero_pd();
		__m256d tmp3 = _mm256_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			__m256d *Dgrow, *Dgrow1, pdbr, pdbr1;

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];

			tmp1 = _mm256_fmadd_pd(pdbr1, Dgrow1[0], _mm256_fmadd_pd(pdbr, Dgrow[0], tmp1));
			//tmp1 = _mm256_add_pd(_mm256_add_pd(tmp1, _mm256_mul_pd(pdbr, Dgrow[0])), _mm256_mul_pd(pdbr1, Dgrow1[0]));
			tmp2 = _mm256_fmadd_pd(pdbr1, Dgrow1[1], _mm256_fmadd_pd(pdbr, Dgrow[1], tmp2));
			//tmp2 = _mm256_add_pd(_mm256_add_pd(tmp2, _mm256_mul_pd(pdbr, Dgrow[1])), _mm256_mul_pd(pdbr1, Dgrow1[1]));
			tmp3 = _mm256_fmadd_pd(pdbr1, Dgrow1[2], _mm256_fmadd_pd(pdbr, Dgrow[2], tmp3));
			//tmp3 = _mm256_add_pd(_mm256_add_pd(tmp3, _mm256_mul_pd(pdbr, Dgrow[2])), _mm256_mul_pd(pdbr1, Dgrow1[2]));
		}
		dgi += 3;
		tmp1 = _mm256_mul_pd(tmp1, avx_Scale);
		_mm256_store_pd(&dyda[i], tmp1);
		tmp2 = _mm256_mul_pd(tmp2, avx_Scale);
		_mm256_store_pd(&dyda[i + 4], tmp2);
		tmp3 = _mm256_mul_pd(tmp3, avx_Scale);
		_mm256_store_pd(&dyda[i + 8], tmp3);
	}
	for (; i < ncoef03; i += 4) //1 * 4doubles
	{
		__m256d tmp1 = _mm256_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			__m256d *Dgrow, *Dgrow1, pdbr, pdbr1;

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];

			//tmp1 = _mm256_add_pd(_mm256_add_pd(tmp1, _mm256_mul_pd(pdbr, Dgrow[0])), _mm256_mul_pd(pdbr1, Dgrow1[0]));
			//auto a1 = _mm256_mul_pd(pdbr, Dgrow[0]);
			//auto c1 = _mm256_add_pd(tmp1, a1);
			//
			//auto b1 = _mm256_mul_pd(pdbr1, Dgrow1[0]);
			//tmp1 = _mm256_add_pd(c1, b1);

			//auto c = _mm256_fmadd_pd(pdbr, Dgrow[0], tmp1);
			//tmp1 = _mm256_fmadd_pd(pdbr1, Dgrow1[0], c);
			
			tmp1 = _mm256_fmadd_pd(pdbr1, Dgrow1[0], _mm256_fmadd_pd(pdbr, Dgrow[0], tmp1));
		}
		dgi++;
		tmp1 = _mm256_mul_pd(tmp1, avx_Scale);
		_mm256_store_pd(&dyda[i], tmp1);
	}
	/* Ders. of brightness w.r.t. rotation parameters */
	avx_dyda1 = _mm256_hadd_pd(avx_dyda1, avx_dyda2);
	avx_dyda1 = _mm256_add_pd(avx_dyda1, _mm256_permute2f128_pd(avx_dyda1, avx_dyda1, 1));
	avx_dyda1 = _mm256_mul_pd(avx_dyda1, avx_Scale);
	_mm256_store_pd(g, avx_dyda1);
	dyda[ncoef0 - 3 + 1 - 1] = g[0];
	dyda[ncoef0 - 3 + 2 - 1] = g[1];
	avx_dyda3 = _mm256_hadd_pd(avx_dyda3, avx_dyda3);
	avx_dyda3 = _mm256_add_pd(avx_dyda3, _mm256_permute2f128_pd(avx_dyda3, avx_dyda3, 1));
	avx_dyda3 = _mm256_mul_pd(avx_dyda3, avx_Scale);
	_mm256_store_pd(g, avx_dyda3);
	dyda[ncoef0 - 3 + 3 - 1] = g[0];
	/* Ders. of br. w.r.t. cl, cls */
	avx_d = _mm256_hadd_pd(avx_d, avx_d1);
	__m256d avx_dperm = _mm256_permute2f128_pd(avx_d, avx_d, 1);
	avx_d = _mm256_add_pd(avx_d, avx_dperm);
	//avx_d = _mm256_add_pd(avx_d, _mm256_permute2f128_pd(avx_d, avx_d, 1));
	avx_d = _mm256_mul_pd(avx_d, avx_Scale);
	avx_d = _mm256_mul_pd(avx_d, avx_cl1);
	_mm256_store_pd(g, avx_d);
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

	return(br);
}
