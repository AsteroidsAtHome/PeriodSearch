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

inline __m512d hadd_pd(__m512d a, __m512d b) {
  __m512i idx1 = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
  __m512i idx2 = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
  return _mm512_add_pd(_mm512_mask_permutex2var_pd(a, 0xff, idx1, b), _mm512_mask_permutex2var_pd(a, 0xff, idx2, b));
}

inline __m512d blendv_pd(__m512d a, __m512d b, __m512d c) {
	__m512i result = _mm512_ternarylogic_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(b), _mm512_srai_epi64(_mm512_castpd_si512(c), 63), 0xd8);

	return _mm512_castsi512_pd(result);
}

// -mavx512dq
inline __m512d cmp_pd(__m512d a, __m512d b) {
	__m512i result = _mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_GT_OS));

	return _mm512_castsi512_pd(result);
}

inline int movemask_pd(__m512d a) {
	return (int) _mm512_cmpneq_epi64_mask(_mm512_setzero_si512(), _mm512_and_si512(_mm512_set1_epi64(0x8000000000000000ULL), _mm512_castpd_si512(a)));
}

inline __m512d permute4(__m512d a) {
  //1 2 3 4 5 6 7 8
  //5 6 7 8 1 2 3 4
  __m512i idx = _mm512_set_epi32(0, 11, 0, 10, 0, 9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4);
  return _mm512_mask_permutex2var_pd(a, 0xff, idx, a);
}

inline __m512d permute3(__m512d a) {
  //1 2 3 4 5 6 7 8
  //3 4 5 6 7 8 1 2
  __m512i idx = _mm512_set_epi32(0, 9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2);
  return _mm512_mask_permutex2var_pd(a, 0xff, idx, a);
}

inline __m512d hpermute_add_pd(__m512d a) {
  __m512d tmp = _mm512_add_pd(a, permute3(a));
  return _mm512_add_pd(tmp, permute4(tmp));
}

#define INNER_CALC \
		 res_br=_mm512_add_pd(res_br,avx_pbr);	\
			__m512d avx_sum1,avx_sum10,avx_sum2,avx_sum20,avx_sum3,avx_sum30; \
			\
				avx_sum1=_mm512_mul_pd(avx_Nor1,avx_de11); \
				avx_sum1=_mm512_fmadd_pd(avx_Nor2,avx_de21, avx_sum1); \
				avx_sum1=_mm512_fmadd_pd(avx_Nor3,avx_de31, avx_sum1); \
\
				avx_sum10=_mm512_mul_pd(avx_Nor1,avx_de011); \
				avx_sum10=_mm512_fmadd_pd(avx_Nor2,avx_de021, avx_sum10); \
				avx_sum10=_mm512_fmadd_pd(avx_Nor3,avx_de031, avx_sum10); \
				\
				avx_sum2=_mm512_mul_pd(avx_Nor1,avx_de12); \
				avx_sum2=_mm512_fmadd_pd(avx_Nor2,avx_de22, avx_sum2); \
				avx_sum2=_mm512_fmadd_pd(avx_Nor3,avx_de32, avx_sum2); \
				\
				avx_sum20=_mm512_mul_pd(avx_Nor1,avx_de012); \
				avx_sum20=_mm512_fmadd_pd(avx_Nor2,avx_de022, avx_sum20); \
				avx_sum20=_mm512_fmadd_pd(avx_Nor3,avx_de032, avx_sum20); \
				\
				avx_sum3=_mm512_mul_pd(avx_Nor1,avx_de13); \
				avx_sum3=_mm512_fmadd_pd(avx_Nor2,avx_de23, avx_sum3); \
				avx_sum3=_mm512_fmadd_pd(avx_Nor3,avx_de33, avx_sum3); \
				\
				avx_sum30=_mm512_mul_pd(avx_Nor1,avx_de013); \
				avx_sum30=_mm512_fmadd_pd(avx_Nor2,avx_de023, avx_sum30); \
				avx_sum30=_mm512_fmadd_pd(avx_Nor3,avx_de033, avx_sum30); \
				\
			avx_sum1=_mm512_mul_pd(avx_sum1,avx_dsmu); \
			avx_sum2=_mm512_mul_pd(avx_sum2,avx_dsmu); \
			avx_sum3=_mm512_mul_pd(avx_sum3,avx_dsmu); \
			avx_sum10=_mm512_mul_pd(avx_sum10,avx_dsmu0); \
			avx_sum20=_mm512_mul_pd(avx_sum20,avx_dsmu0); \
			avx_sum30=_mm512_mul_pd(avx_sum30,avx_dsmu0); \
			\
			avx_dyda1=_mm512_fmadd_pd(avx_Area,_mm512_add_pd(avx_sum1,avx_sum10), avx_dyda1); \
			avx_dyda2=_mm512_fmadd_pd(avx_Area,_mm512_add_pd(avx_sum2,avx_sum20), avx_dyda2); \
			avx_dyda3=_mm512_fmadd_pd(avx_Area,_mm512_add_pd(avx_sum3,avx_sum30), avx_dyda3); \
			\
			avx_d=_mm512_fmadd_pd(_mm512_mul_pd(avx_lmu,avx_lmu0),avx_Area, avx_d); \
			avx_d1=_mm512_add_pd(avx_d1,_mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(avx_Area,avx_lmu),avx_lmu0),_mm512_add_pd(avx_lmu,avx_lmu0)));
// end of inner_calc
#define INNER_CALC_DSMU \
	  avx_Area=_mm512_load_pd(&Area[i]); \
	  avx_dnom=_mm512_add_pd(avx_lmu,avx_lmu0); \
	  avx_s=_mm512_mul_pd(_mm512_mul_pd(avx_lmu,avx_lmu0),_mm512_add_pd(avx_cl,_mm512_div_pd(avx_cls,avx_dnom))); \
	  avx_pdbr=_mm512_mul_pd(_mm512_load_pd(&Darea[i]),avx_s); \
	  avx_pbr=_mm512_mul_pd(avx_Area,avx_s); \
	  avx_powdnom=_mm512_div_pd(avx_lmu0,avx_dnom); \
	  avx_powdnom=_mm512_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu=_mm512_fmadd_pd(avx_cl,avx_lmu0, _mm512_mul_pd(avx_cls,avx_powdnom)); \
	  avx_powdnom=_mm512_div_pd(avx_lmu,avx_dnom); \
	  avx_powdnom=_mm512_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu0=_mm512_fmadd_pd(avx_cl,avx_lmu, _mm512_mul_pd(avx_cls,avx_powdnom));
// end of inner_calc_dsmu


double bright_avx512(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef)
{
	int ncoef0, i, j, k,
		incl_count = 0;

	double cos_alpha, br, cl, cls, alpha,
		e[4], e0[4],
		php[N_PHOT_PAR + 1], dphp[N_PHOT_PAR + 1],
		de[4][4], de0[4][4], tmat[4][4],
		dtm[4][4][4];

	__m512d *Dg_row[MAX_N_FAC + 3], dbr[MAX_N_FAC + 3];

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
	__m512d avx_e1 = _mm512_set1_pd(e[1]);
	__m512d avx_e2 = _mm512_set1_pd(e[2]);
	__m512d avx_e3 = _mm512_set1_pd(e[3]);
	__m512d avx_e01 = _mm512_set1_pd(e0[1]);
	__m512d avx_e02 = _mm512_set1_pd(e0[2]);
	__m512d avx_e03 = _mm512_set1_pd(e0[3]);
	__m512d avx_de11 = _mm512_set1_pd(de[1][1]);
	__m512d avx_de12 = _mm512_set1_pd(de[1][2]);
	__m512d avx_de13 = _mm512_set1_pd(de[1][3]);
	__m512d avx_de21 = _mm512_set1_pd(de[2][1]);
	__m512d avx_de22 = _mm512_set1_pd(de[2][2]);
	__m512d avx_de23 = _mm512_set1_pd(de[2][3]);
	__m512d avx_de31 = _mm512_set1_pd(de[3][1]);
	__m512d avx_de32 = _mm512_set1_pd(de[3][2]);
	__m512d avx_de33 = _mm512_set1_pd(de[3][3]);
	__m512d avx_de011 = _mm512_set1_pd(de0[1][1]);
	__m512d avx_de012 = _mm512_set1_pd(de0[1][2]);
	__m512d avx_de013 = _mm512_set1_pd(de0[1][3]);
	__m512d avx_de021 = _mm512_set1_pd(de0[2][1]);
	__m512d avx_de022 = _mm512_set1_pd(de0[2][2]);
	__m512d avx_de023 = _mm512_set1_pd(de0[2][3]);
	__m512d avx_de031 = _mm512_set1_pd(de0[3][1]);
	__m512d avx_de032 = _mm512_set1_pd(de0[3][2]);
	__m512d avx_de033 = _mm512_set1_pd(de0[3][3]);

	__m512d avx_tiny = _mm512_set1_pd(TINY);
	__m512d avx_cl = _mm512_set1_pd(cl), avx_cl1 = _mm512_set_pd(0, 0, 0, 0, 0, 0, 1, cl), avx_cls = _mm512_set1_pd(cls), avx_11 = _mm512_set1_pd(1.0);
	__m512d avx_Scale = _mm512_set1_pd(Scale);
	__m512d res_br = _mm512_setzero_pd();
	__m512d avx_dyda1 = _mm512_setzero_pd();
	__m512d avx_dyda2 = _mm512_setzero_pd();
	__m512d avx_dyda3 = _mm512_setzero_pd();
	__m512d avx_d = _mm512_setzero_pd();
	__m512d avx_d1 = _mm512_setzero_pd();
	double g[8];

	for (i = 0; i < Numfac; i += 8)
	{
		__m512d avx_lmu, avx_lmu0, cmpe, cmpe0, cmp;
		__m512d avx_Nor1 = _mm512_load_pd(&Nor[0][i]);
		__m512d avx_Nor2 = _mm512_load_pd(&Nor[1][i]);
		__m512d avx_Nor3 = _mm512_load_pd(&Nor[2][i]);
		__m512d avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr;
		__m512d avx_Area;

		avx_lmu = _mm512_mul_pd(avx_e1, avx_Nor1);
		avx_lmu = _mm512_fmadd_pd(avx_e2, avx_Nor2, avx_lmu);
		//avx_lmu = _mm512_add_pd(avx_lmu, _mm512_mul_pd(avx_e2, avx_Nor2));
		avx_lmu = _mm512_fmadd_pd(avx_e3, avx_Nor3, avx_lmu);
		//avx_lmu = _mm512_add_pd(avx_lmu, _mm512_mul_pd(avx_e3, avx_Nor3));
		avx_lmu0 = _mm512_mul_pd(avx_e01, avx_Nor1);
		avx_lmu0 = _mm512_fmadd_pd(avx_e02, avx_Nor2, avx_lmu0);
		//avx_lmu0 = _mm512_add_pd(avx_lmu0, _mm512_mul_pd(avx_e02, avx_Nor2));
		avx_lmu0 = _mm512_fmadd_pd(avx_e03, avx_Nor3, avx_lmu0);
		//avx_lmu0 = _mm512_add_pd(avx_lmu0, _mm512_mul_pd(avx_e03, avx_Nor3));

		cmpe = cmp_pd(avx_lmu, avx_tiny);
		cmpe0 = cmp_pd(avx_lmu0, avx_tiny);
		cmp = _mm512_and_pd(cmpe, cmpe0);
		int icmp = movemask_pd(cmp);

		if (icmp)
		{
			INNER_CALC_DSMU

			avx_pbr = blendv_pd(_mm512_setzero_pd(), avx_pbr, cmp);
			avx_dsmu = blendv_pd(_mm512_setzero_pd(), avx_dsmu, cmp);
			avx_dsmu0 = blendv_pd(_mm512_setzero_pd(), avx_dsmu0, cmp);
			avx_lmu = blendv_pd(_mm512_setzero_pd(), avx_lmu, cmp);
			avx_lmu0 = blendv_pd(avx_11, avx_lmu0, cmp); //abychom nedelili nulou

			_mm512_store_pd(g, avx_pdbr);
			if (icmp & 1)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i];
				dbr[incl_count++] = _mm512_set1_pd(g[0]);
			}
			if (icmp & 2)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i + 1];
				dbr[incl_count++] = _mm512_set1_pd(g[1]);
			}
			if (icmp & 4)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i + 2];
				dbr[incl_count++] = _mm512_set1_pd(g[2]);
			}
			if (icmp & 8)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i + 3];
				dbr[incl_count++] = _mm512_set1_pd(g[3]);
			}
			if (icmp & 16)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i + 4];
				dbr[incl_count++] = _mm512_set1_pd(g[4]);
			}
			if (icmp & 32)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i + 5];
				dbr[incl_count++] = _mm512_set1_pd(g[5]);
			}
			if (icmp & 64)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i + 6];
				dbr[incl_count++] = _mm512_set1_pd(g[6]);
			}
			if (icmp & 128)
			{
				Dg_row[incl_count] = (__m512d*)&Dg[i + 7];
				dbr[incl_count++] = _mm512_set1_pd(g[7]);
			}
			INNER_CALC
		}
	}

	dbr[incl_count] = _mm512_setzero_pd();
	//   dbr[incl_count+1]=_mm512_setzero_pd();
	Dg_row[incl_count] = Dg_row[0];
	//   Dg_row[incl_count+1] = Dg_row[0];
	res_br = hadd_pd(res_br, res_br);
	res_br = hpermute_add_pd(res_br);
	_mm512_storeu_pd(g, res_br);
	br = g[0];

	/* Derivatives of brightness w.r.t. g-coeffs */
	int ncoef03 = ncoef0 - 3, dgi = 0, cyklus1 = (ncoef03 / 16) * 16;

	for (i = 0; i < cyklus1; i += 16) //2 * 8 doubles
	{
		__m512d tmp1 = _mm512_setzero_pd();
		__m512d tmp2 = _mm512_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			__m512d *Dgrow, *Dgrow1, pdbr, pdbr1;

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];

			tmp1 = _mm512_fmadd_pd(pdbr1, Dgrow1[0], _mm512_fmadd_pd(pdbr, Dgrow[0], tmp1));
			tmp2 = _mm512_fmadd_pd(pdbr1, Dgrow1[1], _mm512_fmadd_pd(pdbr, Dgrow[1], tmp2));
		}
		dgi += 2;
		tmp1 = _mm512_mul_pd(tmp1, avx_Scale);
		_mm512_store_pd(&dyda[i], tmp1);
		tmp2 = _mm512_mul_pd(tmp2, avx_Scale);
		_mm512_store_pd(&dyda[i + 8], tmp2);
	}

	for (; i < ncoef03; i += 8) //1 * 8 doubles
	{
		__m512d tmp1 = _mm512_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			__m512d *Dgrow, *Dgrow1, pdbr, pdbr1;

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];

		    tmp1 = _mm512_fmadd_pd(pdbr1, Dgrow1[0], _mm512_fmadd_pd(pdbr, Dgrow[0], tmp1));
		}
		dgi++;
		tmp1 = _mm512_mul_pd(tmp1, avx_Scale);
		_mm512_store_pd(&dyda[i], tmp1);
    }

	/* Ders. of brightness w.r.t. rotation parameters */
	avx_dyda1 = hadd_pd(avx_dyda1, avx_dyda2);
	avx_dyda1 = hpermute_add_pd(avx_dyda1);
	avx_dyda1 = _mm512_mul_pd(avx_dyda1, avx_Scale);
	_mm512_store_pd(g, avx_dyda1);
	dyda[ncoef0 - 3 + 1 - 1] = g[0];
	dyda[ncoef0 - 3 + 2 - 1] = g[1];
	avx_dyda3 = hadd_pd(avx_dyda3, avx_dyda3);
	avx_dyda3 = hpermute_add_pd(avx_dyda3);
	avx_dyda3 = _mm512_mul_pd(avx_dyda3, avx_Scale);
	_mm512_store_pd(g, avx_dyda3);
	dyda[ncoef0 - 3 + 3 - 1] = g[0];
	/* Ders. of br. w.r.t. cl, cls */
	avx_d = hadd_pd(avx_d, avx_d1);
	avx_d = hpermute_add_pd(avx_d);
	avx_d = _mm512_mul_pd(avx_d, avx_Scale);
	avx_d = _mm512_mul_pd(avx_d, avx_cl1);
	_mm512_store_pd(g, avx_d);
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
