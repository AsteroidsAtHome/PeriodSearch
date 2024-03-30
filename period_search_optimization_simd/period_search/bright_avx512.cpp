/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006 - Josef Durec
   25.3.2024 - Pavel Rosicky
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <immintrin.h>
#include "CalcStrategyAvx512.hpp"

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif
inline static __m512d blendv_pd(__m512d a, __m512d b, __m512d c) {
	__m512i result = _mm512_ternarylogic_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(b), _mm512_srai_epi64(_mm512_castpd_si512(c), 63), 0xd8);

	return _mm512_castsi512_pd(result);
}

// -mavx512dq
#if defined(__GNUC__)
__attribute__((target("avx512dq,avx512f")))
#endif
inline static __m512d cmp_pd(__m512d a, __m512d b) {
	__m512i result = _mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_GT_OS));

	return _mm512_castsi512_pd(result);
}

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif
inline static int movemask_pd(__m512d a) {
	return (int)_mm512_cmpneq_epi64_mask(_mm512_setzero_si512(), _mm512_and_si512(_mm512_set1_epi64(0x8000000000000000ULL), _mm512_castpd_si512(a)));
}

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif
inline static double reduce_pd(__m512d a) {
	__m256d b = _mm256_add_pd(_mm512_castpd512_pd256(a), _mm512_extractf64x4_pd(a, 1));
	__m128d d = _mm_add_pd(_mm256_castpd256_pd128(b), _mm256_extractf128_pd(b, 1));
	double* f = (double*)&d;
	return _mm_cvtsd_f64(d) + f[1];
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

#if defined(__GNUC__)
__attribute__((target("avx512dq,avx512f")))
#endif
void CalcStrategyAvx512::bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double& br)
{
	int i, j, k;
	incl_count = 0;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);				/* Lambert */
	cls = cg[ncoef];						/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php);				/* computes also Scale */

	matrix(cg[ncoef0], t, tmat, dtm);

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
	__m512d avx_cl = _mm512_set1_pd(cl);
	__m512d avx_cls = _mm512_set1_pd(cls);
	__m512d avx_11 = _mm512_set1_pd(1.0);
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
		avx_lmu = _mm512_fmadd_pd(avx_e3, avx_Nor3, avx_lmu);
		avx_lmu0 = _mm512_mul_pd(avx_e01, avx_Nor1);
		avx_lmu0 = _mm512_fmadd_pd(avx_e02, avx_Nor2, avx_lmu0);
		avx_lmu0 = _mm512_fmadd_pd(avx_e03, avx_Nor3, avx_lmu0);

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
	Dg_row[incl_count] = Dg_row[0];
	br = reduce_pd(res_br);

	/* Derivatives of brightness w.r.t. g-coeffs */
	int ncoef03 = ncoef0 - 3, dgi = 0, cyklus1 = (ncoef03 / 16) * 16;

	for (i = 0; i < cyklus1; i += 16) //2 * 8 doubles
	{
		__m512d tmp1 = _mm512_setzero_pd();
		__m512d tmp2 = _mm512_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			__m512d* Dgrow, * Dgrow1, pdbr, pdbr1;

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
			__m512d* Dgrow, * Dgrow1, pdbr, pdbr1;

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
	dyda[ncoef0 - 3 + 1 - 1] = reduce_pd(avx_dyda1) * Scale;
	dyda[ncoef0 - 3 + 2 - 1] = reduce_pd(avx_dyda2) * Scale;
	dyda[ncoef0 - 3 + 3 - 1] = reduce_pd(avx_dyda3) * Scale;

	/* Ders. of br. w.r.t. cl, cls */
	dyda[ncoef - 1 - 1] = reduce_pd(avx_d) * Scale * cl;
	dyda[ncoef - 1] = reduce_pd(avx_d1) * Scale;

	/* Ders. of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		dyda[ncoef0 + i - 1] = br * dphp[i];

	/* Scaled brightness */
	br *= Scale;
}
