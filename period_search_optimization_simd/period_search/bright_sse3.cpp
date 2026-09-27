#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <pmmintrin.h> // SSE3
#include "CalcStrategySse3.hpp"

#define INNER_CALC \
		 res_br=_mm_add_pd(res_br,avx_pbr);	\
	        __m128d avx_sum1,avx_sum10,avx_sum2,avx_sum20,avx_sum3,avx_sum30; \
			\
				avx_sum1=_mm_mul_pd(avx_Nor1,avx_de11); \
				avx_sum1=_mm_add_pd(avx_sum1,_mm_mul_pd(avx_Nor2,avx_de21)); \
				avx_sum1=_mm_add_pd(avx_sum1,_mm_mul_pd(avx_Nor3,avx_de31)); \
\
				avx_sum10=_mm_mul_pd(avx_Nor1,avx_de011); \
				avx_sum10=_mm_add_pd(avx_sum10,_mm_mul_pd(avx_Nor2,avx_de021)); \
				avx_sum10=_mm_add_pd(avx_sum10,_mm_mul_pd(avx_Nor3,avx_de031)); \
				\
				avx_sum2=_mm_mul_pd(avx_Nor1,avx_de12); \
				avx_sum2=_mm_add_pd(avx_sum2,_mm_mul_pd(avx_Nor2,avx_de22)); \
				avx_sum2=_mm_add_pd(avx_sum2,_mm_mul_pd(avx_Nor3,avx_de32)); \
				\
				avx_sum20=_mm_mul_pd(avx_Nor1,avx_de012); \
				avx_sum20=_mm_add_pd(avx_sum20,_mm_mul_pd(avx_Nor2,avx_de022)); \
				avx_sum20=_mm_add_pd(avx_sum20,_mm_mul_pd(avx_Nor3,avx_de032)); \
				\
				avx_sum3=_mm_mul_pd(avx_Nor1,avx_de13); \
				avx_sum3=_mm_add_pd(avx_sum3,_mm_mul_pd(avx_Nor2,avx_de23)); \
				avx_sum3=_mm_add_pd(avx_sum3,_mm_mul_pd(avx_Nor3,avx_de33)); \
				\
				avx_sum30=_mm_mul_pd(avx_Nor1,avx_de013); \
				avx_sum30=_mm_add_pd(avx_sum30,_mm_mul_pd(avx_Nor2,avx_de023)); \
				avx_sum30=_mm_add_pd(avx_sum30,_mm_mul_pd(avx_Nor3,avx_de033)); \
				\
			avx_sum1=_mm_mul_pd(avx_sum1,avx_dsmu); \
			avx_sum2=_mm_mul_pd(avx_sum2,avx_dsmu); \
			avx_sum3=_mm_mul_pd(avx_sum3,avx_dsmu); \
			avx_sum10=_mm_mul_pd(avx_sum10,avx_dsmu0); \
			avx_sum20=_mm_mul_pd(avx_sum20,avx_dsmu0); \
			avx_sum30=_mm_mul_pd(avx_sum30,avx_dsmu0); \
			\
            avx_dyda1=_mm_add_pd(avx_dyda1,_mm_mul_pd(avx_Area,_mm_add_pd(avx_sum1,avx_sum10))); \
            avx_dyda2=_mm_add_pd(avx_dyda2,_mm_mul_pd(avx_Area,_mm_add_pd(avx_sum2,avx_sum20))); \
            avx_dyda3=_mm_add_pd(avx_dyda3,_mm_mul_pd(avx_Area,_mm_add_pd(avx_sum3,avx_sum30))); \
			\
			avx_d=_mm_add_pd(avx_d,_mm_mul_pd(_mm_mul_pd(avx_lmu,avx_lmu0),avx_Area)); \
			avx_d1=_mm_add_pd(avx_d1,_mm_div_pd(_mm_mul_pd(_mm_mul_pd(avx_Area,avx_lmu),avx_lmu0),_mm_add_pd(avx_lmu,avx_lmu0)));
// end of inner_calc
#define INNER_CALC_DSMU \
	  avx_Area=_mm_load_pd(&gl.Area[i]); \
	  avx_dnom=_mm_add_pd(avx_lmu,avx_lmu0); \
	  avx_s=_mm_mul_pd(_mm_mul_pd(avx_lmu,avx_lmu0),_mm_add_pd(avx_cl,_mm_div_pd(avx_cls,avx_dnom))); \
	  avx_pdbr=_mm_mul_pd(_mm_load_pd(&gl.Darea[i]),avx_s); \
	  avx_pbr=_mm_mul_pd(avx_Area,avx_s); \
	  avx_powdnom=_mm_div_pd(avx_lmu0,avx_dnom); \
	  avx_powdnom=_mm_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu=_mm_add_pd(_mm_mul_pd(avx_cls,avx_powdnom),_mm_mul_pd(avx_cl,avx_lmu0)); \
	  avx_powdnom=_mm_div_pd(avx_lmu,avx_dnom); \
	  avx_powdnom=_mm_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu0=_mm_add_pd(_mm_mul_pd(avx_cls,avx_powdnom),_mm_mul_pd(avx_cl,avx_lmu));
// end of inner_calc_dsmu


#if defined(__GNUC__)
__attribute__((target("sse3")))
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
void CalcStrategySse3::bright(const double t, std::vector<double>& cg, const int ncoef, globals& gl)
{
	int i, j, k; // ncoef0,
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
	//SSE3
	__m128d avx_e1 = _mm_loaddup_pd(&e[1]);
	__m128d avx_e2 = _mm_loaddup_pd(&e[2]);
	__m128d avx_e3 = _mm_loaddup_pd(&e[3]);
	__m128d avx_e01 = _mm_loaddup_pd(&e0[1]);
	__m128d avx_e02 = _mm_loaddup_pd(&e0[2]);
	__m128d avx_e03 = _mm_loaddup_pd(&e0[3]);
	__m128d avx_de11 = _mm_loaddup_pd(&de[1][1]);
	__m128d avx_de12 = _mm_loaddup_pd(&de[1][2]);
	__m128d avx_de13 = _mm_loaddup_pd(&de[1][3]);
	__m128d avx_de21 = _mm_loaddup_pd(&de[2][1]);
	__m128d avx_de22 = _mm_loaddup_pd(&de[2][2]);
	__m128d avx_de23 = _mm_loaddup_pd(&de[2][3]);
	__m128d avx_de31 = _mm_loaddup_pd(&de[3][1]);
	__m128d avx_de32 = _mm_loaddup_pd(&de[3][2]);
	__m128d avx_de33 = _mm_loaddup_pd(&de[3][3]);
	__m128d avx_de011 = _mm_loaddup_pd(&de0[1][1]);
	__m128d avx_de012 = _mm_loaddup_pd(&de0[1][2]);
	__m128d avx_de013 = _mm_loaddup_pd(&de0[1][3]);
	__m128d avx_de021 = _mm_loaddup_pd(&de0[2][1]);
	__m128d avx_de022 = _mm_loaddup_pd(&de0[2][2]);
	__m128d avx_de023 = _mm_loaddup_pd(&de0[2][3]);
	__m128d avx_de031 = _mm_loaddup_pd(&de0[3][1]);
	__m128d avx_de032 = _mm_loaddup_pd(&de0[3][2]);
	__m128d avx_de033 = _mm_loaddup_pd(&de0[3][3]);
	__m128d avx_Scale = _mm_loaddup_pd(&Scale);

	__m128d avx_tiny = _mm_set1_pd(TINY);
	__m128d avx_cl = _mm_set1_pd(cl), avx_cl1 = _mm_set_pd(1, cl), avx_cls = _mm_set1_pd(cls), avx_11 = _mm_set1_pd(1.0);
	__m128d res_br = _mm_setzero_pd();
	__m128d avx_dyda1 = _mm_setzero_pd();
	__m128d avx_dyda2 = _mm_setzero_pd();
	__m128d avx_dyda3 = _mm_setzero_pd();
	__m128d avx_d = _mm_setzero_pd();
	__m128d avx_d1 = _mm_setzero_pd();

	for (i = 0; i < Numfac; i += 2)
	{
		__m128d avx_lmu, avx_lmu0, cmpe, cmpe0, cmp;
		__m128d avx_Nor1 = _mm_load_pd(&gl.Nor[0][i]);
		__m128d avx_Nor2 = _mm_load_pd(&gl.Nor[1][i]);
		__m128d avx_Nor3 = _mm_load_pd(&gl.Nor[2][i]);
		__m128d avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr;
		__m128d avx_Area;

		avx_lmu = _mm_mul_pd(avx_e1, avx_Nor1);
		avx_lmu = _mm_add_pd(avx_lmu, _mm_mul_pd(avx_e2, avx_Nor2));
		avx_lmu = _mm_add_pd(avx_lmu, _mm_mul_pd(avx_e3, avx_Nor3));
		avx_lmu0 = _mm_mul_pd(avx_e01, avx_Nor1);
		avx_lmu0 = _mm_add_pd(avx_lmu0, _mm_mul_pd(avx_e02, avx_Nor2));
		avx_lmu0 = _mm_add_pd(avx_lmu0, _mm_mul_pd(avx_e03, avx_Nor3));

		cmpe = _mm_cmpgt_pd(avx_lmu, avx_tiny);
		cmpe0 = _mm_cmpgt_pd(avx_lmu0, avx_tiny);
		cmp = _mm_and_pd(cmpe, cmpe0);
		int icmp = _mm_movemask_pd(cmp);

		if (icmp & 1)  //first and second or only first
		{
			INNER_CALC_DSMU
				if (icmp & 2)
				{
					//0
					Dg_row[incl_count] = (__m128d*)& gl.Dg[i];
					dbr[incl_count++] = _mm_movedup_pd(avx_pdbr);

					//1
					Dg_row[incl_count] = (__m128d*)& gl.Dg[i + 1];
					dbr[incl_count++] = _mm_movedup_pd(_mm_shuffle_pd(avx_pdbr, avx_pdbr, 1));
				}
				else
				{
					avx_pbr = _mm_shuffle_pd(avx_pbr, _mm_setzero_pd(), 0);
					avx_dsmu = _mm_shuffle_pd(avx_dsmu, _mm_setzero_pd(), 0);
					avx_dsmu0 = _mm_shuffle_pd(avx_dsmu0, _mm_setzero_pd(), 0);
					avx_lmu = _mm_shuffle_pd(avx_lmu, _mm_setzero_pd(), 0);
					avx_lmu0 = _mm_shuffle_pd(avx_lmu0, avx_11, 0); //abychom nedelili nulou
					//0
					Dg_row[incl_count] = (__m128d*)& gl.Dg[i];
					dbr[incl_count++] = _mm_movedup_pd(avx_pdbr);
				}

			INNER_CALC
		}
		else if (icmp & 2)
		{
			INNER_CALC_DSMU
				avx_pbr = _mm_shuffle_pd(avx_pbr, _mm_setzero_pd(), 1);
			avx_dsmu = _mm_shuffle_pd(_mm_setzero_pd(), avx_dsmu, _MM_SHUFFLE2(1, 0));
			avx_dsmu0 = _mm_shuffle_pd(_mm_setzero_pd(), avx_dsmu0, _MM_SHUFFLE2(1, 0));
			avx_lmu = _mm_shuffle_pd(_mm_setzero_pd(), avx_lmu, _MM_SHUFFLE2(1, 0));
			avx_lmu0 = _mm_shuffle_pd(avx_11, avx_lmu0, _MM_SHUFFLE2(1, 0));
			//1
			Dg_row[incl_count] = (__m128d*)& gl.Dg[i + 1];
			dbr[incl_count++] = _mm_movedup_pd(_mm_shuffle_pd(avx_pdbr, avx_pdbr, 1));

			INNER_CALC
		}
	}

	dbr[incl_count] = _mm_setzero_pd();
	dbr[incl_count + 1] = _mm_setzero_pd();
	dbr[incl_count + 2] = _mm_setzero_pd();
	dbr[incl_count + 3] = _mm_setzero_pd();
	Dg_row[incl_count] = Dg_row[0];
	Dg_row[incl_count + 1] = Dg_row[0];
	Dg_row[incl_count + 2] = Dg_row[0];
	Dg_row[incl_count + 3] = Dg_row[0];

	res_br = _mm_hadd_pd(res_br, res_br);
	gl.ymod = _mm_cvtsd_f64(res_br);

	/* Derivatives of brightness w.r.t. g-coefficients */
	int ncoef03 = ncoef0 - 3, dgi = 0, cyklus1 = (ncoef03 / 10) * 10;

	for (i = 0; i < cyklus1; i += 10) //5 * 2doubles
	{
		__m128d tmp1;
		__m128d tmp2;
		__m128d tmp3;
		__m128d tmp4;
		__m128d tmp5;
		__m128d* Dgrow, * Dgrow1, * Dgrow2, * Dgrow3, pdbr, pdbr1, pdbr2, pdbr3;

		Dgrow = &Dg_row[0][dgi];
		pdbr = dbr[0];
		Dgrow1 = &Dg_row[1][dgi];
		pdbr1 = dbr[1];
		Dgrow2 = &Dg_row[2][dgi];
		pdbr2 = dbr[2];
		Dgrow3 = &Dg_row[3][dgi];
		pdbr3 = dbr[3];

		tmp1 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(pdbr, Dgrow[0]), _mm_mul_pd(pdbr1, Dgrow1[0])), _mm_mul_pd(pdbr2, Dgrow2[0])), _mm_mul_pd(pdbr3, Dgrow3[0]));
		tmp2 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(pdbr, Dgrow[1]), _mm_mul_pd(pdbr1, Dgrow1[1])), _mm_mul_pd(pdbr2, Dgrow2[1])), _mm_mul_pd(pdbr3, Dgrow3[1]));
		tmp3 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(pdbr, Dgrow[2]), _mm_mul_pd(pdbr1, Dgrow1[2])), _mm_mul_pd(pdbr2, Dgrow2[2])), _mm_mul_pd(pdbr3, Dgrow3[2]));
		tmp4 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(pdbr, Dgrow[3]), _mm_mul_pd(pdbr1, Dgrow1[3])), _mm_mul_pd(pdbr2, Dgrow2[3])), _mm_mul_pd(pdbr3, Dgrow3[3]));
		tmp5 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(pdbr, Dgrow[4]), _mm_mul_pd(pdbr1, Dgrow1[4])), _mm_mul_pd(pdbr2, Dgrow2[4])), _mm_mul_pd(pdbr3, Dgrow3[4]));

		for (j = 4; j < incl_count; j += 4)
		{
			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];
			Dgrow2 = &Dg_row[j + 2][dgi];
			pdbr2 = dbr[j + 2];
			Dgrow3 = &Dg_row[j + 3][dgi];
			pdbr3 = dbr[j + 3];

			tmp1 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(tmp1, _mm_mul_pd(pdbr, Dgrow[0])), _mm_mul_pd(pdbr1, Dgrow1[0])), _mm_mul_pd(pdbr2, Dgrow2[0])), _mm_mul_pd(pdbr3, Dgrow3[0]));
			tmp2 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(tmp2, _mm_mul_pd(pdbr, Dgrow[1])), _mm_mul_pd(pdbr1, Dgrow1[1])), _mm_mul_pd(pdbr2, Dgrow2[1])), _mm_mul_pd(pdbr3, Dgrow3[1]));
			tmp3 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(tmp3, _mm_mul_pd(pdbr, Dgrow[2])), _mm_mul_pd(pdbr1, Dgrow1[2])), _mm_mul_pd(pdbr2, Dgrow2[2])), _mm_mul_pd(pdbr3, Dgrow3[2]));
			tmp4 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(tmp4, _mm_mul_pd(pdbr, Dgrow[3])), _mm_mul_pd(pdbr1, Dgrow1[3])), _mm_mul_pd(pdbr2, Dgrow2[3])), _mm_mul_pd(pdbr3, Dgrow3[3]));
			tmp5 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(tmp5, _mm_mul_pd(pdbr, Dgrow[4])), _mm_mul_pd(pdbr1, Dgrow1[4])), _mm_mul_pd(pdbr2, Dgrow2[4])), _mm_mul_pd(pdbr3, Dgrow3[4]));
		}

		dgi += 5;
		tmp1 = _mm_mul_pd(tmp1, avx_Scale);
		_mm_store_pd(&gl.dyda[i], tmp1);
		tmp2 = _mm_mul_pd(tmp2, avx_Scale);
		_mm_store_pd(&gl.dyda[i + 2], tmp2);
		tmp3 = _mm_mul_pd(tmp3, avx_Scale);
		_mm_store_pd(&gl.dyda[i + 4], tmp3);
		tmp4 = _mm_mul_pd(tmp4, avx_Scale);
		_mm_store_pd(&gl.dyda[i + 6], tmp4);
		tmp5 = _mm_mul_pd(tmp5, avx_Scale);
		_mm_store_pd(&gl.dyda[i + 8], tmp5);
	}

	for (; i < ncoef03; i += 4) //2 * 2doubles
	{
		__m128d tmp1;
		__m128d tmp2;
		__m128d* Dgrow, * Dgrow1, * Dgrow2, * Dgrow3, pdbr, pdbr1, pdbr2, pdbr3;

		Dgrow = &Dg_row[0][dgi];
		pdbr = dbr[0];
		Dgrow1 = &Dg_row[1][dgi];
		pdbr1 = dbr[1];
		Dgrow2 = &Dg_row[2][dgi];
		pdbr2 = dbr[2];
		Dgrow3 = &Dg_row[3][dgi];
		pdbr3 = dbr[3];

		tmp1 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(pdbr, Dgrow[0]), _mm_mul_pd(pdbr1, Dgrow1[0])), _mm_mul_pd(pdbr2, Dgrow2[0])), _mm_mul_pd(pdbr3, Dgrow3[0]));
		tmp2 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(pdbr, Dgrow[1]), _mm_mul_pd(pdbr1, Dgrow1[1])), _mm_mul_pd(pdbr2, Dgrow2[1])), _mm_mul_pd(pdbr3, Dgrow3[1]));

		for (j = 4; j < incl_count; j += 4)
		{

			Dgrow = &Dg_row[j][dgi];
			pdbr = dbr[j];
			Dgrow1 = &Dg_row[j + 1][dgi];
			pdbr1 = dbr[j + 1];
			Dgrow2 = &Dg_row[j + 2][dgi];
			pdbr2 = dbr[j + 2];
			Dgrow3 = &Dg_row[j + 3][dgi];
			pdbr3 = dbr[j + 3];

			tmp1 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(tmp1, _mm_mul_pd(pdbr, Dgrow[0])), _mm_mul_pd(pdbr1, Dgrow1[0])), _mm_mul_pd(pdbr2, Dgrow2[0])), _mm_mul_pd(pdbr3, Dgrow3[0]));
			tmp2 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(tmp2, _mm_mul_pd(pdbr, Dgrow[1])), _mm_mul_pd(pdbr1, Dgrow1[1])), _mm_mul_pd(pdbr2, Dgrow2[1])), _mm_mul_pd(pdbr3, Dgrow3[1]));
		}

		dgi += 2;
		tmp1 = _mm_mul_pd(tmp1, avx_Scale);
		_mm_store_pd(&gl.dyda[i], tmp1);
		tmp2 = _mm_mul_pd(tmp2, avx_Scale);
		_mm_store_pd(&gl.dyda[i + 2], tmp2);
	}

	/* Derivatives of brightness w.r.t. rotation parameters */
	avx_dyda1 = _mm_hadd_pd(avx_dyda1, avx_dyda2);
	avx_dyda1 = _mm_mul_pd(avx_dyda1, avx_Scale);
	_mm_storeu_pd(&gl.dyda[ncoef0 - 3 + 1 - 1], avx_dyda1); //unaligned memory because of odd index
	avx_dyda3 = _mm_hadd_pd(avx_dyda3, avx_dyda3);
	avx_dyda3 = _mm_mul_pd(avx_dyda3, avx_Scale);
	gl.dyda[ncoef0 - 3 + 3 - 1] = _mm_cvtsd_f64(avx_dyda3);

	/* Derivatives of br. w.r.t. cl, cls */
	avx_d = _mm_hadd_pd(avx_d, avx_d1);
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
