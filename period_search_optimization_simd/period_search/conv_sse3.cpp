/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
//#ifdef NO_SSE3
//#include <emmintrin.h>
//#else
#include <pmmintrin.h>
//#endif
#include "CalcStrategySse3.hpp"

#if defined(__GNUC__)
__attribute__((target("sse3")))
#endif

void CalcStrategySse3::conv(int nc, double dres[], int ma, double &result)
{
	int i, j;
	//double res;

	result = 0;
	for (j = 1; j <= ma; j++)
	{
		dres[j] = 0;
	}

	for (i = 0; i < Numfac; i++)
	{
		result += Area[i] * Nor[nc - 1][i];
		__m128d avx_Darea = _mm_set1_pd(Darea[i]);
		__m128d avx_Nor = _mm_set1_pd(Nor[nc - 1][i]);
		double* Dg_row = Dg[i];
		for (j = 0; j < Ncoef; j += 2)
		{
			__m128d avx_dres = _mm_load_pd(&dres[j]);
			__m128d avx_Dg = _mm_load_pd(&Dg_row[j]);

			avx_dres = _mm_add_pd(avx_dres, _mm_mul_pd(_mm_mul_pd(avx_Darea, avx_Dg), avx_Nor));

			_mm_store_pd(&dres[j], avx_dres);
		}
	}

	//return(res);
}
