/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"
#include <immintrin.h>
#include "CalcStrategyFma.hpp"

#if defined(__GNUC__)
__attribute__((target("avx,fma")))
#endif

void CalcStrategyFma::conv(int nc, double dres[], int ma, double &result)
{
    int i, j;

    result = 0;
    for (j = 1; j <= ma; j++)
        dres[j] = 0;

    for (i = 0; i < Numfac; i++)
    {
        result += Area[i] * Nor[nc - 1][i];
        __m256d avx_Darea = _mm256_set1_pd(Darea[i]);
        __m256d avx_Nor = _mm256_set1_pd(Nor[nc - 1][i]);
        double *Dg_row = Dg[i];
        for (j = 0; j < Ncoef; j += 4)
        {
            __m256d avx_dres = _mm256_load_pd(&dres[j]);
            __m256d avx_Dg = _mm256_load_pd(&Dg_row[j]);

            avx_dres = _mm256_fmadd_pd(_mm256_mul_pd(avx_Darea, avx_Dg), avx_Nor, avx_dres);

            _mm256_store_pd(&dres[j], avx_dres);
        }
    }
}
