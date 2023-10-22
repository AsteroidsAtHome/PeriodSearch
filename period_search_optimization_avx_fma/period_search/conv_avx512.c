/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"
#include <immintrin.h>

double conv_avx512(int nc, double dres[], int ma)
{
    int i, j;
    double res;
    //__m512d ymm0 = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

    res = 0;
    for (j = 1; j <= ma; j++)
        dres[j] = 0;

    for (i = 0; i < Numfac; i++)
    {
        res += Area[i] * Nor[nc - 1][i];
        __m512d avx_Darea = _mm512_set1_pd(Darea[i]);
        __m512d avx_Nor = _mm512_set1_pd(Nor[nc - 1][i]);
        double *Dg_row = Dg[i];
        for (j = 0; j < Ncoef; j += 8)
        {
            __m512d avx_dres = _mm512_load_pd(&dres[j]);
            __m512d avx_Dg = _mm512_load_pd(&Dg_row[j]);

            avx_dres = _mm512_fmadd_pd(_mm512_mul_pd(avx_Darea, avx_Dg), avx_Nor, avx_dres);
            //avx_dres = _mm512_add_pd(avx_dres, _mm512_mul_pd(_mm512_mul_pd(avx_Darea, avx_Dg), avx_Nor));

            _mm512_store_pd(&dres[j], avx_dres);
        }
    }

    return(res);
}
