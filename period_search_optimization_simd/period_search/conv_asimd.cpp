/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategyAsimd.hpp"

#if defined __GNUG__ && !defined __clang__
__attribute__((__target__("arch=armv8-a+simd")))
#elif defined __GNUG__ && __clang__
// NOTE: The following generates warning: unsupported architecture 'armv8-a+simd' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// __attribute__((target("arch=armv8-a+simd")))
#endif

void CalcStrategyAsimd::conv(int nc, double dres[], int ma, double &result)
{
    int i, j;
    result = 0;

    for (j = 1; j <= ma; j++)
        dres[j] = 0;

    for (i = 0; i < Numfac; i++) {
        result += Area[i] * Nor[nc - 1][i];
        double *Dg_row = Dg[i];
        float64x2_t avx_Darea = vdupq_n_f64(Darea[i]);
        float64x2_t avx_Nor = vdupq_n_f64(Nor[nc - 1][i]);
        for (j = 0; j < Ncoef; j += 2) {
            float64x2_t avx_dres = vld1q_f64(&dres[j]);
            float64x2_t avx_Dg = vld1q_f64(&Dg_row[j]);

            avx_dres = vfmaq_f64(avx_dres, vmulq_f64(avx_Darea, avx_Dg), avx_Nor);
            vst1q_f64(&dres[j], avx_dres);
        }
    }
}
