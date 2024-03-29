/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include "CalcStrategyAsimd.hpp"
#include "arrayHelpers.hpp"

#if defined(__GNUC__)
__attribute__((__target__("arch=armv8-a+simd")))
#endif
void CalcStrategyAsimd::curv(double cg[])
{
   int i, m, l, k;

   for (i = 1; i <= Numfac; i++)
   {
      double g = 0;
      int n = 0;
	  //m=0
         for (l = 0; l <= Lmax; l++)
		 {
            double fsum;
			n++;
            fsum = cg[n] * Fc[i][0];
            g = g + Pleg[i][l][0] * fsum;
          }
      //
	  for (m = 1; m <= Mmax; m++)
         for (l = m; l <= Lmax; l++)
		 {
            double fsum;
			n++;
            fsum = cg[n] * Fc[i][m];
	        n++;
            fsum = fsum + cg[n] * Fs[i][m];
            g = g + Pleg[i][l][m] * fsum;
          }
      g = exp(g);
      Area[i-1] = Darea[i-1] * g;

      float64x2_t avx_g = vdupq_n_f64(g);
      for (k = 1; k < n; k += 2) {
        float64x2_t avx_pom = vld1q_f64(&Dsph[i][k]);
        avx_pom = vmulq_f64(avx_pom, avx_g);
        vst1q_f64(&Dg[i-1][k-1], avx_pom);
      }
      if (k==n) Dg[i-1][k-1] = g * Dsph[i][k]; //last odd value
   }
}
