/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include "CalcStrategySve.hpp"
#include "arrayHelpers.hpp"

#if defined(__GNUC__)
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::curv(double cg[])
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

      svfloat64_t avx_g = svdup_n_f64(g);
      svbool_t pg = svptrue_b64();
      for (k = 1; k < n; k += svcntd()) {
        svfloat64_t avx_pom = svld1_f64(pg, &Dsph[i][k]);
        avx_pom = svmul_f64_x(pg, avx_pom, avx_g);
        svst1_f64(svptrue_b64(), &Dg[i-1][k-1], avx_pom);
      }
   }
}
