/* Curvature function (and hence facet area) from Laplace series 

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"

void curv(double cg[])
{
   int i, m, n, l, k;
   
   double fsum,
          g;
   
   for (i = 1; i <= Numfac; i++)
   {
      g = 0;
      n = 0;
      for (m = 0; m <= Mmax; m++)
         for (l = m; l <= Lmax; l++)
	 {
            n++;
            fsum = cg[n] * Fc[i][m];
            if (m != 0) 
            {
	       n++;
               fsum = fsum + cg[n] * Fs[i][m];
            }
            g = g + Pleg[i][l][m] * fsum;
          }
      g = exp(g);
      Area[i] = Darea[i] * g;
      for (k = 1; k <= n; k++)
         Dg[i][k] = g * Dsph[i][k];

   }
}
