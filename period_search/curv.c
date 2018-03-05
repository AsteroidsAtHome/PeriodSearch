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
          g[MAX_N_FAC];
   
   for (i = 1; i <= Numfac; i++)
   {
      g[i] = 0;
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
            g[i] = g[i] + Pleg[i][l][m] * fsum;
          }
      g[i] = exp(g[i]);
      Area[i] = Darea[i] * g[i];
      for (k = 1; k <= n; k++)
         Dg[i][k] = g[i] * Dsph[i][k];
   }

}
