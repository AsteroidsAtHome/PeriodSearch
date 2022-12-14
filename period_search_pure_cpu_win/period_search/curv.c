/* Curvature function (and hence facet area) from Laplace series 

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"

void curv(double cg[])
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
        Area[i - 1] = Darea[i - 1] * g;

        for (k = 1; k <= n; k++)
        {
            Dg[i - 1][k - 1] = g * Dsph[i][k];
        }
        if (k <= n) Dg[i - 1][k - 1] = g * Dsph[i][k]; //last odd value
        if (k + 1 <= n) Dg[i - 1][k - 1 + 1] = g * Dsph[i][k + 1]; //last odd value
        if (k + 2 <= n) Dg[i - 1][k - 1 + 2] = g * Dsph[i][k + 2]; //last odd value

         //Dg[i][k] = g * Dsph[i][k];

   }
}
