/* Spherical harmonics functions (unnormalized) for Laplace series
   converted from Mikko's fortran code

   8.11.2006
*/

#include "stdafx.h"
#include <math.h>
#include "globals.h"
#include "declarations.h"

void sphfunc(int ndir, double at[], double af[])
{
   int i, j, m, l, n, k, ibot;

   double aleg[MAX_LM+1][MAX_LM+1][MAX_LM+1];

   for (i = 1; i <= ndir; i++)
   {
     t_s[i][0] = 1;
     t_c[i][0] = 1;
     t_s[i][1] = sin(at[i]);
     t_c[i][1] = cos(at[i]);
     for (j = 2; j <= l_max; j++)
     {
        t_s[i][j] = t_s[i][1] * t_s[i][j-1];
        t_c[i][j] = t_c[i][1] * t_c[i][j-1];
     }
     f_s[i][0] = 0;
     f_c[i][0] = 1;
     for (j = 1; j <= m_max; j++)
     {
        f_s[i][j] = sin(j*af[i]);
        f_c[i][j] = cos(j*af[i]);
     }
   }

   for (m = 0; m <= l_max; m++)
      for (l = 0; l <= l_max; l++)
         for (n = 0; n <= l_max; n++)
            aleg[n][l][m]=0;

   aleg[0][0][0] = 1;
   aleg[1][1][0] = 1;

   for (l = 1; l <= l_max; l++)
      aleg[0][l][l]=aleg[0][l-1][l-1]*(2*l-1);

   for (m = 0; m <= m_max; m++)
      for (l = m+1; l <= l_max; l++)
      {
         if ((2 * ((l - m) / 2)) == (l - m))
	 {
            aleg[0][l][m] = -(l+m-1) * aleg[0][l-2][m] / (1*(l-m));
            ibot = 2;
         }
         else
            ibot=1;

         if (l != 1)
	    for (n = ibot; n <= l-m; n = n+2)
               aleg[n][l][m] = ((2*l-1) * aleg[n-1][l-1][m] - (l+m-1) * aleg[n][l-2][m]) / (1*(l-m));
       }

   for (i = 1; i <= ndir; i++)
   {
      k=0;
      for (m = 0; m <= m_max; m++)
         for (l = m; l <= l_max; l++)
	 {
            pleg[i][l][m] = 0;
            if ((2 * ((l - m) / 2)) == (l - m))
               ibot=0;
            else
               ibot=1;

            for (n = ibot; n <= l-m; n = n+2)
               pleg[i][l][m] = pleg[i][l][m] + aleg[n][l][m] * t_c[i][n] * t_s[i][m];
            k++;
            d_sphere[i][k] = f_c[i][m] * pleg[i][l][m];
            if (m != 0)
	    {
               k++;
               d_sphere[i][k] = f_s[i][m] * pleg[i][l][m];
            }
         }
   }
}

