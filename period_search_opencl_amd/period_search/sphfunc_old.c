/* Spherical harmonics functions (unnormalized) for Laplace series
   converted from Mikko's fortran code

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "declarations.hpp"

void sphfunc_old(int ndir, double at[], double af[])
{
    int i, j, m, l, n, k, ibot;

    double aleg[MAX_LM + 1][MAX_LM + 1][MAX_LM + 1];

    for (i = 1; i <= ndir; i++)
    {
        Ts[i][0] = 1;
        Tc[i][0] = 1;
        Ts[i][1] = sin(at[i]);
        Tc[i][1] = cos(at[i]);
        for (j = 2; j <= Lmax; j++)
        {
            Ts[i][j] = Ts[i][1] * Ts[i][j - 1];
            Tc[i][j] = Tc[i][1] * Tc[i][j - 1];
        }
        Fs[i][0] = 0;
        Fc[i][0] = 1;
        for (j = 1; j <= Mmax; j++)
        {
            Fs[i][j] = sin(j*af[i]);
            Fc[i][j] = cos(j*af[i]);
        }
    }

    for (m = 0; m <= Lmax; m++)
        for (l = 0; l <= Lmax; l++)
            for (n = 0; n <= Lmax; n++)
                aleg[n][l][m] = 0;

    aleg[0][0][0] = 1;
    aleg[1][1][0] = 1;

    for (l = 1; l <= Lmax; l++)
        aleg[0][l][l] = aleg[0][l - 1][l - 1] * (2 * l - 1);

    for (m = 0; m <= Mmax; m++)
        for (l = m + 1; l <= Lmax; l++)
        {
            if ((2 * ((l - m) / 2)) == (l - m))
            {
                aleg[0][l][m] = -(l + m - 1) * aleg[0][l - 2][m] / (1 * (l - m));
                ibot = 2;
            }
            else
                ibot = 1;

            if (l != 1)
                for (n = ibot; n <= l - m; n = n + 2)
                    aleg[n][l][m] = ((2 * l - 1) * aleg[n - 1][l - 1][m] - (l + m - 1) * aleg[n][l - 2][m]) / (1 * (l - m));
        }

    for (i = 1; i <= ndir; i++)
    {
        k = 0;
        for (m = 0; m <= Mmax; m++)
            for (l = m; l <= Lmax; l++)
            {
                Pleg[i][l][m] = 0;
                if ((2 * ((l - m) / 2)) == (l - m))
                    ibot = 0;
                else
                    ibot = 1;

                for (n = ibot; n <= l - m; n = n + 2)
                    Pleg[i][l][m] = Pleg[i][l][m] + aleg[n][l][m] * Tc[i][n] * Ts[i][m];
                k++;
                Dsph[i][k] = Fc[i][m] * Pleg[i][l][m];
                if (m != 0)
                {
                    k++;
                    Dsph[i][k] = Fs[i][m] * Pleg[i][l][m];
                }
            }
    }

    // For Unit tests reference only
    /*printf("\nPleg[%d][%d][%d]:\n", i - 1, l - 1, m - 1);
    for (int r = 0; r <= m - 1; r++)
    {
        for (int q = 0; q <= l - 1; q++)
        {
            printf("\n_pleg_j%d_k%d[] = {", q, r);
            for (int p = 0; p <= i - 1; p++)
            {

                printf("%.30f, ", Pleg[p][q][r]);
                if (i % (i-1) == 0)
                    printf("\n");
            }

            printf("};\n");
        }
    }*/
}

