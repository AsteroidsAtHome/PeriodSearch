/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/
//
//#include <cuda.h>
//#include <math.h>
//#include "globals_CUDA.h"


__kernel void curv(freq_context* CUDA_LCC, double cg[], int brtmpl, int brtmph)
{
    int i, m, n, l, k;

    double fsum,
        g;


    for (i = brtmpl; i <= brtmph; i++)
    {
        g = 0;
        n = 0;
        for (m = 0; m <= CUDA_Mmax; m++)
            for (l = m; l <= CUDA_Lmax; l++)
            {
                n++;
                fsum = cg[n] * CUDA_Fc[i][m];
                if (m != 0)
                {
                    n++;
                    fsum = fsum + cg[n] * CUDA_Fs[i][m];
                }
                g = g + CUDA_Pleg[i][l][m] * fsum;
            }
        g = exp(g);
        (*CUDA_LCC).Area[i] = CUDA_Darea[i] * g;
        for (k = 1; k <= n; k++)
            (*CUDA_LCC).Dg[i + k * (CUDA_Numfac1)] = g * CUDA_Dsph[i][k];

    }
    __syncthreads();
}
