/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/


#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS


#include <cmath>
#include <chrono>
#include "globals.h"


using namespace std;
using namespace std::chrono;




void curv(const std::vector<double> &cg)
{
    for (auto i = 1; i <= Numfac; i++)
    {
        double g = 0;
        auto n = 0;
        //m=0
        for (auto l = 0; l <= Lmax; l++)
        {
            n++;
            auto fsum = cg[n] * Fc[i][0];
            g = g + Pleg[i][l][0] * fsum;
        }
        //
        for (auto m = 1; m <= Mmax; m++)
            for (auto l = m; l <= Lmax; l++)
            {
                n++;
                auto fsum = cg[n] * Fc[i][m];
                n++;
                fsum = fsum + cg[n] * Fs[i][m];
                g = g + Pleg[i][l][m] * fsum;
            }
        g = exp(g);
        Area[i - 1] = Darea[i - 1] * g;

        for (auto k = 1; k <= n; k++)
        {
            Dg[i - 1][k - 1] = g * Dsph[i][k];
        }
   }
}
