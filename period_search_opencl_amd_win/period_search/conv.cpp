/* Convexity regularization function

   8.11.2006
*/

#include <array>
#include "globals.h"
#include "declarations.h"

using std::array;

double conv(const int nc, double dres[], const int ma, array<array<double, MAX_N_PAR>, MAX_N_FAC> &dg)
{
    int j;

    double res = 0;
    for (j = 1; j <= ma; j++)
        dres[j] = 0;
    for (int i = 0; i < Numfac; i++)
    {
        res += Area[i] * Nor[nc - 1][i];
        for (j = 0; j < Ncoef; j++)
            dres[j] += Darea[i] * dg[i][j] * Nor[nc - 1][i];
    }

    return(res);
}
