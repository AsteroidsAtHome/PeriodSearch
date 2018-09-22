/* Areas and normals of the triangulated Gaussian image sphere

   8.11.2006
*/

#include <cmath>
#include "globals.h"
#include "declarations.hpp"
#include "AnglesOfNormals.hpp"
#include "DotProductT.hpp"

using namespace std;
using namespace math;

void areanorm(double t[], double f[], int ndir, int **ifp, struct AnglesOfNormals &normals)
{
    double vx[4], vy[4], vz[4];
    vector<double> vectorC(3);
    vector<double> x(ndir + 1), y(ndir + 1), z(ndir + 1);

    for (auto i = 1; i <= ndir; i++)
    {
        auto const sinTheta = sin(t[i]);
        x[i] = sinTheta * cos(f[i]);
        y[i] = sinTheta * sin(f[i]);
        z[i] = cos(t[i]);
    }

    for (size_t i = 1; i <= normals.numberFacets; i++)
    {
        /* vectors of triangle edges */
        for (size_t j = 2; j <= 3; j++)
        {
            vx[j] = x[ifp[i][j]] - x[ifp[i][1]];
            vy[j] = y[ifp[i][j]] - y[ifp[i][1]];
            vz[j] = z[ifp[i][j]] - z[ifp[i][1]];
        }

        /* The cross product for each triangle */
        vectorC[0] = vy[2] * vz[3] - vy[3] * vz[2];
        vectorC[1] = vz[2] * vx[3] - vz[3] * vx[2];
        vectorC[2] = vx[2] * vy[3] - vx[3] * vy[2];

        /* Areas (on the unit sphere) and normals */
        auto const cLen = sqrt(DotProduct<double>(vectorC, vectorC));

        /* normal */
        Nor[0][i - 1] = vectorC[0] / cLen;
        Nor[1][i - 1] = vectorC[1] / cLen;
        Nor[2][i - 1] = vectorC[2] / cLen;

        /* direction angles of normal */
        normals.theta[i] = acos(Nor[2][i - 1]);
        normals.phi[i] = atan2(Nor[1][i - 1], Nor[0][i - 1]);
        /*at[i] = acos(Nor[2][i - 1]);
        af[i] = atan2(Nor[1][i - 1], Nor[0][i - 1]);*/

        /* triangle area */
        Darea[i - 1] = 0.5 * cLen;
    }
}

