/* Areas and normals of the triangulated Gaussian image sphere

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <cmath>
#include "globals.h"
#include "declarations.hpp"
#include "AnglesOfNormals.hpp"
#include "DotProductT.hpp"

using namespace std;

void areanorm(double *theta, double *phi, const int &ndir, int **ifp, struct AnglesOfNormals &normals)
{
    double vx[4], vy[4], vz[4];
    cl_double3 vectorC;
    vector<double> x(ndir + 1), y(ndir + 1), z(ndir + 1);

    for (auto i = 1; i <= ndir; i++)
    {
        auto const sinTheta = sin(theta[i]);
        x[i] = sinTheta * cos(phi[i]);
        y[i] = sinTheta * sin(phi[i]);
        z[i] = cos(theta[i]);
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
        vectorC.x = vy[2] * vz[3] - vy[3] * vz[2];
        vectorC.y = vz[2] * vx[3] - vz[3] * vx[2];
        vectorC.z = vx[2] * vy[3] - vx[3] * vy[2];

        /* Areas (on the unit sphere) and normals */
        auto const cLen = sqrt(math::DotProduct3(vectorC, vectorC));

        /* normal */
        Nor[0][i - 1] = vectorC.x / cLen;
        Nor[1][i - 1] = vectorC.y / cLen;
        Nor[2][i - 1] = vectorC.z / cLen;

        /* direction angles of normal */
        normals.theta[i] = acos(Nor[2][i - 1]);
        normals.phi[i] = atan2(Nor[1][i - 1], Nor[0][i - 1]);

        /* triangle area */
        Darea[i - 1] = 0.5 * cLen;
    }
}

