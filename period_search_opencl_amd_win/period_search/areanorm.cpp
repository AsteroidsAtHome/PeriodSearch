/* Areas and normals of the triangulated Gaussian image sphere

   8.11.2006
*/

#include <cmath>
#include "globals.h"
#include "declarations.hpp"
#include "VectorT.hpp"
#include "AnglesOfNormals.hpp"

using namespace std;

void areanorm(double t[], double f[], int ndir, int **ifp, struct AnglesOfNormals &angles_n)
{
    int i, j;
    double  sin_t, clen;
    double vx[4], vy[4], vz[4];
    math::VectorT<double> vector_c(3);
    vector<double> x(ndir + 1), y(ndir + 1), z(ndir + 1);

    for (i = 1; i <= ndir; i++)
    {
        sin_t = sin(t[i]);
        x.at(i) = sin_t * cos(f[i]);
        y.at(i) = sin_t * sin(f[i]);
        z.at(i) = cos(t[i]);
    }

    for (i = 1; i <= angles_n.number_facets; i++)
    {
        /* vectors of triangle edges */
        for (j = 2; j <= 3; j++)
        {
            vx[j] = x.at(ifp[i][j]) - x.at(ifp[i][1]);
            vy[j] = y.at(ifp[i][j]) - y.at(ifp[i][1]);
            vz[j] = z.at(ifp[i][j]) - z.at(ifp[i][1]);
        }

        /* The cross product for each triangle */
        vector_c.at(0) = vy[2] * vz[3] - vy[3] * vz[2];
        vector_c.at(1) = vz[2] * vx[3] - vz[3] * vx[2];
        vector_c.at(2) = vx[2] * vy[3] - vx[3] * vy[2];

        /* Areas (on the unit sphere) and normals */
        clen = vector_c.magnitude();

        /* normal */
        Nor[0][i - 1] = vector_c.at(0) / clen;
        Nor[1][i - 1] = vector_c.at(1) / clen;
        Nor[2][i - 1] = vector_c.at(2) / clen;

        /* direction angles of normal */
        angles_n.theta_angle.at(i) = acos(Nor[2][i - 1]);
        angles_n.phi_angle.at(i) = atan2(Nor[1][i - 1], Nor[0][i - 1]);
        /*at[i] = acos(Nor[2][i - 1]);
        af[i] = atan2(Nor[1][i - 1], Nor[0][i - 1]);*/

        /* triangle area */
        Darea[i - 1] = 0.5 * clen;
    }
}

