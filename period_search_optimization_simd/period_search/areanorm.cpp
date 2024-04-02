/* Areas and normals of the triangulated Gaussian image sphere

   8.11.2006
*/

#include <cmath>
#include "globals.h"
#include "declarations.h"
#include "arrayHelpers.hpp"

void areanorm(double t[], double f[], int ndir, int nfac, int **ifp, double at[], double af[])
{
    int i, j;

    double  st, clen2, clen;

	double c[4]{}, vx[4]{}, vy[4]{}, vz[4]{}, * x, * y, * z;

    x = vector_double(ndir);
    y = vector_double(ndir);
    z = vector_double(ndir);

    for (i = 1; i <= ndir; i++)
    {
        st = sin(t[i]);
        x[i] = st * cos(f[i]);
        y[i] = st * sin(f[i]);
        z[i] = cos(t[i]);
		//printf("x[%3d] % 0.6f\ty[%3d] % 0.6f\tz[%3d] % 0.6f\n", i, x[i], i, y[i], i, z[i]);
    }

    for (i = 1; i <= nfac; i++)
    {
        /* vectors of triangle edges */
        for (j = 2; j <= 3; j++)
        {
            vx[j] = x[ifp[i][j]] - x[ifp[i][1]];
            vy[j] = y[ifp[i][j]] - y[ifp[i][1]];
            vz[j] = z[ifp[i][j]] - z[ifp[i][1]];
        }

        /* The cross product for each triangle */
        c[1] = vy[2] * vz[3] - vy[3] * vz[2];
        c[2] = vz[2] * vx[3] - vz[3] * vx[2];
        c[3] = vx[2] * vy[3] - vx[3] * vy[2];

        /* Areas (on the unit sphere) and normals */
        clen2 = c[1] * c[1] + c[2] * c[2] + c[3] * c[3];
        clen = sqrt(clen2);
		//printf("[%3d] % 0.6f\n", i, clen);

        /* normal */
        Nor[0][i - 1] = c[1] / clen;
        Nor[1][i - 1] = c[2] / clen;
        Nor[2][i - 1] = c[3] / clen;

        /* direction angles of normal */
        at[i] = acos(Nor[2][i - 1]);
        af[i] = atan2(Nor[1][i - 1], Nor[0][i - 1]);

        /* triangle area */
        Darea[i - 1] = 0.5 * clen;
		//printf("[%3d] % 0.6f\n", i-1, Darea[i - 1]);
    }

    // NOTE: For unit tests reference only
    //printArray(at, i - 1, "at");
    //printArray(af, i - 1, "af");
	//printArray(Darea, nfac, "Darea");

    deallocate_vector((void *)x);
    deallocate_vector((void *)y);
    deallocate_vector((void *)z);
}