/* rotation matrix and its derivatives
   converted from Mikko's fortran code

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <vector>
#include "globals.h"
#include "constants.h"

void matrix(const double& omg, const double& time, double tmat[][4], double dtm[][4][4])
{
    double dfm[4][4], fmat[4][4];

    int i, j, k;

    /* phase of rotation */
    auto phi = omg * time + Phi_0;
    phi = fmod(phi, 2 * PI); /* may give little different results than Mikko's */
    auto const sinPhi = sin(phi);
    auto const cosPhi = cos(phi);

    /* rotation matrix, Z axis, angle f */
    fmat[1][1] = cosPhi;
    fmat[1][2] = sinPhi;
    fmat[1][3] = 0;
    fmat[2][1] = -sinPhi;
    fmat[2][2] = cosPhi;
    fmat[2][3] = 0;
    fmat[3][1] = 0;
    fmat[3][2] = 0;
    fmat[3][3] = 1;

    /* Ders. w.r.t omg */
    dfm[1][1] = -time * sinPhi;
    dfm[1][2] = time * cosPhi;
    dfm[1][3] = 0;
    dfm[2][1] = -time * cosPhi;
    dfm[2][2] = -time * sinPhi;
    dfm[2][3] = 0;
    dfm[3][1] = 0;
    dfm[3][2] = 0;
    dfm[3][3] = 0;

    /* Construct tmat (complete rotation matrix) and its derivatives */

    for (i = 1; i <= 3; i++)
    {
        //tmat[i].s0 = fmat[i][1] * Blmat[1][1] + fmat[i][2] * Blmat[2][1] + fmat[i][3] * Blmat[3][1];
        //tmat[i].s1 = fmat[i][1] * Blmat[1][2] + fmat[i][2] * Blmat[2][2] + fmat[i][3] * Blmat[3][2];
        //tmat[i].s2 = fmat[i][1] * Blmat[1][3] + fmat[i][2] * Blmat[2][3] + fmat[i][3] * Blmat[3][3];

        for (j = 1; j <= 3; j++)
        {
            tmat[i][j] = 0;
            dtm[1][i][j] = 0;
            dtm[2][i][j] = 0;
            dtm[3][i][j] = 0;

            for (k = 1; k <= 3; k++)
            {
                tmat[i][j] = tmat[i][j] + fmat[i][k] * Blmat[k][j];
                dtm[1][i][j] = dtm[1][i][j] + fmat[i][k] * Dblm[1][k][j];
                dtm[2][i][j] = dtm[2][i][j] + fmat[i][k] * Dblm[2][k][j];
                dtm[3][i][j] = dtm[3][i][j] + dfm[i][k] * Blmat[k][j];
            }
        }
    }
}
