#include <cmath>
#include "globals.h"
#include "arrayHelpers.hpp"
#include "constants.h"

/**
 * @brief Computes the rotation matrix and its derivatives.
 *
 * This function calculates the rotation matrix for a given angular velocity (`omg`) and time (`t`),
 * as well as the derivatives of the rotation matrix with respect to the angular velocity.
 *
 * @param omg The angular velocity in radians per unit time.
 * @param t The time at which the rotation matrix is evaluated.
 * @param tmat A 2D array to store the computed rotation matrix.
 * @param dtm A 3D array to store the derivatives of the rotation matrix with respect to angular velocity.
 *
 * @note The function reads `gl.Blmat` and `gl.Dblm` (computed by blmatrix()).
 *
 * @source Converted from Mikko's Fortran code
 *
 * @date 8.11.2006
 */
void matrix(const double omg, const double t, double tmat[][4], double dtm[][4][4], const struct globals& gl)
{
    double f, cf, sf, dfm[4][4], fmat[4][4];

    int i, j, k;

    /* phase of rotation */
    f = omg * t + Phi_0;
    f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
    cf = cos(f);
    sf = sin(f);
    /* rotation matrix, Z axis, angle f */
    fmat[1][1] = cf;
    fmat[1][2] = sf;
    fmat[1][3] = 0;
    fmat[2][1] = -sf;
    fmat[2][2] = cf;
    fmat[2][3] = 0;
    fmat[3][1] = 0;
    fmat[3][2] = 0;
    fmat[3][3] = 1;
    /* Ders. w.r.t omg */
    dfm[1][1] = -t * sf;
    dfm[1][2] = t * cf;
    dfm[1][3] = 0;
    dfm[2][1] = -t * cf;
    dfm[2][2] = -t * sf;
    dfm[2][3] = 0;
    dfm[3][1] = 0;
    dfm[3][2] = 0;
    dfm[3][3] = 0;
    /* Construct tmat (complete rotation matrix) and its derivatives */
    for (i = 1; i <= 3; i++)
        for (j = 1; j <= 3; j++)
        {
            tmat[i][j] = 0;
            dtm[1][i][j] = 0;
            dtm[2][i][j] = 0;
            dtm[3][i][j] = 0;
            for (k = 1; k <= 3; k++)
            {
                tmat[i][j] = tmat[i][j] + fmat[i][k] * gl.Blmat[k][j];
                dtm[1][i][j] = dtm[1][i][j] + fmat[i][k] * gl.Dblm[1][k][j];
                dtm[2][i][j] = dtm[2][i][j] + fmat[i][k] * gl.Dblm[2][k][j];
                dtm[3][i][j] = dtm[3][i][j] + dfm[i][k] * gl.Blmat[k][j];
            }
        }


    /*printf("\ntmat[4][4]:\n");
    for(int q = 0; q <= 4; q++)
    {
        printf("double _tmat_%d[] ={", q);
        for(int p = 0; p <= 4; p++)
        {
            printf("%.30f, ", tmat[p][q]);
        }
        printf("};\n");
    }

    printf("\ndtm[4][4][4]:\n");
    for (int r = 0; r <= 4; r++) {
        for (int q = 0; q <= 4; q++)
        {
            printf("double _dtm_%d_%d[] ={", q, r);
            for (int p = 0; p <= 4; p++)
            {
                printf("%.30f, ", dtm[p][q][r]);
            }
            printf("};\n");
        }
    }*/
}
