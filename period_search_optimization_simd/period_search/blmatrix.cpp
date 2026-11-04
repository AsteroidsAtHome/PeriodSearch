#include "globals.h"
#include "arrayHelpers.hpp"
#include <cmath>

/**
 * @brief Computes the beta and lambda rotation matrix and its derivatives.
 *
 * This function calculates the rotation matrix and its derivatives with respect to beta and lambda angles.
 * The rotation matrix represents a rotation in 3D space defined by the beta and lambda angles.
 * The derivatives are computed for further use in optimization or sensitivity analysis.
 *
 * @param bet The beta angle (in radians).
 * @param lam The lambda angle (in radians).
 *
 * @note The rotation matrix and its derivatives are stored in the global arrays Blmat and Dblm.
 *
 * @date 8.11.2006
 */
void blmatrix(const double bet, const double lam)
{
    double cb, sb, cl, sl;

    cb = cos(bet);
    sb = sin(bet);
    cl = cos(lam);
    sl = sin(lam);
    gl.Blmat[1][1] = cb * cl;
    gl.Blmat[1][2] = cb * sl;
    gl.Blmat[1][3] = -sb;
    gl.Blmat[2][1] = -sl;
    gl.Blmat[2][2] = cl;
    gl.Blmat[2][3] = 0;
    gl.Blmat[3][1] = sb * cl;
    gl.Blmat[3][2] = sb * sl;
    gl.Blmat[3][3] = cb;
    /* Ders. of Blmat w.r.t. bet */
    gl.Dblm[1][1][1] = -sb * cl;
    gl.Dblm[1][1][2] = -sb * sl;
    gl.Dblm[1][1][3] = -cb;
    gl.Dblm[1][2][1] = 0;
    gl.Dblm[1][2][2] = 0;
    gl.Dblm[1][2][3] = 0;
    gl.Dblm[1][3][1] = cb * cl;
    gl.Dblm[1][3][2] = cb * sl;
    gl.Dblm[1][3][3] = -sb;
    /* Ders. w.r.t. lam */
    gl.Dblm[2][1][1] = -cb * sl;
    gl.Dblm[2][1][2] = cb * cl;
    gl.Dblm[2][1][3] = 0;
    gl.Dblm[2][2][1] = -cl;
    gl.Dblm[2][2][2] = -sl;
    gl.Dblm[2][2][3] = 0;
    gl.Dblm[2][3][1] = -sb * sl;
    gl.Dblm[2][3][2] = sb * cl;
    gl.Dblm[2][3][3] = 0;


    // For Unit tests reference only
    /*printf("Blmat[4][4]:\n");
    for(int j = 0; j <= 3; j++)
    {
        printf("double _blmat_%d[] = { ", j);
        for (int i = 0; i <= 3; i++) {
            printf("%.30f, ", Blmat[i][j]);
        }
        printf("};\n");
    }

    printf("Dblm[3][4][4]:\n");
    for (int k = 0; k <= 3; k++) {
        for (int j = 0; j <= 3; j++)
        {
            printf("double _dblm_%d_%d[] = { ", j, k);
            for (int i = 0; i <= 2; i++) {
                printf("%.30f, ", Dblm[i][j][k]);
            }
            printf("};\n");
        }
    }*/
}
