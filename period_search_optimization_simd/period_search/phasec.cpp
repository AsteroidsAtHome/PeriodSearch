#include <cmath>
#include "globals.h"
#include "arrayHelpers.hpp"

/**
 * @brief Computes the linear-exponential phase function and its derivatives.
 *
 * This function calculates the linear-exponential phase function based on the given parameters
 * and computes its derivatives with respect to the parameters. The model used is an exponential-linear
 * combination where the constant term is set to 1.
 *
 * @param dcdp An array of doubles to store the derivatives of the phase function with respect to the parameters.
 * @param alpha A double representing the phase angle in radians.
 * @param p An array of doubles representing the model parameters.
 *
 * @note The function stores the computed phase function value in gl.Scale.
 *
 * @source Converted from Mikko's Fortran code
 *
 * @date 8.11.2006
 */
void phasec(double dcdp[], const double alpha, double p[], struct globals& gl)
{
    double e, c;

    /* Exp-lin model (const.term=1.) */
    e = exp(-alpha / p[2]);
    c = 1 + p[1] * e + p[3] * alpha;

    /* derivatives */
    dcdp[1] = e;
    dcdp[2] = p[1] * e * alpha / (p[2] * p[2]);
    dcdp[3] = alpha;

    gl.Scale = c;

    //printf("\nScale: %.30f", Scale);
    /*printf("dcdp[6]:\n");
    printf("dcdp[6] = {");
    for(int i = 0; i <= 5; i++)
    {
        printf("%.30f, ", dcdp[i]);
    }
    printf("};\n");*/
}
