/* beta, lambda rotation matrix and its derivatives

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS


#include <cmath>
#include "globals.h"

void blmatrix(const double& beta,const double& lambda)
{
    const auto cosBeta = cos(beta);
    const auto sinBeta = sin(beta);
    const auto cosLambda = cos(lambda);
    const auto sinLambda = sin(lambda);

    Blmat[1][1] = cosBeta * cosLambda;
    Blmat[1][2] = cosBeta * sinLambda;
    Blmat[1][3] = -sinBeta;
    Blmat[2][1] = -sinLambda;
    Blmat[2][2] = cosLambda;
    Blmat[2][3] = 0;
    Blmat[3][1] = sinBeta * cosLambda;
    Blmat[3][2] = sinBeta * sinLambda;
    Blmat[3][3] = cosBeta;

    /* Ders. of Blmat w.r.t. bet */
    Dblm[1][1][1] = -sinBeta * cosLambda;
    Dblm[1][1][2] = -sinBeta * sinLambda;
    Dblm[1][1][3] = -cosBeta;
    Dblm[1][2][1] = 0;
    Dblm[1][2][2] = 0;
    Dblm[1][2][3] = 0;
    Dblm[1][3][1] = cosBeta * cosLambda;
    Dblm[1][3][2] = cosBeta * sinLambda;
    Dblm[1][3][3] = -sinBeta;

    /* Ders. w.r.t. lam */
    Dblm[2][1][1] = -cosBeta * sinLambda;
    Dblm[2][1][2] = cosBeta * cosLambda;
    Dblm[2][1][3] = 0;
    Dblm[2][2][1] = -cosLambda;
    Dblm[2][2][2] = -sinLambda;
    Dblm[2][2][3] = 0;
    Dblm[2][3][1] = -sinBeta * sinLambda;
    Dblm[2][3][2] = sinBeta * cosLambda;
    Dblm[2][3][3] = 0;
}
