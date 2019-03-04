/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "globals.hpp"
#include "declarations.hpp"
#include "constants.h"
#include "OpenClWorker.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

//#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

int offset = 0;
double xx1[4], xx2[4], dy, sig2i, wt, ymod,
    dytemp[POINTS_MAX + 1][MAX_N_PAR + 1],
    coef, ave = 0, trial_chisq, wght;                           //moved here due to 64 debugger bug in vs2010

cl_double dyda[MAX_N_PAR + 1], dave[MAX_N_PAR + 1];             //is zero indexed for aligned memory access

double mrqcof(const CoordinatesDouble3& ndv,
    const LightPoints<double> &lightPoints,
    double sig[],
    std::vector<double> &a,
    const std::vector<int> &ia,
    int ma,
    double **alpha,
    std::vector<double> &beta,
    const int &mfit)
{
    int j, k, l, m, np, np1, np2, jp;

    /* number of fitted parameters */
    int lastma = mfit - 1;
    int lastone = 0;
    for (j = 1; j <= lastma; j++) //ia[0] is skipped because ia[0]=0 is acceptable inside mrqcof
    {
        if (!ia[j]) break;
        lastone = j;
    }

    for (j = 0; j < mfit; j++)
    {
        for (k = 0; k <= j; k++)
            alpha[j][k] = 0;
        beta[j] = 0;
    }
    trial_chisq = 0;
    np = 0;
    np1 = 0;
    np2 = 0;

    for (auto i = 1; i <= Lcurves; i++)
    {
        if (Inrel[i]/* == 1*/) /* is the LC relative? */
        {
            ave = 0;
            for (l = 1; l <= ma; l++)
                dave[l] = 0;
        }

        for (jp = 1; jp <= Lpoints[i]; jp++)
        {
            np++;
            if (i < Lcurves)
                ymod = Bright2(ndv[np], lightPoints.time[np], a, dyda, ma);
            else
                ymod = conv(jp - 1, ma, dyda);

            ytemp[jp] = ymod;

            if (Inrel[i]/* == 1*/)
            {
                for (l = 1; l <= ma; l++)   //last odd value is not problems
                {
                    dave[l] = dave[l] + dyda[l - 1];
                }
            }

            for (l = 1; l <= ma; l++)
            {
                dytemp[jp][l] = dyda[l - 1];
            }
        } /* jp, lpoints */

        for (jp = 1; jp <= Lpoints[i]; jp++)
        {
            np1++;
            if (Inrel[i] /*== 1*/)
            {
                coef = sig[np1] * Lpoints[i] / ave;
                for (l = 1; l <= ma; l++)
                {
                    dytemp[jp][l] = coef * (dytemp[jp][l] - ytemp[jp] * dave[l] / ave);
                }

                ytemp[jp] = coef * ytemp[jp];

                /* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
                dytemp[jp][1] = 0;
            }
        }

        if (ia[0]) //not relative
        {
            for (jp = 1; jp <= Lpoints[i]; jp++)   // 1 to 288
            {
                ymod = ytemp[jp];
                for (l = 1; l <= ma; l++)
                    dyda[l - 1] = dytemp[jp][l];
                np2++;
                sig2i = 1 / (sig[np2] * sig[np2]);
                wght = Weight[np2];
                const auto sig2Iwght = sig2i * wght;
                dy = lightPoints.brightness[np2] - ymod;
                for (l = 0; l <= lastone; l++)  //line of ones
                {
                    wt = dyda[l] * sig2Iwght;
                    for (m = 0; m <= l; m++)
                    {
                        alpha[l][m] = alpha[l][m] + wt * dyda[m];
                    }
                    beta[l] = beta[l] + dy * wt;
                }
                j = 0;
                for (; l <= lastma; l++)  //rest parameters //--------------------> ???
                {
                    if (ia[l])
                    {
                        wt = dyda[l] * sig2Iwght;
                        k = 1;
                        for (m = 1; m <= lastone; m++)
                        {
                            alpha[j][k] = alpha[j][k] + wt * dyda[m];
                        }
                        k += lastone;
                        for (m = lastone + 1; m <= l; m++)
                        {
                            if (ia[m])
                            {
                                alpha[j][k] = alpha[j][k] + wt * dyda[m];
                                k++;
                            }
                        }
                        beta[j] = beta[j] + dy * wt;
                        j++;
                    }
                }
                trial_chisq = trial_chisq + dy * dy * sig2Iwght;
            } /* jp */
        }
        else //relative ia[0]==0
        {
            for (jp = 1; jp <= Lpoints[i]; jp++)
            {
                ymod = ytemp[jp];
                for (l = 1; l <= ma; l++)
                    dyda[l - 1] = dytemp[jp][l];
                np2++;
                sig2i = 1 / (sig[np2] * sig[np2]);
                wght = Weight[np2];
                dy = lightPoints.brightness[np2] - ymod;
                j = 0;
                //
                double sig2iwght = sig2i * wght;
                //l==0
                //
                for (l = 1; l <= lastone; l++)  //line of ones
                {
                    wt = dyda[l] * sig2iwght;
                    k = 0;
                    //m==1
                    //
                    for (m = 1; m <= l; m++)
                    {
                        alpha[j][k] = alpha[j][k] + wt * dyda[m];
                        k++;
                    } /* m */
                    beta[j] = beta[j] + dy * wt;
                    j++;
                } /* l */
                for (; l <= lastma; l++)  //rest parameters
                {
                    if (ia[l])
                    {
                        wt = dyda[l] * sig2iwght;
                        //m==0
                        //
                        int kk = 0;
                        for (m = 1; m <= lastone; m++)
                        {
                            alpha[j][kk] = alpha[j][kk] + wt * dyda[m];
                            kk++;
                        } /* m */
                        k = lastone;
                        for (m = lastone + 1; m <= l; m++)
                        {
                            if (ia[m])
                            {
                                alpha[j][k] = alpha[j][k] + wt * dyda[m];
                                k++;
                            }
                        } /* m */
                        beta[j] = beta[j] + dy * wt;
                        j++;
                    }
                } /* l */
                trial_chisq = trial_chisq + dy * dy * sig2iwght;
            } /* jp */
        }
    } /* i,  lcurves */

    for (j = 1; j < mfit; j++)
        for (k = 0; k <= j - 1; k++)
            alpha[k][j] = alpha[j][k];

    return trial_chisq;
}

