/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <stdio.h>
#include <stdlib.h>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <immintrin.h>

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

/* comment the following line if no YORP */
/*#define YORP*/

#ifdef __GNUC__
double dyda[MAX_N_PAR + 8] __attribute__((aligned(32)));
#else
__declspec(align(32)) double dyda[MAX_N_PAR + 8]; //is zero indexed for aligned memory access
#endif

double xx1[4], xx2[4], dy, sig2i, wt, ymod,
ytemp[MAX_LC_POINTS + 1], dytemp[MAX_LC_POINTS + 1][MAX_N_PAR + 1 + 4],
dave[MAX_N_PAR + 1 + 4],
dave2[MAX_N_PAR + 1 + 4],
coef, ave = 0, trial_chisq, wght;  //moved here due to 64 debugger bug in vs2010

double mrqcof_fma(double **x1, double **x2, double x3[], double y[],
    double sig[], double a[], int ia[], int ma,
    double **alpha, double beta[], int mfit, int lastone, int lastma)
{
    int i, j, k, l, m, np, np1, np2, jp, ic;

    //__m256d ymm0 = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);

    /* N.B. curv and blmatrix called outside bright
       because output same for all points */
    curv(a);

    //   #ifdef YORP
    //      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
      // #else      
    blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
    //   #endif      

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

    for (i = 1; i <= Lcurves; i++)
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
            for (ic = 1; ic <= 3; ic++) /* position vectors */
            {
                xx1[ic] = x1[np][ic];
                xx2[ic] = x2[np][ic];
            }

            if (i < Lcurves)
                ymod = bright_fma(xx1, xx2, x3[np], a, dyda, ma);
            else
                ymod = conv_fma(jp, dyda, ma);
            
            ytemp[jp] = ymod;

            if (Inrel[i]/* == 1*/)
            {
                ave = ave + ymod;
                for (l = 1; l <= ma; l += 4) //last odd value is not problem
                {
                    __m256d avx_dyda = _mm256_load_pd(&dyda[l - 1]);
                	__m256d avx_dave = _mm256_loadu_pd(&dave[l]);
                    avx_dave = _mm256_add_pd(avx_dave, avx_dyda);
                    _mm256_storeu_pd(&dave[l], avx_dave);
                }
            }

            for (l = 1; l <= ma; l++)
            {
                dytemp[jp][l] = dyda[l - 1];
                //				 if (Inrel[i]/* == 1*/) dave[l] = dave[l] + dyda[l-1];
            }
            /* save lightcurves */

            if (Lastcall == 1)
                Yout[np] = ymod;
        } /* jp, lpoints */

        if (Lastcall != 1)
        {
            __m256d avx_ave, avx_coef, avx_ytemp;
            avx_ave = _mm256_set1_pd(ave);
            for (jp = 1; jp <= Lpoints[i]; jp++)
            {
                np1++;
                if (Inrel[i] /*== 1*/)
                {
                    coef = sig[np1] * Lpoints[i] / ave;
                    avx_coef = _mm256_set1_pd(coef);
                    avx_ytemp = _mm256_broadcast_sd(&ytemp[jp]);
                    for (l = 1; l <= ma; l += 4)
                    {
                        __m256d avx_dytemp = _mm256_loadu_pd(&dytemp[jp][l]), avx_dave = _mm256_loadu_pd(&dave[l]);
                        avx_dytemp = _mm256_sub_pd(avx_dytemp, _mm256_div_pd(_mm256_mul_pd(avx_ytemp, avx_dave), avx_ave));
                        avx_dytemp = _mm256_mul_pd(avx_dytemp, avx_coef);
                        _mm256_storeu_pd(&dytemp[jp][l], avx_dytemp);
                    }
                    //			if (l==ma) dytemp[jp][l] = coef * (dytemp[jp][l] - ytemp[jp] * dave[l] / ave); //last odd value is not problem

                    ytemp[jp] = coef * ytemp[jp];
                    /* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
                    dytemp[jp][1] = 0;
                }
            }
            if (ia[0]) //not relative
            {
                for (jp = 1; jp <= Lpoints[i]; jp++)
                {
                    ymod = ytemp[jp];
                    for (l = 1; l <= ma; l++)
                        dyda[l - 1] = dytemp[jp][l];
                    np2++;
                    sig2i = 1 / (sig[np2] * sig[np2]);
                    wght = Weight[np2];
                    dy = y[np2] - ymod;
                    j = 0;
                    //
                    double sig2iwght = sig2i * wght;
                    //l=0
                    wt = dyda[0] * sig2iwght;
                    alpha[j][0] = alpha[j][0] + wt * dyda[0];
                    beta[j] = beta[j] + dy * wt;
                    j++;
                    //
                    for (l = 1; l <= lastone; l++)  //line of ones
                    {
                        wt = dyda[l] * sig2iwght;
                        __m256d avx_wt = _mm256_set1_pd(wt);
                        k = 0;
                        //m=0
                        alpha[j][k] = alpha[j][k] + wt * dyda[0];
                        k++;
                        for (m = 1; m <= l; m += 4)
                        {
                            __m256d avx_alpha = _mm256_loadu_pd(&alpha[j][k]);
                            __m256d avx_dyda = _mm256_loadu_pd(&dyda[m]);
                            avx_alpha = _mm256_fmadd_pd(avx_wt, avx_dyda, avx_alpha);
                            //avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
                            _mm256_storeu_pd(&alpha[j][k], avx_alpha);
                            k += 4;
                        } /* m */
                        beta[j] = beta[j] + dy * wt;
                        j++;
                    } /* l */
                    for (; l <= lastma; l++)  //rest parameters
                    {
                        if (ia[l])
                        {
                            wt = dyda[l] * sig2iwght;
                            __m256d avx_wt = _mm256_set1_pd(wt);
                            k = 0;
                            //m=0
                            alpha[j][k] = alpha[j][k] + wt * dyda[0];
                            k++;
                            int kk = k;
                            for (m = 1; m <= lastone; m += 4)
                            {
                                __m256d avx_alpha = _mm256_loadu_pd(&alpha[j][kk]);
                            	__m256d avx_dyda = _mm256_loadu_pd(&dyda[m]);
                                avx_alpha = _mm256_fmadd_pd(avx_wt, avx_dyda, avx_alpha);
                                //avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
                                _mm256_storeu_pd(&alpha[j][kk], avx_alpha);
                                kk += 4;
                            } /* m */
                            k += lastone;
                            for (m = lastone + 1; m <= l; m++)
                                if (ia[m])
                                {
                                    alpha[j][k] = alpha[j][k] + wt * dyda[m];
                                    k++;
                                }
                            beta[j] = beta[j] + dy * wt;
                            j++;
                        }
                    } /* l */
                    trial_chisq = trial_chisq + dy * dy * sig2iwght;
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
                    dy = y[np2] - ymod;
                    j = 0;
                    //
                    double sig2iwght = sig2i * wght;
                    // l=0
                    //
                    for (l = 1; l <= lastone; l++)  //line of ones
                    {
                        wt = dyda[l] * sig2iwght;
                        __m256d avx_wt = _mm256_set1_pd(wt);
                        k = 0;
                        //m=0
                        //
                        for (m = 1; m <= l; m += 4)
                        {
                            __m256d avx_alpha = _mm256_load_pd(&alpha[j][k]);
                        	__m256d avx_dyda = _mm256_loadu_pd(&dyda[m]);
                            avx_alpha = _mm256_fmadd_pd(avx_wt, avx_dyda, avx_alpha);
                            //avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
                            _mm256_store_pd(&alpha[j][k], avx_alpha);
                            k += 4;
                        } /* m */
                        beta[j] = beta[j] + dy * wt;
                        j++;
                    } /* l */
                    for (; l <= lastma; l++)  //rest parameters
                    {
                        if (ia[l])
                        {
                            wt = dyda[l] * sig2iwght;
                            __m256d avx_wt = _mm256_set1_pd(wt);
                            //m=0
                            //
                            int kk = 0;
                            for (m = 1; m <= lastone; m += 4)
                            {
                                __m256d avx_alpha = _mm256_load_pd(&alpha[j][kk]);
                            	__m256d avx_dyda = _mm256_loadu_pd(&dyda[m]);
                                avx_alpha = _mm256_fmadd_pd(avx_wt, avx_dyda, avx_alpha);
                                //avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
                                _mm256_store_pd(&alpha[j][kk], avx_alpha);
                                kk += 4;
                            } /* m */
                            k = lastone;
                            for (m = lastone + 1; m <= l; m++)
                                if (ia[m])
                                {
                                    alpha[j][k] = alpha[j][k] + wt * dyda[m];
                                    k++;
                                }
                            beta[j] = beta[j] + dy * wt;
                            j++;
                        }
                    } /* l */
                    trial_chisq = trial_chisq + dy * dy * sig2iwght;
                } /* jp */
            }
        } /* Lastcall != 1 */

        if ((Lastcall == 1) && (Inrel[i] == 1))
            Sclnw[i] = Scale * Lpoints[i] * sig[np] / ave;

    } /* i,  lcurves */

    for (j = 1; j < mfit; j++)
        for (k = 0; k <= j - 1; k++)
            alpha[k][j] = alpha[j][k];

    return trial_chisq;
}

