#pragma once
/* slighly changed code from Numerical Recipes
converted from Mikko's fortran code

8.11.2006
*/

#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "globals.hpp"
#include "declarations.hpp"
#include "constants.h"
#include "OpenClWorker.hpp"
#include <iostream>
#include <chrono>
#include "Array2D.h"

using namespace std::chrono;
using std::cout;
using std::endl;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

/* comment the following line if no YORP */
/*#define YORP*/
int offset = 0;
double xx1[4], xx2[4], dy, sig2i, wt, ymod,
//ytemp[POINTS_MAX + 1], 
//dytemp[POINTS_MAX + 1][MAX_N_PAR + 1],
coef, ave = 0, trial_chisq, wght;                               //moved here due to 64 debugger bug in vs2010

cl_double dyda[MAX_N_PAR + 1]; // , dave[MAX_N_PAR + 1];             //is zero indexed for aligned memory access
                               //double *sig2Iwght, *dY;


double MrqcofCl(double **x1, double **x2, double x3[], double y[],
    double sig[], double a[], int ia[], int ma,
    Array2D<cl_double, MAX_N_PAR + 1, MAX_N_PAR + 1> _alpha2D,
    double beta[], int mfit, int lastone, int lastma)
{
    int i, j, k, l, m, np, np1, np2, jp, ic;

    /* N.B. curv and blmatrix called outside bright
    because output same for all points */
    //curv(a);
    curv1D(a);

    //   #ifdef YORP
    //      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
    // #else      
    blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
    //   #endif      

    for (j = 0; j < mfit; j++)
    {
        for (k = 0; k <= j; k++) 
        {
            _alpha2D.set(j, k, 0);
            //alpha[j][k] = 0;
        }
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
                ymod = bright(xx1, xx2, x3[np], a, dyda, ma);
            else
                ymod = conv(jp, dyda, ma);

            ytemp[jp] = ymod;

            if (Inrel[i]/* == 1*/)
            {
                /*ave = ave + ymod;
                auto t1 = high_resolution_clock::now();
                daveCl(dave, dyda, ma);
                auto t2 = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(t2 - t1).count();
                cout << "'daveCl()' uration: " << duration << endl;*/

                for (l = 1; l <= ma; l++)   //last odd value is not problems
                {
                    dave[l] = dave[l] + dyda[l - 1];
                }
            }

            for (l = 1; l <= ma; l++)
            {
                dytemp[jp][l] = dyda[l - 1];
            }
            /* save lightcurves */

            if (Lastcall == 1)
                Yout[np] = ymod;
        } /* jp, lpoints */

        if (Lastcall != 1)
        {
            //    printf("|");
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
                    //if (l == ma) dytemp[jp][l] = coef * (dytemp[jp][l] - ytemp[jp] * dave[l] / ave); //last odd value is not problem

                    ytemp[jp] = coef * ytemp[jp];
                    /* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
                    dytemp[jp][1] = 0;
                }
            }
            if (ia[0]) //not relative
            {
                //double*& ptr = alpha[0]; // <<<<<<<<<<<<<<<<<<<< this is the solution!
                                         //auto t = &*alpha[0];
                                         //auto t1 = high_resolution_clock::now();
                                         /*for (int cnt = 0; cnt <= MAX_N_PAR; cnt++) {
                                         memcpy(&ddd[cnt*(MAX_N_PAR)], &*alpha[cnt], (MAX_N_PAR) * sizeof(double));
                                         }*/

                                         //auto t2 = high_resolution_clock::now();
                                         //auto duration = duration_cast<microseconds>(t2 - t1).count();
                                         //cout << "'curvCl()' Duration: " << duration << "uSec" << endl;

                /*if (Lpoints[i] > 16) {
                    prepareMrqcofNotRel(Lpoints[i], sig, y, ptr, beta, ma);
                    mrqcofNotRel(Lpoints[i], ptr, beta, ma);
                }*/

                /*for (int cnt = 0; cnt <= MAX_N_PAR; cnt++) {
                memcpy(&*alpha[cnt], &ddd[cnt*(MAX_N_PAR)], (MAX_N_PAR) * sizeof(double));
                }*/

                for (jp = 1; jp <= Lpoints[i]; jp++)
                {
                    double sig2iwght;
                    np2++;
                    j = 0;
                    if (np2 > jp) {
                        for (l = 1; l <= ma; l++)
                            dyda[l - 1] = dytemp[jp][l];
                        sig2i = 1 / (sig[np2] * sig[np2]);
                        _sig2iwght[np2] = sig2i * Weight[np2]; //sig2iwght = sig2i * Weight[np2];
                        _dy[np2] = y[np2] - ytemp[jp];  //dy = y[np2] - ytemp[jp];
                                                        //
                                                        //l=0
                        wt = dyda[0] * _sig2iwght[np2]; //wt = dyda[0] * sig2iwght;
                        _alpha2D.set(j, 0, _alpha2D(j, 0) + wt * dyda[0]);
                        //alpha[j][0] = alpha[j][0] + wt * dyda[0];
                        beta[j] = beta[j] + _dy[np2] * wt;  //beta[j] = beta[j] + dy * wt;
                    }

                    j++;
                    //
                    for (l = 1; l <= lastone; l++)  //line of ones
                    {
                        wt = dyda[l] * _sig2iwght[np2]; //wt = dyda[l] * sig2iwght;
                        k = 0;
                        _alpha2D.set(j, k, _alpha2D(j, k) + wt * dyda[0]);
                        //alpha[j][k] = alpha[j][k] + wt * dyda[0];
                        k++;
                        for (m = 1; m <= l; m++)
                        {
                            _alpha2D.set(j, k, _alpha2D(j, k) + wt * dyda[m]);
                            //alpha[j][k] = alpha[j][k] + wt * dyda[m];
                            k++;
                        } /* m */
                        beta[j] = beta[j] + _dy[np2] * wt; //beta[j] = beta[j] + dy * wt;
                        j++;
                    } /* l */
                    for (; l <= lastma; l++)  //rest parameters
                    {
                        if (ia[l])
                        {
                            wt = dyda[l] * _sig2iwght[np2]; //wt = dyda[l] * sig2iwght;
                            k = 0;
                            _alpha2D.set(j, k, _alpha2D(j, k) + wt * dyda[0]);
                            //alpha[j][k] = alpha[j][k] + wt * dyda[0];
                            k++;
                            int kk = k;
                            for (m = 1; m <= lastone; m++)
                            {
                                _alpha2D.set(j, kk, _alpha2D(j, kk) + wt * dyda[m]);
                                //alpha[j][kk] = alpha[j][kk] + wt * dyda[m];
                                kk++;
                            } /* m */
                            k += lastone;
                            for (m = lastone + 1; m <= l; m++)
                            {
                                if (ia[m])
                                {
                                    _alpha2D.set(j, k, _alpha2D(j, k) + wt * dyda[m]);
                                    //alpha[j][k] = alpha[j][k] + wt * dyda[m];
                                    k++;
                                }
                            } /* m */
                            beta[j] = beta[j] + _dy[np2] * wt; //beta[j] = beta[j] + dy * wt;
                            j++;
                        }
                    } /* l */
                    trial_chisq = trial_chisq + _dy[np2] * _dy[np2] * _sig2iwght[np2];  //trial_chisq = trial_chisq + dy * dy * sig2iwght;
                } /* jp */
                  //auto t2 = high_resolution_clock::now();
                  //auto duration = duration_cast<microseconds>(t2 - t1).count();
                  //cout << "'curvCl()' Duration: " << duration << endl;
            }
            else //relative ia[0]==0
            {
                //printf(":");
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
                            _alpha2D.set(j, k, _alpha2D(j, k) + wt * dyda[m]);
                            //alpha[j][k] = alpha[j][k] + wt * dyda[m];
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
                                _alpha2D.set(j, kk, _alpha2D(j, kk) + wt * dyda[m]);
                                //alpha[j][kk] = alpha[j][kk] + wt * dyda[m];
                                kk++;
                            } /* m */
                            k = lastone;
                            for (m = lastone + 1; m <= l; m++)
                            {
                                if (ia[m])
                                {
                                    _alpha2D.set(j, k, _alpha2D(j, k) + wt * dyda[m]);
                                    //alpha[j][k] = alpha[j][k] + wt * dyda[m];
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
        } /* Lastcall != 1 */

        if ((Lastcall == 1) && (Inrel[i] == 1))
            Sclnw[i] = Scale * Lpoints[i] * sig[np] / ave;

    } /* i,  lcurves */

    double tempalpha[MAX_N_PAR + 1][MAX_N_PAR + 1];
    for (j = 1; j < mfit; j++)
        for (k = 0; k <= j - 1; k++)
            _alpha2D.set(k, j, _alpha2D(j, k));
            //alpha[k][j] = alpha[j][k];


    //deallocate_vector(sig2Iwght);
    //deallocate_vector(dY);

    return trial_chisq;
}