/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <cstdio>
#include <cstdlib>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "CalcStrategySve.hpp"
#include "arrayHelpers.hpp"

/* comment the following line if no YORP */
/*#define YORP*/

//double xx1[4], xx2[4], dy, sig2i, wt, dyda[MAX_N_PAR + 1], ymod,
//ytemp[MAX_LC_POINTS + 1], dytemp[MAX_LC_POINTS + 1][MAX_N_PAR + 1],
//dave[MAX_N_PAR + 1],
//coef, ave = 0, trial_chisq, wght;

#if defined(__GNUC__)
__attribute__((__target__("+sve")))
#endif
double CalcStrategySve::mrqcof(double** x1, double** x2, double x3[], double y[],
	double sig[], double a[], int ia[], int ma,
	double** alpha, double beta[], int mfit, int lastone, int lastma)
{
	int i, j, k, l, m, np, np1, np2, jp, ic;

    svbool_t pg = svptrue_b64();

	/* N.B. curv and blmatrix called outside bright
	   because output same for all points */
	CalcStrategySve::curv(a);

	//   #ifdef YORP
	//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
	  // #else
	blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
	//   #endif

	//for (j = 1; j <= mfit; j++)
	for (j = 0; j < mfit; j++)
	{
		//for (k = 1; k <= j; k++)
		for (k = 0; k <= j; k++)
		{
			alpha[j][k] = 0;
		}
		beta[j] = 0;
	}

	trial_chisq = 0;
	np = 0;
	np1 = 0;
	np2 = 0;

	for (i = 1; i <= Lcurves; i++)
	{
		if (Inrel[i] == 1) /* is the LC relative? */
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
				ymod = CalcStrategySve::bright(xx1, xx2, x3[np], a, dyda, ma);
			else
				ymod = CalcStrategySve::conv(jp, dyda, ma);

			//printf("[%3d][%3d] % 0.6f\n", i, jp, ymod);
			/*if(jp == 1)
				printf("[%3d][%3d] % 0.6f\n", i, jp, dyda[1]);*/
			//printArray(dyda, 208, "dyda");

			ytemp[jp] = ymod;

			//if (Inrel[i]/* == 1*/)
			//{
			//	ave = ave + ymod;
			//	for (l = 1; l <= ma; l++)
			//	{
			//		dytemp[jp][l] = dyda[l - 1];
			//		dave[l] += dyda[l - 1];
			//	}
			//}
			//else
			//{
			//	for (l = 1; l <= ma; l++)
			//	{
			//		dytemp[jp][l] = dyda[l - 1];
			//	}
			//}
			if (Inrel[i] == 1)
				ave = ave + ymod;

            for (l = 1; l <= ma; l += svcntd()) {
                svfloat64_t avx_dyda = svld1_f64(pg, &dyda[l - 1]);
                svfloat64_t avx_dave = svld1_f64(pg, &dave[l]);

                avx_dave = svadd_f64_x(pg, avx_dave, avx_dyda);

                svst1_f64(pg, &dytemp[jp][l], avx_dyda);
                svst1_f64(pg, &dave[l], avx_dave);
            }
		    /* save lightcurves */

            if (Lastcall == 1) 
	          Yout[np] = ymod;
      	} /* jp, lpoints */

		if (Lastcall != 1)
		{
			svfloat64_t avx_ave, avx_coef, avx_ytemp;
            avx_ave = svdup_n_f64(ave);

			for (jp = 1; jp <= Lpoints[i]; jp++)
			{
				np1++;
				if (Inrel[i] == 1)
				{
					coef = sig[np1] * Lpoints[i] / ave;

					svfloat64_t avx_coef = svdup_n_f64(coef);
                    svfloat64_t avx_ytemp = svdup_n_f64(ytemp[jp]);

                    for (l = 1; l <= ma; l += svcntd()) {
                        svfloat64_t avx_dytemp = svld1_f64(pg, &dytemp[jp][l]);
                        svfloat64_t avx_dave = svld1_f64(pg, &dave[l]);

                        svfloat64_t avx_intermediate = svdiv_f64_x(pg, svmul_f64_x(pg, avx_ytemp, avx_dave), avx_ave);
                        avx_dytemp = svmul_f64_x(pg, svsub_f64_x(pg, avx_dytemp, avx_intermediate), avx_coef);

                        svst1_f64(pg, &dytemp[jp][l], avx_dytemp);
                    }

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
                        svfloat64_t avx_wt = svdup_n_f64(wt);

						k = 0;
						//m=0
						alpha[j][k] = alpha[j][k] + wt * dyda[0];
						k++;
                        for (m = 1; m <= l; m += svcntd()) {
                            svfloat64_t avx_alpha = svld1_f64(pg, &alpha[j][k]);
                            svfloat64_t avx_dyda = svld1_f64(pg, &dyda[m]);
    
                            svfloat64_t avx_result = svmla_f64_x(pg, avx_alpha, avx_wt, avx_dyda);

                            svst1_f64(pg, &alpha[j][k], avx_result);

                            k += svcntd();
                        } /* m */

						beta[j] = beta[j] + dy * wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							wt = dyda[l] * sig2iwght;
							svfloat64_t avx_wt = svdup_n_f64(wt);
							k = 0;
							//m=0
							alpha[j][k] = alpha[j][k] + wt * dyda[0];
							k++;
							int kk = k;
                            for (m = 1; m <= lastone; m += svcntd()) {
                                svfloat64_t avx_alpha = svld1_f64(pg, &alpha[j][kk]);
                                svfloat64_t avx_dyda = svld1_f64(pg, &dyda[m]);
    
                                svfloat64_t avx_result = svmla_f64_x(pg, avx_alpha, avx_wt, avx_dyda);

                                svst1_f64(pg, &alpha[j][kk], avx_result);

                                kk += svcntd();
                            } /* m */

							k += lastone;
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
					//l=0
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						wt = dyda[l] * sig2iwght;
                        svfloat64_t avx_wt = svdup_n_f64(wt);
						k = 0;
						//m=0
                        //
                        for (m = 1; m <= l; m += svcntd()) {
                            svfloat64_t avx_alpha = svld1_f64(pg, &alpha[j][k]);
                            svfloat64_t avx_dyda = svld1_f64(pg, &dyda[m]);
    
                            svfloat64_t avx_result = svmla_f64_x(pg, avx_alpha, avx_wt, avx_dyda);

                            svst1_f64(pg, &alpha[j][k], avx_result);

                            k += svcntd();
                        } /* m */

						beta[j] = beta[j] + dy * wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							wt = dyda[l] * sig2iwght;
							svfloat64_t avx_wt = svdup_n_f64(wt);
							//m=0
							//
							int kk = 0;
                            for (m = 1; m <= lastone; m += svcntd()) {
                                svfloat64_t avx_alpha = svld1_f64(pg, &alpha[j][kk]);
                                svfloat64_t avx_dyda = svld1_f64(pg, &dyda[m]);
    
                                svfloat64_t avx_result = svmla_f64_x(pg, avx_alpha, avx_wt, avx_dyda);

                                svst1_f64(pg, &alpha[j][kk], avx_result);

                                kk += svcntd();
                            } /* m */
							// k += lastone;
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
		} /* Lastcall != 1 */

		if ((Lastcall == 1) && (Inrel[i] == 1))
			Sclnw[i] = Scale * Lpoints[i] * sig[np] / ave;

	} /* i,  lcurves */

	for (j = 1; j < mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];

	return trial_chisq;
}

