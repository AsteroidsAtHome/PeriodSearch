/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <cstdio>
#include <cstdlib>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "CalcStrategyAsimd.hpp"
#include "arrayHelpers.hpp"
#include <arm_neon.h>

/* comment the following line if no YORP */
/*#define YORP*/

#if defined __GNUG__ && !defined __clang__
__attribute__((__target__("arch=armv8-a+simd")))
#elif defined __GNUG__ && __clang__
// NOTE: The following generates warning: unsupported architecture 'armv8-a+simd' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// __attribute__((target("arch=armv8-a+simd")))
#endif

void CalcStrategyAsimd::mrqcof(double** x1, double** x2, double x3[], double y[],
	double sig[], double a[], int ia[], int ma,
	double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq)
{
	int i, j, k, l, m, np, np1, np2, jp, ic;

	/* N.B. curv and blmatrix called outside bright because output same for all points */
	CalcStrategyAsimd::curv(a);

	//   #ifdef YORP
	//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
	//   #else
	blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
	//   #endif

	for (j = 0; j < mfit; j++)
	{
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
			{
				CalcStrategyAsimd::bright(xx1, xx2, x3[np], a, dyda, ma, ymod);
			}
			else
			{
				CalcStrategyAsimd::conv(jp, dyda, ma, ymod);
			}

			ytemp[jp] = ymod;

			if (Inrel[i] == 1)
                ave += ymod;

            for (l = 1; l <= ma; l += 2) {
        		float64x2_t avx_dyda = vld1q_f64(&dyda[l - 1]);
        		float64x2_t avx_dave = vld1q_f64(&dave[l]);

        		avx_dave = vaddq_f64(avx_dave, avx_dyda);

        		vst1q_f64(&dytemp[jp][l], avx_dyda);
        		vst1q_f64(&dave[l], avx_dave);
    		}
		    /* save lightcurves */

            if (Lastcall == 1)
	          Yout[np] = ymod;
      	} /* jp, lpoints */

		if (Lastcall != 1)
		{
			float64x2_t avx_ave, avx_coef, avx_ytemp;
     		avx_ave = vdupq_n_f64(ave);

			for (jp = 1; jp <= Lpoints[i]; jp++)
			{
				np1++;
				if (Inrel[i] == 1)
				{
					coef = sig[np1] * Lpoints[i] / ave;

					float64x2_t avx_coef = vdupq_n_f64(coef);
            		float64x2_t avx_ytemp = vld1q_dup_f64(&ytemp[jp]);

            		for (l = 1; l <= ma; l += 2) {
               			float64x2_t avx_dytemp = vld1q_f64(&dytemp[jp][l]);
               			float64x2_t avx_dave = vld1q_f64(&dave[l]);

               			avx_dytemp = vsubq_f64(avx_dytemp, vdivq_f64(vmulq_f64(avx_ytemp, avx_dave), avx_ave));
               			avx_dytemp = vmulq_f64(avx_dytemp, avx_coef);

               			vst1q_f64(&dytemp[jp][l], avx_dytemp);
            		}

					ytemp[jp] *= coef;
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
					alpha[j][0] += wt * dyda[0];
					beta[j] += dy * wt;
					j++;
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						wt = dyda[l] * sig2iwght;
						float64x2_t avx_wt = vdupq_n_f64(wt);
						k = 0;
						//m=0
						alpha[j][k] += wt * dyda[0];
						k++;
						for (m = 1; m <= l; m += 2) {
               				float64x2_t avx_alpha = vld1q_f64(&alpha[j][k]);
               				float64x2_t avx_dyda = vld1q_f64(&dyda[m]);
               				float64x2_t avx_result = vmlaq_f64(avx_alpha, avx_wt, avx_dyda);

               				vst1q_f64(&alpha[j][k], avx_result);

               				k += 2;
            			} /* m */
						beta[j] += dy * wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							wt = dyda[l] * sig2iwght;
							float64x2_t avx_wt = vdupq_n_f64(wt);
							k = 0;
							//m=0
							alpha[j][k] += wt * dyda[0];
							k++;
							int kk = k;
							for (m = 1; m <= lastone; m += 2) {
               					float64x2_t avx_alpha = vld1q_f64(&alpha[j][kk]);
               					float64x2_t avx_dyda = vld1q_f64(&dyda[m]);
               					float64x2_t avx_result = vmlaq_f64(avx_alpha, avx_wt, avx_dyda);

               					vst1q_f64(&alpha[j][kk], avx_result);

               					kk += 2;
            				} /* m */
							k += lastone;
							for (m = lastone + 1; m <= l; m++)
							{
								if (ia[m])
								{
									alpha[j][k] += wt * dyda[m];
									k++;
								}
							} /* m */
							beta[j] += dy * wt;
							j++;
						}
					} /* l */
					trial_chisq += dy * dy * sig2iwght;
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
						float64x2_t avx_wt = vdupq_n_f64(wt);
						k = 0;
						//m=0
						//
						for (m = 1; m <= l; m += 2) {
               				float64x2_t avx_alpha = vld1q_f64(&alpha[j][k]);
               				float64x2_t avx_dyda = vld1q_f64(&dyda[m]);
               				float64x2_t avx_result = vmlaq_f64(avx_alpha, avx_wt, avx_dyda);

               				vst1q_f64(&alpha[j][k], avx_result);

               				k += 2;
            			} /* m */
						beta[j] += dy * wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							wt = dyda[l] * sig2iwght;
							float64x2_t avx_wt = vdupq_n_f64(wt);
							//m=0
							//
							int kk = 0;
							for (m = 1; m <= lastone; m += 2) {
               					float64x2_t avx_alpha = vld1q_f64(&alpha[j][kk]);
               					float64x2_t avx_dyda = vld1q_f64(&dyda[m]);
               					float64x2_t avx_result = vmlaq_f64(avx_alpha, avx_wt, avx_dyda);

               					vst1q_f64(&alpha[j][kk], avx_result);

               					kk += 2;
            				} /* m */
							// k += lastone;
							k = lastone;
							for (m = lastone + 1; m <= l; m++)
							{
								if (ia[m])
								{
									alpha[j][k] += wt * dyda[m];
									k++;
								}
							} /* m */
							beta[j] += dy * wt;
							j++;
						}
					} /* l */
					trial_chisq += dy * dy * sig2iwght;
				} /* jp */
			}
		} /* Lastcall != 1 */

		if ((Lastcall == 1) && (Inrel[i] == 1))
			Sclnw[i] = Scale * Lpoints[i] * sig[np] / ave;

	} /* i,  lcurves */

	for (j = 1; j < mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];
}

