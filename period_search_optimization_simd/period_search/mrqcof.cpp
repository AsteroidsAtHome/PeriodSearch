#include <vector>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

/* comment the following line if no YORP */
/*#define YORP*/

/**
 * @brief Computes the coefficient matrix and right-hand side vector for the Marquardt method.
 *
 * This function calculates the coefficient matrix (alpha) and right-hand side vector (beta_sv)
 * for the nonlinear least-squares fitting using the Marquardt method. It also computes the
 * trial chi-squared value based on the given data and parameters.
 *
 * @param x1 A reference to a 2D vector of doubles representing the first set of independent variable data.
 * @param x2 A reference to a 2D vector of doubles representing the second set of independent variable data.
 * @param x3 A reference to a vector of doubles representing additional independent variable data points.
 * @param y A reference to a vector of doubles representing the dependent variable data.
 * @param sig A reference to a vector of doubles representing the standard deviations of the data points.
 * @param a A reference to a vector of doubles representing the initial parameters.
 * @param ia A reference to a vector of integers indicating which parameters are to be fitted.
 * @param ma An integer representing the total number of parameters.
 * @param beta_sv A reference to a vector of doubles to store the right-hand side vector.
 * @param mfit An integer representing the number of parameters to be fitted.
 * @param lastone An integer representing the last parameter to be fitted.
 * @param lastma An integer representing the last parameter in the list of all parameters.
 * @param trial_chisq A reference to a double to store the trial chi-squared value.
 * @param gl A reference to a globals structure containing necessary global data.
 * @param isCovar A boolean indicating whether the covariance matrix should be computed.
 *
 * @note The function modifies the global variables related to the fitting process. Converted from Mikko's Fortran code.
 *		 'mrqcof' is used by 'mrqmin' to evaluate coefficients.
 *
 * @source Numerical Recipes: Nonlinear least-squares fit, Marquardt�s method.
 *
 * @date 8.11.2006
 */
void CalcStrategyNone::mrqcof(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
	std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, int ma,
	std::vector<double>& beta_sv, int mfit, int lastone, int lastma, double& trial_chisq, globals& gl, const bool isCovar)
{
	int i, j, k, l, m, np, np1, np2, jp, ic;
#if defined __GNUC__
	AlignedOuterVector& alpha = isCovar ? gl.covar : gl.alpha;
#else
	 auto& alpha = isCovar ? gl.covar : gl.alpha;
#endif

	/* N.B. curv and blmatrix called outside bright because output same for all points */
	CalcStrategyNone::curv(a, gl);

	// #ifdef YORP
	//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
	// #else
	blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
	// #endif

	for (j = 0; j < mfit; j++)
	{
		for (k = 0; k <= j; k++)
		{
			alpha[j][k] = 0;
		}
		beta_sv[j] = 0;
	}

	trial_chisq = 0;
	np = 0;
	np1 = 0;
	np2 = 0;

	for (i = 1; i <= gl.Lcurves; i++)
	{
		if (gl.Inrel[i] == 1) /* is the LC relative? */
		{
			gl.ave = 0;
			for (l = 1; l <= ma; l++)
				gl.dave[l] = 0;
		}
		for (jp = 1; jp <= gl.Lpoints[i]; jp++)
		{
			np++;
			for (ic = 1; ic <= 3; ic++) /* position vectors */
			{
				gl.xx1[ic] = x1[np][ic];
				gl.xx2[ic] = x2[np][ic];
			}

			if (i < gl.Lcurves)
			{
				CalcStrategyNone::bright(x3[np], a, ma, gl);
			}
			else
			{
				CalcStrategyNone::conv(jp, ma, gl);
			}

			gl.ytemp[jp] = gl.ymod;

			if (gl.Inrel[i] == 1)
				gl.ave += gl.ymod;

			for (l = 1; l <= ma; l++)
			{
				gl.dytemp[jp][l] = gl.dyda[l - 1];
				gl.dave[l] += gl.dyda[l - 1];
			}
			/* save lightcurves */
		} /* jp, lpoints */

		if (Lastcall != 1)
		{
			for (jp = 1; jp <= gl.Lpoints[i]; jp++)
			{
				np1++;
				if (gl.Inrel[i] == 1)
				{
					gl.coef = sig[np1] * gl.Lpoints[i] / gl.ave;
					for (l = 1; l <= ma; l++)
					{
						gl.dytemp[jp][l] = gl.coef * (gl.dytemp[jp][l] - gl.ytemp[jp] * gl.dave[l] / gl.ave);
					}

					gl.ytemp[jp] *= gl.coef;
					/* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
					gl.dytemp[jp][1] = 0;
				}
			}

			if (ia[0]) //not relative
			{
				for (jp = 1; jp <= gl.Lpoints[i]; jp++)
				{
					gl.ymod = gl.ytemp[jp];
					for (l = 1; l <= ma; l++)
						gl.dyda[l - 1] = gl.dytemp[jp][l];
					np2++;
					gl.sig2i = 1 / (sig[np2] * sig[np2]);
					gl.wght = gl.Weight[np2];
					gl.dy = y[np2] - gl.ymod;
					j = 0;
					//
					double sig2iwght = gl.sig2i * gl.wght;
					//l=0
					gl.wt = gl.dyda[0] * sig2iwght;
					alpha[j][0] += gl.wt * gl.dyda[0];
					beta_sv[j] += gl.dy * gl.wt;
					j++;
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						gl.wt = gl.dyda[l] * sig2iwght;
						k = 0;
						//m=0
						alpha[j][k] += gl.wt * gl.dyda[0];
						k++;
						for (m = 1; m <= l; m++)
						{
							alpha[j][k] += gl.wt * gl.dyda[m];
							k++;
						} /* m */
						beta_sv[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = gl.dyda[l] * sig2iwght;
							k = 0;
							//m=0
							alpha[j][k] += gl.wt * gl.dyda[0];
							k++;
							int kk = k;
							for (m = 1; m <= lastone; m++)
							{
								alpha[j][kk] = alpha[j][kk] + gl.wt * gl.dyda[m];
								kk++;
							} /* m */
							k += lastone;
							for (m = lastone + 1; m <= l; m++)
							{
								if (ia[m])
								{
									alpha[j][k] += gl.wt * gl.dyda[m];
									k++;
								}
							} /* m */
							beta_sv[j] += gl.dy * gl.wt;
							j++;
						}
					} /* l */

					trial_chisq += gl.dy * gl.dy * sig2iwght;
				} /* jp */
			}
			else //relative ia[0]==0
			{
				for (jp = 1; jp <= gl.Lpoints[i]; jp++)
				{
					gl.ymod = gl.ytemp[jp];
					for (l = 1; l <= ma; l++)
						gl.dyda[l - 1] = gl.dytemp[jp][l];
					np2++;
					gl.sig2i = 1 / (sig[np2] * sig[np2]);
					gl.wght = gl.Weight[np2];
					gl.dy = y[np2] - gl.ymod;
					j = 0;
					//
					double sig2iwght = gl.sig2i * gl.wght;
					//l=0
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						gl.wt = gl.dyda[l] * sig2iwght;
						k = 0;
						//m=0
						//
						for (m = 1; m <= l; m++)
						{
							alpha[j][k] += gl.wt * gl.dyda[m];
							k++;
						} /* m */
						beta_sv[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = gl.dyda[l] * sig2iwght;
							//m=0
							//
							int kk = 0;
							for (m = 1; m <= lastone; m++)
							{
								alpha[j][kk] += gl.wt * gl.dyda[m];
								kk++;
							} /* m */
							// k += lastone;
							k = lastone;
							for (m = lastone + 1; m <= l; m++)
							{
								if (ia[m])
								{
									alpha[j][k] += gl.wt * gl.dyda[m];
									k++;
								}
							} /* m */
							beta_sv[j] += gl.dy * gl.wt;
							j++;
						}
					} /* l */

					trial_chisq += gl.dy * gl.dy * sig2iwght;
				} /* jp */
			}
		} /* Lastcall != 1 */

		// TODO: Remove this:
		//if ((Lastcall == 1) && (Inrel[i] == 1))
		//    Sclnw[i] = Scale * Lpoints[i] * sig[np] / gl.ave;

	} /* i,  lcurves */

	//printf("np: %d\n", np);

	for (j = 1; j < mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];
}
