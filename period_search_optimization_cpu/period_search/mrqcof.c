/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <cstdio>
#include <cstdlib>
#include "globals.h"
#include "declarations.h"
#include "constants.h"

/* comment the following line if no YORP */
/*#define YORP*/
double xx1[4], xx2[4], dy, sig2i, wt, dyda[MAX_N_PAR + 1], ymod,
ytemp[POINTS_MAX + 1], dytemp[POINTS_MAX + 1][MAX_N_PAR + 1],
dave[MAX_N_PAR + 1],
coef, ave = 0, trial_chisq, wght;

double mrqcof(double** x1, double** x2, double x3[], double y[],
	double sig[], double a[], int ia[], int ma,
	double** alpha, double beta[], int mfit, int lastone, int lastma)
{
	int i, j, k, l, m, np, np1, np2, jp, ic;


	/* N.B. curv and blmatrix called outside bright
	   because output same for all points */
	curv(a);

	//   #ifdef YORP
	//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
	  // #else
	blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
	//   #endif

	for (j = 1; j <= mfit; j++)
	{
		for (k = 1; k <= j; k++)
			alpha[j][k] = 0;
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
				ymod = bright(xx1, xx2, x3[np], a, dyda, ma);
			else
				ymod = conv(jp, dyda, ma);

			ytemp[jp] = ymod;

			if (Inrel[i] == 1)
				ave = ave + ymod;

			for (l = 1; l <= ma; l++)
			{
				dytemp[jp][l] = dyda[l];
				if (Inrel[i] == 1)
					dave[l] = dave[l] + dyda[l];
			}
			/* save lightcurves */

			if (Lastcall == 1)
				Yout[np] = ymod;
		} /* jp, lpoints */

		if (Lastcall != 1)
		{
			for (jp = 1; jp <= Lpoints[i]; jp++)
			{
				np1++;
				if (Inrel[i] == 1)
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

			if (ia[1]) //not relative
			{
				for (jp = 1; jp <= Lpoints[i]; jp++)
				{
					ymod = ytemp[jp];
					for (l = 1; l <= ma; l++)
						dyda[l] = dytemp[jp][l];
					np2++;
					sig2i = 1 / (sig[np2] * sig[np2]);
					wght = Weight[np2];
					dy = y[np2] - ymod;
					j = 0;
					//
					double sig2iwght = sig2i * wght;
					//
					for (l = 1; l <= lastone; l++)
					{
						j++;
						wt = dyda[l] * sig2iwght;
						k = 0;
						for (m = 1; m <= l; m++)
						{
							k++;
							alpha[j][k] = alpha[j][k] + wt * dyda[m];
						} /* m */
						beta[j] = beta[j] + dy * wt;
					} /* l */
					for (; l <= lastma; l++)
					{
						if (ia[l])
						{
							j++;
							wt = dyda[l] * sig2iwght;
							k = 0;
							for (m = 1; m <= lastone; m++)
							{
								k++;
								alpha[j][k] = alpha[j][k] + wt * dyda[m];
							} /* m */
							for (; m <= l; m++)
							{
								if (ia[m])
								{
									k++;
									alpha[j][k] = alpha[j][k] + wt * dyda[m];
								}
							} /* m */
							beta[j] = beta[j] + dy * wt;
						}
					} /* l */
					trial_chisq = trial_chisq + dy * dy * sig2iwght;
				} /* jp */
			}
			else //relative ia[1]==0
			{
				for (jp = 1; jp <= Lpoints[i]; jp++)
				{
					ymod = ytemp[jp];
					for (l = 1; l <= ma; l++)
						dyda[l] = dytemp[jp][l];
					np2++;
					sig2i = 1 / (sig[np2] * sig[np2]);
					wght = Weight[np2];
					dy = y[np2] - ymod;
					j = 0;
					//
					double sig2iwght = sig2i * wght;
					//l==1
					//
					for (l = 2; l <= lastone; l++)
					{
						j++;
						wt = dyda[l] * sig2iwght;
						k = 0;
						//m==1
						//
						for (m = 2; m <= l; m++)
						{
							k++;
							alpha[j][k] = alpha[j][k] + wt * dyda[m];
						} /* m */
						beta[j] = beta[j] + dy * wt;
					} /* l */
					for (; l <= lastma; l++)
					{
						if (ia[l])
						{
							j++;
							wt = dyda[l] * sig2iwght;
							k = 0;
							//m==1
							//
							for (m = 2; m <= lastone; m++)
							{
								k++;
								alpha[j][k] = alpha[j][k] + wt * dyda[m];
							} /* m */
							for (; m <= l; m++)
							{
								if (ia[m])
								{
									k++;
									alpha[j][k] = alpha[j][k] + wt * dyda[m];
								}
							} /* m */
							beta[j] = beta[j] + dy * wt;
						}
					} /* l */
					trial_chisq = trial_chisq + dy * dy * sig2iwght;
				} /* jp */
			}
		} /* Lastcall != 1 */

		if ((Lastcall == 1) && (Inrel[i] == 1))
			Sclnw[i] = Scale * Lpoints[i] * sig[np] / ave;

	} /* i,  lcurves */

	for (j = 2; j <= mfit; j++)
		for (k = 1; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];

	return trial_chisq;
}

