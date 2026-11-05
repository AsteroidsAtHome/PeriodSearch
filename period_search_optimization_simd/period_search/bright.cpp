#include <cmath>
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

/**
 * @brief Computes integrated brightness of all visible and illuminated areas and its derivatives.
 *
 * This function calculates the integrated brightness of all visible and illuminated areas based on the provided time `t`,
 * coefficient vector `cg`, and global data. It also computes the derivatives of the brightness with respect to the coefficients.
 *
 * @param t The time at which the brightness is evaluated.
 * @param cg A reference to a vector of doubles containing the coefficients for the brightness calculation.
 * @param ncoef An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables `ymod` and `dyda`.
 *
 * @date 8.11.2006
 * @author Josef Durec
 */
void CalcStrategyNone::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j;
	incl_count = 0;
	double *ee = gl.xx1;
	double *ee0 = gl.xx2;

	tmpdyda1 = 0;
	tmpdyda2 = 0;
	tmpdyda3 = 0;
	tmpdyda4 = 0;
	tmpdyda5 = 0;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);				/* Lambert */
	cls = cg[ncoef];						/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	double alpha = acos(cos_alpha);

	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php, gl);				/* computes also gl.Scale */

	matrix(cg[ncoef0], t, tmat, dtm, gl);

	gl.ymod = 0;

	/* Directions (and derivatives) in the rotating system */
	for (i = 1; i <= 3; i++)
	{
		e[i] = 0;
		e0[i] = 0;
		for (j = 1; j <= 3; j++)
		{
			e[i] += tmat[i][j] * ee[j];
			e0[i] += tmat[i][j] * ee0[j];
			de[i][j] = 0;
			de0[i][j] = 0;
			for (int k = 1; k <= 3; k++)
			{
				de[i][j] += dtm[j][i][k] * ee[k];
				de0[i][j] += dtm[j][i][k] * ee0[k];
			}
		}
	}

	/*Integrated brightness (phase coefficients used later) */
    for (i = 0; i < Numfac; i++)
	{
        const double lmu = e[1] * gl.Nor[0][i] + e[2] * gl.Nor[1][i] + e[3] * gl.Nor[2][i];
        const double lmu0 = e0[1] * gl.Nor[0][i] + e0[2] * gl.Nor[1][i] + e0[3] * gl.Nor[2][i];
		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + cls / dnom);
			gl.ymod += gl.Area[i] * s;
			//
			incl[incl_count] = i;
			dbr[incl_count++] = gl.Darea[i] * s;
			//
            const double dsmu = cls * pow(lmu0 / dnom, 2) + cl * lmu0;
            const double dsmu0 = cls * pow(lmu / dnom, 2) + cl * lmu;

			double sum1 = 0, sum2 = 0, sum3 = 0;
			double sum10 = 0, sum20 = 0, sum30 = 0;

			for (j = 1; j <= 3; j++)
			{
				sum1  += gl.Nor[j-1][i] * de[j][1];
				sum10 += gl.Nor[j-1][i] * de0[j][1];
				sum2  += gl.Nor[j-1][i] * de[j][2];
				sum20 += gl.Nor[j-1][i] * de0[j][2];
				sum3  += gl.Nor[j-1][i] * de[j][3];
				sum30 += gl.Nor[j-1][i] * de0[j][3];
			}

			tmpdyda1 += gl.Area[i] * (dsmu * sum1 + dsmu0 * sum10);
			tmpdyda2 += gl.Area[i] * (dsmu * sum2 + dsmu0 * sum20);
			tmpdyda3 += gl.Area[i] * (dsmu * sum3 + dsmu0 * sum30);
			tmpdyda4 += gl.Area[i] * lmu * lmu0;
			tmpdyda5 += gl.Area[i] * lmu * lmu0 / (lmu + lmu0);
		}
	}

	/* Derivatives of brightness w.r.t. g-coefficients */
	for (i = 1; i <= ncoef0 - 3; i++)
	{
		tmpdyda = 0;
		for (j = 0; j < incl_count; j++)
		{
			tmpdyda += dbr[j] * gl.Dg[incl[j]][i - 1];
		}
		gl.dyda[i - 1] = gl.Scale * tmpdyda;
	}

	/* Derivatives of brightness w.r.t. rotation parameters */
	gl.dyda[ncoef0 - 3 + 1 - 1] = gl.Scale * tmpdyda1;
	gl.dyda[ncoef0 - 3 + 2 - 1] = gl.Scale * tmpdyda2;
	gl.dyda[ncoef0 - 3 + 3 - 1] = gl.Scale * tmpdyda3;

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];

	/* Derivatives of br. w.r.t. cl, cls */
	gl.dyda[ncoef - 1 - 1] = gl.Scale * tmpdyda4 * cl;
	gl.dyda[ncoef - 1] = gl.Scale * tmpdyda5;

	/* Scaled brightness */
	gl.ymod *= gl.Scale;
}
