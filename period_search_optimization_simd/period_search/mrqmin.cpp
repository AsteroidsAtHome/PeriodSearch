// ReSharper disable IdentifierTypo
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "arrayHelpers.hpp"

/**
 * @brief Performs a nonlinear least-squares fit using the Marquardt method.
 *
 * This function implements the Marquardt method for nonlinear least-squares fitting.
 * It adjusts the parameters to minimize the chi-squared value, based on the given data and initial parameters.
 *
 * @param x1 A reference to a 2D vector of doubles representing the independent variable data.
 * @param x2 A reference to a 2D vector of doubles representing additional independent variable data.
 * @param x3 A reference to a vector of doubles representing additional data points.
 * @param y A reference to a vector of doubles representing the dependent variable data.
 * @param sig A reference to a vector of doubles representing the standard deviations of the data points.
 * @param a A reference to a vector of doubles representing the initial parameters.
 * @param ia A reference to a vector of integers indicating which parameters are to be fitted.
 * @param ma An integer representing the total number of parameters.
 * @param gl The per-thread globals instance; the fit state (Chisq, Ochisq, Alamda,
 *        Niter, Lastcall, Deallocate, atry, beta, da, alpha, covar) and the
 *        calculation strategy (ctx) are stored there so that each OpenMP worker
 *        thread keeps its own LM state.
 *
 * @return An integer error code:
 *         - 0: No error
 *         - Non-zero: Indicates an error occurred during fitting
 *
 * @note The fit state is modified in `gl`. Converted from Mikko's Fortran code.
 *
 * @source Numerical Recipes: Nonlinear least-squares fit, Marquardt’s method.
 *
 * @date 8.11.2006
 */
int mrqmin(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
		std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, const int ma, globals& gl)
{
	int j, k, l, err_code;
	// mfit/lastone/lastma are derived from ma and ia (both constant for a whole
	// fit). In the serial code they were function-statics, so the values
	// computed on the first call persisted into every later call. Function-
	// statics are unsafe across threads, so they are recomputed on EVERY call
	// instead (O(ma), negligible next to the fit). Recomputing them on every
	// call - not only the first - is essential: the covariance build, the Gauss
	// solve and both mrqcof calls all rely on them being the true values.
	int mfit, lastone, lastma;

	//static double* beta, * da, * atry; //beta, da are zero indexed

	double trial_chisq;
	//double** alpha, ** covar; // Dummy placeholders

	auto& ctx = gl.ctx; // per-thread calculation strategy

	/* deallocates memory when used in period_search */
	if (gl.Deallocate == 1)
	{
		//deallocate_vector((void*)atry);
		//deallocate_vector((void*)beta);
		//deallocate_vector((void*)da);
		return(0);
	}

	/* number of fitted parameters (valid for every call of a fit) */
	mfit = 0;
	lastma = 0;
	for (j = 0; j < ma; j++)
	{
		if (ia[j])
		{
			mfit++;
			lastma = j;
		}
	}

	lastone = 0;
	for (j = 1; j <= lastma; j++) //ia[0] is skipped because ia[0]=0 is acceptable inside mrqcof
	{
		if (!ia[j]) break;
		lastone = j;
	}

	if (gl.Lastcall != 1)
	{
		if (gl.Alamda < 0)
		{
			//atry = vector_double(ma);
			//beta = vector_double(ma);
			//da = vector_double(ma);

			std::fill(gl.atry.begin(), gl.atry.end(), 0.0);
			std::fill(gl.beta.begin(), gl.beta.end(), 0.0);
			std::fill(gl.da.begin(), gl.da.end(), 0.0);

			gl.Alamda = Alamda_start; /* initial alambda */

			//gl.isCovar = false;
			// NOTE: Use gl.alpha
			ctx.CalculateMrqcof(x1, x2, x3, y, sig, a, ia, ma, gl.beta, mfit, lastone, lastma, trial_chisq, gl, false);

			gl.Ochisq = trial_chisq;
			for (j = 1; j <= ma; j++)
			{
				gl.atry[j] = a[j];
			}
		}

		for (j = 0; j < mfit; j++)
		{
			for (k = 0; k < mfit; k++)
			{
				gl.covar[j][k] = gl.alpha[j][k];
			}

			gl.covar[j][j] = gl.alpha[j][j] * (1 + gl.Alamda);
			gl.da[j] = gl.beta[j];
		}

		//calcCtx.CalculateGaussErrc(covar, mfit, da.data(), err_code);

		//auto flat2DCover = flatten2Dvector(gl.covar);
		//// Create double** from the flattened vector
		//auto rows = gl.covar.size();
		//auto cols = gl.covar[0].size();
		//double** flatCovar = new double*[rows];
	 //   for (size_t i = 0; i < rows; ++i)
	 //   {
		//	flatCovar[i] = &flat2DCover[i * cols];
	 //   }

		//calcCtx.CalculateGaussErrc(flatCovar, mfit, da.data(), err_code);
		ctx.CalculateGaussErrc(gl, mfit, gl.da, err_code);

		if (err_code != 0) return(err_code);

		j = 0;
		for (l = 1; l <= ma; l++)
		{
			if (ia[l - 1])
			{
				gl.atry[l] = a[l] + gl.da[j];
				j++;
			}
		}
	} /* Lastcall != 1 */

	if (gl.Lastcall == 1)
	{
		for (l = 1; l <= ma; l++)
		{
			gl.atry[l] = a[l];
		}
	}

	//gl.isCovar = true;
	// NOTE: Use gl.covar
	ctx.CalculateMrqcof(x1, x2, x3, y, sig, gl.atry, ia, ma, gl.da, mfit, lastone, lastma, trial_chisq, gl, true);

	gl.Chisq = trial_chisq;

	if (gl.Lastcall == 1)
	{
		//deallocate_vector((void*)atry);
		//deallocate_vector((void*)beta);
		//deallocate_vector((void*)da);
		return(0);
	}

	if (trial_chisq < gl.Ochisq)
	{
		gl.Alamda = gl.Alamda / Alamda_incr;
		for (j = 0; j < mfit; j++)
		{
			for (k = 0; k < mfit; k++)
			{
				gl.alpha[j][k] = gl.covar[j][k];
			}
			gl.beta[j] = gl.da[j];
		}

		for (l = 1; l <= ma; l++)
		{
			a[l] = gl.atry[l];
		}
	}
	else
	{
		gl.Alamda = Alamda_incr * gl.Alamda;
		gl.Chisq = gl.Ochisq;
	}

	return(0);
}

