/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <cmath>
#include <vector>
#include "globals.h"
#include "constants.h"
#include "CalcStrategySve.hpp"
#include "arrayHelpers.hpp"

/**
 * @brief Computes the curvature function and facet area from the Laplace series.
 *
 * This function calculates the curvature function and hence the facet area based on the
 * Laplace series using the provided coefficients and global data. The results are stored
 * in the global variables.
 *
 * @param cg A reference to a vector of doubles containing the coefficients for the Laplace series.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables Area and Dg.
 *
 * @date 8.11.2006
 */
#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::curv(std::vector<double>& cg, globals &gl)
{
	const int cnt = static_cast<int>(svcntd());

	for (auto i = 1; i <= Numfac; i++)
	{
		double g = 0;
		int n = 0;
		// m=0
		for (auto l = 0; l <= Lmax; l++)
		{
			n++;
			const double fsum = cg[n] * Fc[i][0];
			g += Pleg[i][l][0] * fsum;
		}
		//
		for (auto m = 1; m <= Mmax; m++)
		{
			for (auto l = m; l <= Lmax; l++)
			{
				n++;
				double fsum = cg[n] * Fc[i][m];
				n++;
				fsum += cg[n] * Fs[i][m];
				g += Pleg[i][l][m] * fsum;
			}
		}

		g = exp(g);
		gl.Area[i - 1] = gl.Darea[i - 1] * g;

		svfloat64_t avx_g = svdup_n_f64(g);

		for (auto k = 1; k <= n; k += cnt)
		{
			const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(k), static_cast<int64_t>(n + 1));
			svfloat64_t avx_pom = svld1_f64(pg, &Dsph[i][k]);
			avx_pom = svmul_f64_x(pg, avx_pom, avx_g);
			svst1_f64(pg, &gl.Dg[i - 1][k - 1], avx_pom);
		}
	}
}
