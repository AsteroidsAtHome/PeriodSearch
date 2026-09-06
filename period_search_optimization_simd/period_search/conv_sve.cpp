/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategySve.hpp"
#include "arrayHelpers.hpp"

/**
 * @brief Computes the convexity regularization function.
 *
 * This function calculates the convexity regularization function, updating the global
 * variables ymod and dyda based on the given parameters and the global data.
 *
 * @param nc An integer representing the current coefficient index.
 * @param ma An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables ymod and dyda.
 *
 * @date 8.11.2006
 */
#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::conv(const int nc, const int ma, globals &gl)
{
	const int cnt = static_cast<int>(svcntd());

	gl.ymod = 0;

	for (auto j = 1; j <= ma; j++)
	{
		gl.dyda[j] = 0;
	}

	for (auto i = 0; i < Numfac; i++)
	{
		gl.ymod += gl.Area[i] * gl.Nor[nc - 1][i];
		double *Dg_row = gl.Dg[i];
		svfloat64_t avx_Darea = svdup_n_f64(gl.Darea[i]);
		svfloat64_t avx_Nor = svdup_n_f64(gl.Nor[nc - 1][i]);

		for (auto j = 0; j < Ncoef; j += cnt)
		{
			const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(j), static_cast<int64_t>(Ncoef));
			svfloat64_t avx_dres = svld1_f64(pg, &gl.dyda[j]);
			svfloat64_t avx_Dg = svld1_f64(pg, &Dg_row[j]);

			avx_dres = svmla_f64_x(pg, avx_dres, svmul_f64_x(pg, avx_Darea, avx_Dg), avx_Nor);
			svst1_f64(pg, &gl.dyda[j], avx_dres);
		}
	}
}
