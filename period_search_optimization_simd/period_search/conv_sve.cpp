/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategySve.hpp"

#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
double CalcStrategySve::conv(int nc, double dres[], int ma)
{
    int i, j;
    double res;

    res = 0;
    for (j = 1; j <= ma; j++)
        dres[j] = 0;

    size_t cnt = svcntd();
    for (i = 0; i < Numfac; i++) {
        res += Area[i] * Nor[nc - 1][i];
        double *Dg_row = Dg[i];
		svfloat64_t avx_Darea = svdup_n_f64(Darea[i]);
		svfloat64_t avx_Nor = svdup_n_f64(Nor[nc - 1][i]);
		for (j = 0; j < Ncoef; j += cnt) {
			svbool_t pg = svwhilelt_b64(j, Ncoef);
    		svfloat64_t avx_dres = svld1_f64(pg, &dres[j]);
    		svfloat64_t avx_Dg = svld1_f64(pg, &Dg_row[j]);

    		avx_dres = svmla_f64_x(pg, avx_dres, svmul_f64_x(pg, avx_Darea, avx_Dg), avx_Nor);
    		svst1_f64(pg, &dres[j], avx_dres);
		}
    }
    return res;
}
