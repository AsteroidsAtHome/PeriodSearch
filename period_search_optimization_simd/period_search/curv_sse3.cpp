/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <cmath>
#include "globals.h"
#include "constants.h"
#include <pmmintrin.h>
#include "CalcStrategySse3.hpp"

#if defined(__GNUC__)
__attribute__((target("sse3")))
#endif
void CalcStrategySse3::curv(double cg[])
{
	int i, m, l, k;

	for (i = 1; i <= Numfac; i++)
	{
		double g = 0;
		int n = 0;
		//m=0
		for (l = 0; l <= Lmax; l++)
		{
			double fsum;
			n++;
			fsum = cg[n] * Fc[i][0];
			g = g + Pleg[i][l][0] * fsum;
		}
		//
		for (m = 1; m <= Mmax; m++)
			for (l = m; l <= Lmax; l++)
			{
				double fsum;
				n++;
				fsum = cg[n] * Fc[i][m];
				n++;
				fsum = fsum + cg[n] * Fs[i][m];
				g = g + Pleg[i][l][m] * fsum;
			}
		g = exp(g);
		Area[i - 1] = Darea[i - 1] * g;

		__m128d avx_g = _mm_set1_pd(g);
		for (k = 1; k < n; k += 2)
		{
			__m128d avx_pom = _mm_loadu_pd(&Dsph[i][k]);
			avx_pom = _mm_mul_pd(avx_pom, avx_g);
			_mm_store_pd(&Dg[i - 1][k - 1], avx_pom);
		}
		if (k == n) Dg[i - 1][k - 1] = g * Dsph[i][k]; //last odd value

	}
}