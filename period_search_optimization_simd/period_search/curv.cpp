/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

void CalcStrategyNone::curv(double cg[])
{
	int i, m, l, k;
	int n;
	double fsum, g;

	for (i = 1; i <= Numfac; i++)
	{
		g = 0;
		n = 0;
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
		{
			for (l = m; l <= Lmax; l++)
			{
				double fsum;
				n++;
				fsum = cg[n] * Fc[i][m];
				n++;
				fsum = fsum + cg[n] * Fs[i][m];
				g = g + Pleg[i][l][m] * fsum;
			}
		}


		// NOTE: Faster execution path not using condition expression "if (m != 0)"
		//m=0

		g = exp(g);
		printf("[%d] g: %0.6f\n", i, g);

		//Area[i] = Darea[i] * g;
		Area[i - 1] = Darea[i - 1] * g;

		for (k = 1; k <= n; k++)
		{
			//Dg[i][k] = g * Dsph[i][k];
			Dg[i - 1][k - 1] = g * Dsph[i][k];

			//if (i == 101 && k == 50)
			//{
			//	printf("% 0.6f\n", Dg[100][49]);
			//}
		}
	}

	//printArray(Area, i - 1, "Area");
	//printf("g: % 0.6f\n", g);
	//printArray(cg, 49, "cg");

	/*printf("Dsph[%d][%d]:\n", i, n);
	for (int q = 0; q <= n; q++)
	{
		printf("_Dsph_%d[] = { ", q);
		for (int p = 0; p <= i; p++)
		{
			printf("% 0.6f, ", Dsph[p][q]);
			if (p % 9 == 0)
				printf("\n");
		}
		printf("};\n");
	}*/
}
