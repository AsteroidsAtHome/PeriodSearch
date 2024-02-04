/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include "arrayHelpers.hpp"

void curv(double cg[])
{
	int i, m, n, l, k;

	double fsum, g;

	for (i = 1; i <= Numfac; i++)
	{
		g = 0;
		n = 0;
		//printf("[%3d] ", i);

		for (m = 0; m <= Mmax; m++)
		{
			for (l = m; l <= Lmax; l++)
			{
				n++;
				fsum = cg[n] * Fc[i][m];
				if (m != 0)
				{
					n++;
					double t = cg[n] * Fs[i][m];
					//printf("% 0.6f ", t);
					fsum = fsum + t;

				}
				//printf("% 0.6f ", Pleg[i][l][m]);
				g = g + Pleg[i][l][m] * fsum;

				//if (m != 0)
				//	printf("[%d] %0.6f  ", l, g);

			}

			//printf("[%3d][%3d] % 0.6f\n", i, m, Fc[i][m]);
			//printf("[%3d][%3d] % 0.6f\n", i, m, Fs[i][m]);
			//if (m != 0) printf("\n");
		}

		//printf("\n");

		g = exp(g);

		//printf("[%d] g: %0.6f\n", i, g);
		//printf("[%3d][0] % 0.6f\n", i, Fc[i][0]);

		Area[i] = Darea[i] * g;
		for (k = 1; k <= n; k++)
			Dg[i][k] = g * Dsph[i][k];

	}

	//printArray(Area, i, "Area");
	//printf("g: % 0.6f\n", g);

	printf("Dsph[%d][%d]:\n", i, n);
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
	}
}
