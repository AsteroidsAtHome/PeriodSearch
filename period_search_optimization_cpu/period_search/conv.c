/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"

double conv(int nc, double dres[], int ma)
{
	int i, j;

	double res;

	res = 0;
	for (j = 1; j <= ma; j++)
		dres[j] = 0;

	for (i = 1; i <= Numfac; i++)
	{
		res += Area[i] * Nor[i][nc];
		for (j = 1; j <= Ncoef; j++)
			dres[j] += Darea[i] * Dg[i][j] * Nor[i][nc];
	}

	return(res);
}
