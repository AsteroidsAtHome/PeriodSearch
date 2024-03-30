/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategyNone.hpp"

void CalcStrategyNone::conv(int nc, double dres[], int ma, double &result)
{
	int i, j;

	result = 0;
	for (j = 1; j <= ma; j++)
		dres[j] = 0;

	for (i = 0; i < Numfac; i++)
	{
		result += Area[i] * Nor[nc - 1][i];

		for (j = 0; j < Ncoef; j++)
		{
			dres[j] += Darea[i] * Dg[i][j] * Nor[nc - 1][i];
		}
	}
}
