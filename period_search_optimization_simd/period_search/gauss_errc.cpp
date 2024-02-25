/* from Numerical Recipes */

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "declarations.h"
#include "CalcStrategyNone.hpp"

int CalcStrategyNone::gauss_errc(double** a, int n, double b[])
{
	int *indxc, *indxr, *ipiv;
	int i, icol = 0, irow = 0, j, k, l, ll;
	double big, dum, pivinv, temp;

	indxc = vector_int(n + 1);
	indxr = vector_int(n + 1);
	ipiv = vector_int(n + 1);

	//memset(ipiv + 1, 0, n * sizeof(int));
	memset(ipiv, 0, n * sizeof(int));

	for (i = 1; i <= n; i++) {
		big = 0.0;
		for (j = 0; j < n; j++) //* 1 -> 0
		{
			if (ipiv[j] != 1)
			{
				for (k = 0; k < n; k++) {//* 1 -> 0
					//if(j==50 && k == 50)
					//	printf("big % 0.6f; a[%3d][%3d] % 0.6f\n", big, j, k, a[j][k]);
					if (ipiv[k] == 0) {
						if (fabs(a[j][k]) >= big) {
							//printf("big % 0.6f; a[%3d][%3d] % 0.6f\n", big, j, k, a[j][k]);
							big = fabs(a[j][k]);
							irow = j;
							icol = k;
						}
					}
					else if (ipiv[k] > 1) {
						deallocate_vector((void*)ipiv);
						deallocate_vector((void*)indxc);
						deallocate_vector((void*)indxr);
						return(1);
					}
				}
			}
		}

		++(ipiv[icol]);
		if (irow != icol) {
			for (l = 0; l < n; l++) SWAP(a[irow][l], a[icol][l])
				SWAP(b[irow], b[icol])
		}
		indxr[i] = irow;
		indxc[i] = icol;
		//printf("i[%3d] %3d", i, icol);
		//printf("\n");
		if (a[icol][icol] == 0.0)
		{
			deallocate_vector((void*)ipiv);
			deallocate_vector((void*)indxc);
			deallocate_vector((void*)indxr);
			return(2);
		}
		pivinv = 1.0 / a[icol][icol];

		//printf("i[%3d] %3d % 0.6f\n", i, icol, a[icol][icol]);

		a[icol][icol] = 1.0;

		for (l = 0; l < n; l++)
		{
			a[icol][l] *= pivinv;
		}

		b[icol] *= pivinv;
		for (ll = 0; ll < n; ll++)
		{
			if (ll != icol)
			{
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				for (l = 0; l < n; l++)
				{
					a[ll][l] -= a[icol][l] * dum;
				}

				b[ll] -= b[icol] * dum;
			}
		}
	}

	for (l = n; l >= 1; l--)
	{
		if (indxr[l] != indxc[l])
			for (k = 0; k < n; k++)
			{
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
			}
	}

	deallocate_vector((void*)ipiv);
	deallocate_vector((void*)indxc);
	deallocate_vector((void*)indxr);

	return(0);
}
#undef SWAP
