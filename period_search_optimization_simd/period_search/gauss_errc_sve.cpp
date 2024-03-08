/* from Numerical Recipes */

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "declarations.h"
#include "CalcStrategySve.hpp"

#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
__attribute__((__target__("+sve")))
#endif
void CalcStrategySve::gauss_errc(double** a, int n, double b[], int &error)
{
	int *indxc, *indxr, *ipiv;
	int i, icol = 0, irow = 0, j, k, l, ll;
	double big, dum, pivinv, temp;

	indxc = vector_int(n + 1);
	indxr = vector_int(n + 1);
	ipiv = vector_int(n + 1);

	int cnt = svcntd();

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
						//return(1);
						error = 1;
						return;
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
			//return(2);
			error = 2;
			return;
		}
		pivinv = 1.0 / a[icol][icol];

		//printf("i[%3d] %3d % 0.6f\n", i, icol, a[icol][icol]);

		svfloat64_t avx_pivinv = svdup_n_f64(pivinv);
		a[icol][icol] = 1.0;

		for (l = 0; l < n; l += cnt) {
			svbool_t pg = svwhilelt_b64(l, n);
    		svfloat64_t avx_a1 = svld1_f64(pg, &a[icol][l]);
    		avx_a1 = svmul_f64_x(pg, avx_a1, avx_pivinv);
    		svst1_f64(pg, &a[icol][l], avx_a1);
		}

		b[icol] *= pivinv;
		for (ll = 0; ll < n; ll++)
		{
			if (ll != icol)
			{
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				svfloat64_t avx_dum = svdup_n_f64(dum);

				for (l = 0; l < n; l += cnt) {
					svbool_t pg = svwhilelt_b64(l, n);
    				svfloat64_t avx_a = svld1_f64(pg, &a[ll][l]);
    				svfloat64_t avx_aa = svld1_f64(pg, &a[icol][l]);
    				svfloat64_t avx_result = svmls_f64_x(pg, avx_a, avx_aa, avx_dum);
    				svst1_f64(pg, &a[ll][l], avx_result);
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

	//return(0);
	error = 0;
	return;
}
#undef SWAP
