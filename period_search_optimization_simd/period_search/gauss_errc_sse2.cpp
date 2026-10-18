#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdlib>
#include <vector>
//#include <string.h>
#include "declarations.h"

#if !defined __APPLE__
#include <malloc.h>
#endif

#include <emmintrin.h>
#include "CalcStrategySse2.hpp"


#if defined(__GNUC__)
__attribute__((target("sse2")))
#endif

/**
* @brief Solves a linear system of equations using Gaussian elimination with partial pivoting.
*
* This function implements the Gaussian elimination algorithm with partial pivoting to solve a
* linear system of equations. It rearranges the covariance matrix and the right-hand side vector
* to find the solution.
*
* @param gl A reference to a globals structure containing the covariance matrix and other global data.
* @param n The dimension of the system (number of equations/variables).
* @param b A vector of doubles representing the right-hand side vector of the system.
* @param error An integer reference to store error codes:
*              - 0: No error
*              - 1: Singular matrix
*              - 2: Zero pivot element
*
* @note The function modifies the covariance matrix `covar` in place.
*
* @source Numerical Recipes
*
* @date 8.11.2006
*/
void CalcStrategySse2::gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int &error)
{
	//int * indxc, * indxr;
	//short* ipiv;
	int i, icol = 0, irow = 0, j, k, l, ll, ipivsize;
	double big, dum, pivinv, temp;

	auto& a = gl.covar;

	//indxc = vector_int(n + 1);
	std::vector<int> indxc(n + 1 + 1, 0);
	//indxr = vector_int(n + 1);
	std::vector<int> indxr(n + 1 + 1, 0);

	ipivsize = (n >> 3) << 3;
	if (n % 8) ipivsize += 8;

//#if !defined _WIN32 && !defined __APPLE__
//	ipiv = (short*)memalign(16, ipivsize * sizeof(short)); //is zero indexed
//#elif defined __APPLE__
//	posix_memalign((void**)&ipiv, 16, ipivsize * sizeof(short));
//#else
//	ipiv = (short*)_aligned_malloc(ipivsize * sizeof(short), 16); //is zero indexed
//#endif

#if defined __GNUC__ || (defined(_MSC_VER) && _MSC_VER >= 1900)
	std::vector<short, AlignedAllocatorNew<short, 16>> ipiv(ipivsize, 0); // is zero indexed
#else
	std::vector<short> ipiv(ipivsize, 0); // is zero indexed (WARNING: data not guaranteed 16-byte aligned)
#endif

	__m128i avx_zeros = _mm_setzero_si128();

	//memset(ipiv, 0, n * sizeof(short));


	for (j = n; j < ipivsize; j++) ipiv[j] = 1;

	for (i = 1; i <= n; i++)
	{
		big = 0.0;
		for (j = 0; j < n; j++)
		{
			if (ipiv[j] != 1)
			{
				for (k = 0; k < n; k += 8)
				{
					__m128i avx_ipiv = _mm_load_si128((__m128i*) & ipiv[k]);
					__m128i avx_iserror = _mm_cmpgt_epi16(avx_ipiv, _mm_set1_epi16(1));
					int ria = _mm_movemask_epi8(avx_iserror);

					if (ria)
					{
//#if !defined _WIN32
//						free(ipiv);
//#else
//						_aligned_free(ipiv);
//#endif
						//deallocate_vector((void*)indxc);
						//deallocate_vector((void*)indxr);
						error = 1;

						return;
					}

					__m128i avx_iszero = _mm_cmpeq_epi16(avx_ipiv, avx_zeros);
					ria = _mm_movemask_epi8(avx_iszero);

					if (ria & 1)
					{
						if (fabs(a[j][k]) >= big)
						{
							big = fabs(a[j][k]);
							irow = j;
							icol = k;
						}

					}
					if (ria & 4)
					{
						if (fabs(a[j][k + 1]) >= big)
						{
							big = fabs(a[j][k + 1]);
							irow = j;
							icol = k + 1;
						}

					}
					if (ria & 16)
					{
						if (fabs(a[j][k + 2]) >= big)
						{
							big = fabs(a[j][k + 2]);
							irow = j;
							icol = k + 2;
						}

					}
					if (ria & 64)
					{
						if (fabs(a[j][k + 3]) >= big)
						{
							big = fabs(a[j][k + 3]);
							irow = j;
							icol = k + 3;
						}
					}
					if (ria & 256)
					{
						if (fabs(a[j][k + 4]) >= big)
						{
							big = fabs(a[j][k + 4]);
							irow = j;
							icol = k + 4;
						}
					}
					if (ria & 1024)
					{
						if (fabs(a[j][k + 5]) >= big)
						{
							big = fabs(a[j][k + 5]);
							irow = j;
							icol = k + 5;
						}
					}
					if (ria & 4096)
					{
						if (fabs(a[j][k + 6]) >= big)
						{
							big = fabs(a[j][k + 6]);
							irow = j;
							icol = k + 6;
						}
					}
					if (ria & 16384)
					{
						if (fabs(a[j][k + 7]) >= big)
						{
							big = fabs(a[j][k + 7]);
							irow = j;
							icol = k + 7;
						}
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

		if (a[icol][icol] == 0.0)
		{

//#if !defined _WIN32
//			free(ipiv);
//#else
//			_aligned_free(ipiv);
//#endif
			//deallocate_vector((void*)indxc);
			//deallocate_vector((void*)indxr);
			error = 2;

			return;
		}

		pivinv = 1.0 / a[icol][icol];
		__m128d avx_pivinv;
		avx_pivinv = _mm_set1_pd(pivinv);
		a[icol][icol] = 1.0;

		for (l = 0; l < (n - 1); l += 2)
		{
			__m128d avx_a1 = _mm_load_pd(&a[icol][l]);
			avx_a1 = _mm_mul_pd(avx_a1, avx_pivinv);
			_mm_store_pd(&a[icol][l], avx_a1);
		}

		if (l == (n - 1)) a[icol][l] *= pivinv; //last odd value

		b[icol] *= pivinv;
		for (ll = 0; ll < n; ll++) {
			if (ll != icol) {
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				__m128d avx_dum;
				avx_dum = _mm_set1_pd(dum);

				for (l = 0; l < (n - 1); l += 2)
				{
					__m128d avx_a = _mm_load_pd(&a[ll][l]), avx_aa = _mm_load_pd(&a[icol][l]);
					avx_a = _mm_sub_pd(avx_a, _mm_mul_pd(avx_aa, avx_dum));
					_mm_store_pd(&a[ll][l], avx_a);
				}

				if (l == (n - 1)) a[ll][l] -= a[icol][l] * dum; //last odd value

				b[ll] -= b[icol] * dum;
			}
		}
	}

	for (l = n; l >= 1; l--)
	{
		if (indxr[l] != indxc[l])
			for (k = 0; k < n; k++)
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
	}

//#if !defined _WIN32
//	free(ipiv);
//#else
//	_aligned_free(ipiv);
//#endif

	//deallocate_vector((void*)indxc);
	//deallocate_vector((void*)indxr);
	error = 0;

	return;
}
#undef SWAP
