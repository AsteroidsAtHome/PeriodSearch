#pragma once

#include <immintrin.h>
#include "CalcStrategy.hpp"

#ifndef CSA
#define CSA


class CalcStrategyAvx : public CalcStrategy
{
public:

	CalcStrategyAvx() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq);

	//virtual double bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef);
	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br);

	virtual void conv(int nc, double dres[], int ma, double &result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int &error);

	__m256d mm256_msub_pd(__m256d a, __m256d b, __m256d c)
	{
		__m256d dst = _mm256_sub_pd(_mm256_mul_pd(a, b), c);

		return dst;
	}

	__m256d mm256_madd_pd(__m256d a, __m256d b, __m256d c)
	{
		__m256d dst = _mm256_add_pd(_mm256_mul_pd(a, b), c);

		return dst;
	}
};

#endif