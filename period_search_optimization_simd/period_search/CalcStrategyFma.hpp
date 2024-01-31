#pragma once

#include <immintrin.h>
#include "CalcStrategy.hpp"
#include "CalcStrategyAvx.hpp"

#ifndef CSF
#define CSF

class CalcStrategyFma : public CalcStrategyAvx //public CalcStrategy
{
public:

	CalcStrategyFma() {};

	virtual double mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma);

	virtual double bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef);

	virtual double conv(int nc, double dres[], int ma);

	virtual void curv(double cg[]);

	virtual int gauss_errc(double** a, int n, double b[]);

	__m256d mm256_msub_pd(__m256d a, __m256d b, __m256d c)
	{
		__m256d dst = _mm256_fmsub_pd(a, b, c);
	}

	__m256d mm256_madd_pd(__m256d a, __m256d b, __m256d c)
	{
		__m256d dst = _mm256_fmadd_pd(a, b, c);
	}
};

#endif