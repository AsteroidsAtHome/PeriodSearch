#pragma once

#include "CalcStrategy.hpp"
#include <arm_neon.h>

#ifndef CSASIMD
#define CSASIMD

class CalcStrategyAsimd : public CalcStrategy
{
public:

	CalcStrategyAsimd() {};

	virtual double mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma);

	virtual double bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef);

	virtual double conv(int nc, double dres[], int ma);

	virtual void curv(double cg[]);

	virtual int gauss_errc(double** a, int n, double b[]);
};

#endif