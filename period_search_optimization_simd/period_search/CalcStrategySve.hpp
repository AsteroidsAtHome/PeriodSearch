#pragma once

#include "CalcStrategy.hpp"
#include <arm_sve.h>

#ifndef CSSVE
#define CSSVE

class CalcStrategySve : public CalcStrategy
{
public:

	CalcStrategySve() {};

	virtual double mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma);

	virtual double bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef);

	virtual double conv(int nc, double dres[], int ma);

	virtual void curv(double cg[]);

	virtual int gauss_errc(double** a, int n, double b[]);
};

#endif