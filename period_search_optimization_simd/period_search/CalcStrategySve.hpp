#pragma once

#include "CalcStrategy.hpp"
#include "constants.h"
#if defined __x86_64__ || defined(__i386__) || _WIN32
  #include "sve_emulator.hpp"
#else
  #include <arm_sve.h>
#endif

#ifndef CSSVE
#define CSSVE

class CalcStrategySve : public CalcStrategy
{
public:

	CalcStrategySve() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq);

	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br);

	virtual void conv(int nc, double dres[], int ma, double &result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int &error);

private:
	int incl[MAX_N_FAC]; //array of indexes of facets to Area, Dg, Nor. !!!!!!!!!!!incl IS ZERO INDEXED

	int incl_count;
	int ncoef0;
};

#endif