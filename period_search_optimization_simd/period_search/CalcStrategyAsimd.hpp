#pragma once

#include "CalcStrategy.hpp"
#include "constants.h"
#include <arm_neon.h>

#ifndef CSASIMD
#define CSASIMD

class CalcStrategyAsimd : public CalcStrategy
{
public:

	CalcStrategyAsimd() {};
	~CalcStrategyAsimd() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq);

	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br);

	virtual void conv(int nc, double dres[], int ma, double &result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int &error);

private:
	float64x2_t* Dg_row[MAX_N_FAC + 3];
	float64x2_t dbr[MAX_N_FAC + 3];

	double php[N_PHOT_PAR + 1];
	double dphp[N_PHOT_PAR + 1];

	double e[4], e0[4];
	double de[4][4];
	double de0[4][4];
	double tmat[4][4];
	double dtm[4][4][4];

	double cos_alpha;
	double cl;
	double cls;
	double alpha;

	int ncoef0;
	int incl_count;
};

#endif