#pragma once

#include <immintrin.h>
#include "CalcStrategy.hpp"

#ifndef CSA5
#define CSA5

class alignas(64) CalcStrategyAvx512 : public CalcStrategy
{
public:
	CalcStrategyAvx512() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq);

	//virtual double bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef);
	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br);

	virtual void conv(int nc, double dres[], int ma, double &result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int &error);

private:
	__m512d* Dg_row[MAX_N_FAC + 3]{};
	__m512d dbr[MAX_N_FAC + 3]{};

	double alpha = 0.0;
	double cos_alpha = 0.0;
	double cl = 0.0;
	double cls = 0.0;
	double e[4]{};
	double e0[4]{};
	double php[N_PHOT_PAR + 1]{};
	double dphp[N_PHOT_PAR + 1]{};
	double de[4][4]{};
	double de0[4][4]{};
	double tmat[4][4]{};
	double dtm[4][4][4]{};

	int	ncoef0 = 0;
	int incl_count = 0;
};

#endif