#pragma once

#include "CalcStrategy.hpp"
#include "constants.h"

#ifndef CSNO
#define CSNO

class CalcStrategyNone : public CalcStrategy
{
public:
#if defined _WIN32
#pragma warning(disable:26495)
#endif

	CalcStrategyNone() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq);

	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br);

	virtual void conv(int nc, double dres[], int ma, double &result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int &error);

private:
	double php[N_PHOT_PAR + 1];
	double dphp[N_PHOT_PAR + 1];
	double dbr[MAX_N_FAC]; //IS ZERO INDEXED
	double e[4];
	double e0[4];
	double de[4][4];
	double de0[4][4];
	double tmat[4][4];
	double dtm[4][4][4];

	double cos_alpha;
	double alpha;
	double cl;
	double cls;
	double dnom;
	double tmpdyda;
	double s;

	double tmpdyda1;
	double tmpdyda2;
	double tmpdyda3;
	double tmpdyda4;
	double tmpdyda5;

	int	incl[MAX_N_FAC]; //array of indexes of facets to Area, Dg, Nor. !!!!!!!!!!!incl IS ZERO INDEXED
	int ncoef0;
	int incl_count = 0;
};

#endif