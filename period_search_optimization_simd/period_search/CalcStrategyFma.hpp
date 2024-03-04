#pragma once

#include <immintrin.h>
#include "CalcStrategy.hpp"
#include "CalcStrategyAvx.hpp"
#include "constants.h"

#ifndef CSF
#define CSF

class alignas(64) CalcStrategyFma : public CalcStrategyAvx //public CalcStrategy
{
public:

	CalcStrategyFma() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double& trial_chisq);

	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double& br);

	virtual void conv(int nc, double dres[], int ma, double& result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int& error);

	//__m256d mm256_msub_pd(__m256d a, __m256d b, __m256d c)
	//{
	//	__m256d dst = _mm256_fmsub_pd(a, b, c);
	//}

	////TODO: Change to return result by reference:
	//// void mm256_madd_pd(__m256d a, __m256d b, __m256d c, __m256d &dst)
	//__m256d mm256_madd_pd(__m256d a, __m256d b, __m256d c)
	//{
	//	__m256d dst = _mm256_fmadd_pd(a, b, c);
	//}

private:
	__m256d* Dg_row[MAX_N_FAC + 3]{};
	__m256d dbr[MAX_N_FAC + 3]{};

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