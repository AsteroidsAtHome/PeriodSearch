#pragma once

#include "CalcStrategy.hpp"
#include "constants.h"
#include <arm_neon.h>
#include "arrayHelpers.hpp"

#ifndef CSASIMD
#define CSASIMD

class CalcStrategyAsimd final : public CalcStrategy
{
public:

	CalcStrategyAsimd() = default;

	void mrqcof(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
		std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, int ma,
		std::vector<double>& beta, int mfit, int lastone, int lastma, double& trial_chisq, globals& gl, const bool isCovar) override;

	void bright(double t, std::vector<double>& cg, int ncoef, globals &gl) override;

	void conv(int nc, int ma, globals &gl) override;

	void curv(std::vector<double>& cg, globals &gl) override;

	void gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int &error) override;

private:
	float64x2_t* Dg_row[MAX_N_FAC + 4 + DG_PREFETCH_ROWS];
	double dbr[MAX_N_FAC + 4];

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