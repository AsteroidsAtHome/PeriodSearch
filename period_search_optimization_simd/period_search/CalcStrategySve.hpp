#pragma once

#include "CalcStrategy.hpp"
#include "constants.h"
#include "arrayHelpers.hpp"

#if defined(__aarch64__)
  #include <arm_sve.h>
#else
  #include "sve_emulator.hpp"
#endif

#ifndef CSSVE
#define CSSVE

// Sizeless SVE types may not be used as class members, array elements or pointer targets,
// so everything that has to survive between the loops below is kept as plain doubles and
// the vectors are rebuilt where they are needed. 2048 bits (32 doubles) is the
// architectural maximum vector length, which bounds every scratch buffer.
#define SVE_MAX_LANES 32

class CalcStrategySve final : public CalcStrategy
{
public:

	CalcStrategySve() = default;

	void mrqcof(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
		std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, int ma,
		std::vector<double>& beta, int mfit, int lastone, int lastma, double& trial_chisq, globals& gl, const bool isCovar) override;

	void bright(double t, std::vector<double>& cg, int ncoef, globals &gl) override;

	void conv(int nc, int ma, globals &gl) override;

	void curv(std::vector<double>& cg, globals &gl) override;

	void gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int &error) override;

private:
	double* Dg_row[MAX_N_FAC + 3]{};	// row of Dg belonging to the i-th included facet
	double dbr[MAX_N_FAC + 3]{};		// its brightness derivative, broadcast on use

	double php[N_PHOT_PAR + 1]{};
	double dphp[N_PHOT_PAR + 1]{};

	double e[4]{};
	double e0[4]{};
	double de[4][4]{};
	double de0[4][4]{};
	double tmat[4][4]{};
	double dtm[4][4][4]{};

	double cos_alpha = 0.0;
	double alpha = 0.0;
	double cl = 0.0;
	double cls = 0.0;

	int ncoef0 = 0;
	int incl_count = 0;
};

#endif
