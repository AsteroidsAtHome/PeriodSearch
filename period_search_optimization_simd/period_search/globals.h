// ReSharper disable IdentifierTypo
#pragma once
#include <cstdio>
#include <vector>
#include "constants.h"
#include "CalcStrategy.hpp"

extern int Lmax, Mmax, Niter, Lastcall,
	Ncoef, Numfac, Nphpar,
	Deallocate;

extern double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,

Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Blmat[4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4];

extern std::vector<double> atry;
extern std::vector<double> beta;
extern std::vector<double> da;

extern CalcContext calcCtx;

extern struct SIMDSupport
{
	bool hasAVX512dq = false;
	bool hasAVX512 = false;
	bool hasFMA = false;
	bool hasAVX = false;
	bool hasSSE3 = false;
	bool hasSSE2 = false;
	bool hasASIMD = false;
	bool hasSVE = false;
	bool isBulldozer = false;
} CPUopt;