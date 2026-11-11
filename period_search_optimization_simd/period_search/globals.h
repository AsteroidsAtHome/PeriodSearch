// ReSharper disable IdentifierTypo
#pragma once
#include <cstdio>
#include <vector>
#include "constants.h"
#include "CalcStrategy.hpp"

/* Shared read-only fit parameters and tables.
   Per-fit mutable state (Chisq, Ochisq, Alamda, Niter, Lastcall, Deallocate,
   Scale, Blmat, Dblm, atry, beta, da, the calc strategy ctx) is now a member
   of the globals struct (arrayHelpers.hpp) so each worker thread gets its own
   copy. */
extern int Lmax, Mmax,
Ncoef, Numfac, Nphpar;

extern double Alamda_incr, Alamda_start, Phi_0,

Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];

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