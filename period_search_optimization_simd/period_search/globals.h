#pragma once
#include <cstdio>

#include "constants.h"
#include "CalcStrategy.hpp"

extern int Lmax, Mmax, Niter, Lastcall,
           Ncoef, Numfac, Lcurves, Nphpar,
           Lpoints[MAX_LC+1], Inrel[MAX_LC+1],
	   Deallocate;

extern double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,
               Sclnw[MAX_LC+1],
	      Yout[MAX_N_OBS+1],
              Fc[MAX_N_FAC+1][MAX_LM+1], Fs[MAX_N_FAC+1][MAX_LM+1],
	      Tc[MAX_N_FAC+1][MAX_LM+1], Ts[MAX_N_FAC+1][MAX_LM+1],
	       Dsph[MAX_N_FAC+1][MAX_N_PAR+1],
               Blmat[4][4],
              Pleg[MAX_N_FAC+1][MAX_LM+1][MAX_LM+1],
              Dblm[3][4][4],
	      Weight[MAX_N_OBS+1];

#ifdef __GNUC__
  extern double Nor[3][MAX_N_FAC+8] __attribute__ ((aligned (64))),
	          Area[MAX_N_FAC+8] __attribute__ ((aligned (64))),
			  Darea[MAX_N_FAC+8] __attribute__ ((aligned (64))),
			  Dg[MAX_N_FAC+16][MAX_N_PAR+8] __attribute__ ((aligned (64)));
#else
  extern __declspec(align(64)) double Nor[3][MAX_N_FAC+8], Area[MAX_N_FAC+8], Darea[MAX_N_FAC+8],Dg[MAX_N_FAC+16][MAX_N_PAR+8]; //All are zero indexed
#endif

//extern CalcContext caclContext(std::unique_ptr<CalcStrategy>());
	extern CalcContext calcCtx;

#ifdef __GNUC__
	extern double dyda[MAX_N_PAR + 16] __attribute__((aligned(64)));
#else
	extern __declspec(align(64)) double dyda[MAX_N_PAR + 16]; //is zero indexed for aligned memory access
#endif

	extern double xx1[4], xx2[4], dy, sig2i, wt, ymod,
		ytemp[MAX_LC_POINTS + 1], dytemp[MAX_LC_POINTS + 1][MAX_N_PAR + 1 + 4],
		dave[MAX_N_PAR + 1 + 4],
		dave2[MAX_N_PAR + 1 + 4],
		coef, ave, trial_chisq, wght;

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
	} CPUopt;