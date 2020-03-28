#pragma once
//#include <stdio.h>
#include "constants.h"

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
extern double Nor[3][MAX_N_FAC+2] __attribute__ ((aligned (16))),
	          Area[MAX_N_FAC+2] __attribute__ ((aligned (16))),
			  Darea[MAX_N_FAC+2] __attribute__ ((aligned (16))),
			  Dg[MAX_N_FAC+4][MAX_N_PAR+10] __attribute__ ((aligned (16)));
#else
extern __declspec(align(16)) double Nor[3][MAX_N_FAC+2], Area[MAX_N_FAC+2], Darea[MAX_N_FAC+2],Dg[MAX_N_FAC+4][MAX_N_PAR+10]; //All are zero indexed
#endif