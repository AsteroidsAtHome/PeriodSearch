#pragma once
#include <stdio.h>

#include "constants.h"

extern int Lmax, Mmax, Niter, Lastcall,
Ncoef, Numfac, Lcurves, Nphpar,
Lpoints[MAX_LC + 1], Inrel[MAX_LC + 1],
Deallocate;

extern double Ochisq, Chisq, _Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale, _scale,
Sclnw[MAX_LC + 1],
Yout[MAX_N_OBS + 1],
Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1], _dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Blmat[4][4], _blmat[4][4],
Xx1[4], Xx2[4], _xx1[4], _xx2[4],
tmat[4][4], dtm[4][4][4], _tmat[4][4], _dtm[4][4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1], _pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4], _dblm[3][4][4],
Weight[MAX_N_OBS + 1],
php[N_PHOT_PAR + 1], dphp[N_PHOT_PAR + 1], _dphp[N_PHOT_PAR + 1],
Dyda[MAX_N_PAR + 8], _dyda[MAX_N_PAR + 8];


#ifdef __GNUC__
extern double Nor[3][MAX_N_FAC + 4] __attribute__((aligned(32))),
Area[MAX_N_FAC + 4] __attribute__((aligned(32))),
Darea[MAX_N_FAC + 4] __attribute__((aligned(32))),
Dg[MAX_N_FAC + 8][MAX_N_PAR + 4] __attribute__((aligned(32)));
#else
extern __declspec(align(32)) double Nor[3][MAX_N_FAC + 4], Area[MAX_N_FAC + 4], Darea[MAX_N_FAC + 4], Dg[MAX_N_FAC + 8][MAX_N_PAR + 4], //All are zero indexed
_nor[3][MAX_N_FAC + 4], _area[MAX_N_FAC + 4], _darea[MAX_N_FAC + 4], _dg[MAX_N_FAC + 8][MAX_N_PAR + 4];
#endif





