#pragma once
#include "../period_search/constants.h"

extern double _dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];

void get_t(double t[]);
void get_f(double f[]);
void get_at(double at[]);
void get_af(double af[]);
void get_ifp(int **ifp);
void get_dsph();
void get_cg_first(int Ncoef, double _cg_first[]);
void get_fitmat_a(double **_fitmat);
void get_fitmat_b(double **_fitmat);
void get_indx(int indx[]);
void get_fitvec_a(double fitvec[]);
void get_fitvec_b(double fitvec[]);
void get_ee(int ndata, double **ee);
void get_ee0(int ndata, double **ee0);
void get_tim(int ndata, double tim[]);
void get_brightness(int ndata, double brightness[]);
void get_cg24(double cg[]);
void get_ia24(int ia[]);
void get_covar(int iMax, double **covar);
void get_aalpha(int iMax, double **aalpha);
void get_weight(int iMax, double weight[]);
void get_inrel(int iMax, int inrel[]);
void get_pleg();
void get_dsph2();
void get_sig(double sig[]);