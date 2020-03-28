#pragma once
#include <CL/cl.hpp>

extern cl_int ClPrepare(int deviceId, double* beta_pole, double* lambda_pole, double* par, double cl, double Alambda_start, double Alambda_incr,
    double ee[][3], double ee0[][3], double* tim, double Phi_0, int checkex, int ndata);

extern cl_int ClPrecalc(double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
    int ndata, int* ia, int* ia_par, int* new_conw, double* cg_first, double* sig, int Numfac, double* brightness);