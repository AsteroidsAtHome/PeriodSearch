#pragma once
#include <CL/cl.hpp>
#include "mfile.h"


extern cl_int ClPrepare(cl_int deviceId, cl_double* beta_pole, cl_double* lambda_pole, cl_double* par, cl_double cl, cl_double Alambda_start, cl_double Alambda_incr,
	cl_double ee[][3], cl_double ee0[][3], cl_double* tim, cl_double Phi_0, cl_int checkex, cl_int ndata);

extern cl_int ClPrecalc(cl_double freq_start, cl_double freq_end, cl_double freq_step, cl_double stop_condition, cl_int n_iter_min, cl_double* conw_r,
	cl_int ndata, cl_int* ia, cl_int* ia_par, cl_int* new_conw, cl_double* cg_first, cl_double* sig, cl_int Numfac, cl_double* brightness, cl_double lcoef, int n_coef);

extern int CUDAStart(int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
	int ndata, int* ia, int* ia_par, double* cg_first, MFILE& mf, double escl, double* sig, int Numfac, double* brightness);

int DoCheckpoint(MFILE &mf, int nlines, int newConw, double conwr);