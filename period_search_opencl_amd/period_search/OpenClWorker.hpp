#pragma once
#include "constants.h"

void Init(cl_double cg[]);
void curvCl(); //, double Fc[][MAX_LM + 1], double Fs[][MAX_LM + 1], double Dsph[][MAX_N_PAR + 1], double Dg[][MAX_N_PAR + 1]);
void daveCl(double *dave, double *dyda, int ma);
void sigSetBuffers(double *sig, double *weight, double *sig2iwght, double *dy, double *y, double *ymod);
void sig2IwghtF(const int offset, const int range, double *sig, double *weight, double *sig2iwght, double *dy, double *y, double *ymod);