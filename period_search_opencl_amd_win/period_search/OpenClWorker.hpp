#pragma once
#include "constants.h"

void Init();
void curvCl(double cg[]); //, double Fc[][MAX_LM + 1], double Fs[][MAX_LM + 1], double Dsph[][MAX_N_PAR + 1], double Dg[][MAX_N_PAR + 1]);
void daveCl(double *dave, double *dyda);