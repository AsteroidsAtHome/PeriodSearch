#pragma once
#include <CL/cl.hpp>
#include <stdio.h>
#include "constants.h"

extern int Lmax, Mmax, Niter, Lastcall,
Ncoef, Numfac, Lcurves, Nphpar,
Lpoints[MAX_LC + 1], Inrel[MAX_LC + 1],
Deallocate;

extern double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,
//Area[MAX_N_FAC + 1], 
tmpArea[MAX_N_FAC + 1],
//Darea[MAX_N_FAC + 1], 
Sclnw[MAX_LC + 1],
Yout[MAX_N_OBS + 1],
//Fc[MAX_N_FAC + 1][MAX_LM + 1], 
//Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
//Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1], Dg[MAX_N_FAC + 1][MAX_N_PAR + 1],
Nor[3][MAX_N_FAC + 1],
Blmat[4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4],
Weight[MAX_N_OBS + 1],
ytemp[POINTS_MAX + 1];

/*Nor[MAX_N_FAC + 1][3], */

// OpenCL
extern cl_double Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1], Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1], Dg[MAX_N_FAC + 1][MAX_N_PAR + 1];
extern cl_double Area[MAX_N_FAC + 1], Darea[MAX_N_FAC + 1];

extern std::string kernelCurv, kernelDaveFile, kernelSig2wghtFile;
extern std::vector<cl::Platform> platforms;
extern cl::Context context;
extern std::vector<cl::Device> devices;
extern cl::Program program;
extern cl::Kernel kernel, kernelDave, kernelSig2wght;
extern cl::CommandQueue queue;
extern unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
extern cl::Buffer bufCg, bufArea, bufDarea, bufDg, bufFc, bufFs, bufDsph, bufPleg, bufMmax, bufLmax, bufX, bufY, bufZ;
extern cl::Buffer bufSig, bufSig2iwght, bufDy, bufWeight, bufYmod;
extern cl::Buffer bufDave, bufDyda;
extern cl::Buffer bufD;
