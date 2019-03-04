#pragma once
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "constants.h"
#include "LightPoint.h"
#include <CL/cl.hpp>


extern int Lmax, Mmax, Niter, Lastcall,
Ncoef, Numfac, Lcurves, Nphpar,
Lpoints[MAX_LC + 1], Inrel[MAX_LC + 1],
Deallocate;

extern double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,
//Area[MAX_N_FAC + 1],
tmpArea[MAX_N_FAC + 1],
//Darea[MAX_N_FAC + 1],
Sclnw[MAX_LC + 1],
//Yout[MAX_N_OBS + 1],
//Fc[MAX_N_FAC + 1][MAX_LM + 1],
//Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
//Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1], Dg[MAX_N_FAC + 1][MAX_N_PAR + 1],
Nor[3][MAX_N_FAC + 1],
Blmat[4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4],
Weight[MAX_N_OBS + 1],
ytemp[POINTS_MAX + 1], tmpYtemp[POINTS_MAX + 1],
dytemp[POINTS_MAX + 1][MAX_N_PAR + 1];
extern cl_double dave[MAX_N_PAR + 1];
//extern double alpha[MAX_N_PAR + 1][MAX_N_PAR + 1];
extern double ddd[(MAX_N_PAR + 1) * (MAX_N_PAR + 1)];

//extern double *sig;
//extern Beta *_beta;

// Device specifics
extern size_t deviceMaxWorkItems[3];
extern size_t deviceMaxWorkgroupSize;
extern cl_uint deviceMaxWorkItemDimensions;
extern bool mrqcofClInitialized, mrqcofNotRelClInitialized;

extern double _dy[20000];
extern double _sig2iwght[20000]; // , _alpha[MAX_N_PAR * MAX_N_PAR], _beta[MAX_N_PAR];
extern cl_uint testBufRes[20000];
extern int ma;
extern bool prep;
//extern double **aalpha;

/*Nor[MAX_N_FAC + 1][3], */

// OpenCL
extern cl_double Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1], Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1], Dg[MAX_N_FAC + 1][MAX_N_PAR + 1];
extern cl_double Area[MAX_N_FAC + 1], Darea[MAX_N_FAC + 1];
//extern cl_double _aalpha[MAX_N_PAR][MAX_N_PAR];

extern std::string kernelCurv, kernelMrqcofClFile, kernelMrqcofMidClFile, kernelMrqcofNotRelClFile, kernelDaveFile, kernelSig2wghtFile;
extern std::vector<cl::Platform> platforms;
extern cl::Context context;
extern std::vector<cl::Device> devices;
extern cl::Program program;
extern cl::Kernel kernel, kernelMrqcofCl, kernelMrqcofMidCl, kernelMrqcofNotRelCl, kernelDave, kernelSig2wght;
extern cl::CommandQueue queue;
extern unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
extern cl::Buffer bufCg, bufArea, bufDarea, bufDg, bufFc, bufFs, bufDsph, bufPleg, bufMmax, bufLmax, bufX, bufY, bufZ;
extern cl::Buffer bufSig2iwght, bufDy, bufWeight, bufYmod;
extern cl::Buffer bufDave, bufDyda;
extern cl::Buffer bufD, testBuf;
extern cl::Buffer bufAlpha, bufBeta;
extern cl_mem pinnedBuffer, deviceBuffer;


//extern std::vector<double> atry, beta, da;
//extern struct EllipsoidFunctionContext<double> ellfitContext;