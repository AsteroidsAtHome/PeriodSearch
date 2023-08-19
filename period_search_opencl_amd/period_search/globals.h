#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>

#include <cstdio>
#include "constants.h"
#include <string>

extern cl_int l_max, m_max, n_iter, last_call,
n_coef, num_fac, l_curves, n_ph_par,
deallocate;

extern cl_double o_chi_square, chi_square, a_lambda, a_lamda_incr, a_lamda_start, scale, // phi_0,
area[MAX_N_FAC + 1], d_area[MAX_N_FAC + 1], sclnw[MAX_LC + 1],
y_out[MAX_N_OBS + 1],
f_c[MAX_N_FAC + 1][MAX_LM + 1], f_s[MAX_N_FAC + 1][MAX_LM + 1],
t_c[MAX_N_FAC + 1][MAX_LM + 1], t_s[MAX_N_FAC + 1][MAX_LM + 1],
d_sphere[MAX_N_FAC + 1][MAX_N_PAR + 1], d_g[MAX_N_FAC + 1][MAX_N_PAR + 1],
normal[MAX_N_FAC + 1][3], bl_matrix[4][4],
pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
d_bl_matrix[3][4][4];
//weight[MAX_N_OBS + 1];

extern cl_int CUDA_grid_dim;
extern cl::Program program;

//extern std::vector<cl_int2, int> texture;

// OpenCL
extern cl_int max_l_points;
extern cl_double phi_0;
extern cl_double Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1], Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1], Dg[MAX_N_FAC + 1][MAX_N_PAR + 1];
extern cl_double Area[MAX_N_FAC + 1], Darea[MAX_N_FAC + 1];
extern cl_double weight[MAX_N_OBS + 1];
extern cl_int l_points[MAX_LC + 1], in_rel[MAX_LC + 1];

extern std::string kernelCurv, kernelDaveFile, kernelSig2wghtFile;
extern std::vector<cl::Platform> platforms;
extern std::vector<cl::Device> devices;
extern cl::Context context;
extern cl::Program program;
extern cl::Kernel kernel, kernelDave, kernelSig2wght;
extern cl::CommandQueue queue;
extern unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
extern cl::Buffer bufCg, bufArea, bufDarea, bufDg, bufFc, bufFs, bufDsph, bufPleg, bufMmax, bufLmax, bufX, bufY, bufZ;
extern cl::Buffer bufSig, bufSig2iwght, bufDy, bufWeight, bufYmod;
extern cl::Buffer bufDave, bufDyda;
extern cl::Buffer bufD;


// NOTE: global to one thread
//struct FreqContext
//{
//	//	double Area[MAX_N_FAC+1];
//	double* Area;
//	//	double Dg[(MAX_N_FAC+1)*(MAX_N_PAR+1)];
//	double* Dg;
//	//	double alpha[MAX_N_PAR+1][MAX_N_PAR+1];
//	double* alpha;
//	//	double covar[MAX_N_PAR+1][MAX_N_PAR+1];
//	double* covar;
//	//	double dytemp[(POINTS_MAX+1)*(MAX_N_PAR+1)]
//	double* dytemp;
//	//	double ytemp[POINTS_MAX+1],
//	double* ytemp;
//	double cg[MAX_N_PAR + 1];
//	double Ochisq, Chisq, Alamda;
//	double atry[MAX_N_PAR + 1], beta[MAX_N_PAR + 1], da[MAX_N_PAR + 1];
//	double Blmat[4][4];
//	double Dblm[3][4][4];
//	//mrqcof locals
//	double dyda[MAX_N_PAR + 1], dave[MAX_N_PAR + 1];
//	double trial_chisq, ave;
//	int np, np1, np2;
//	//bright
//	double e_1[POINTS_MAX + 1], e_2[POINTS_MAX + 1], e_3[POINTS_MAX + 1], e0_1[POINTS_MAX + 1], e0_2[POINTS_MAX + 1], e0_3[POINTS_MAX + 1], de[POINTS_MAX + 1][4][4], de0[POINTS_MAX + 1][4][4];
//	double jp_Scale[POINTS_MAX + 1];
//	double jp_dphp_1[POINTS_MAX + 1], jp_dphp_2[POINTS_MAX + 1], jp_dphp_3[POINTS_MAX + 1];
//	// gaus
//	int indxc[MAX_N_PAR + 1], indxr[MAX_N_PAR + 1], ipiv[MAX_N_PAR + 1];
//	//global
//	double freq;
//	int isNiter;
//	double iter_diff, rchisq, dev_old, dev_new;
//	int Niter;
//	double chck[4];
//	int isAlamda; //Alamda<0 for init
//	//
//	int isInvalid;
//	//test
//};
//
//struct FreqResult
//{
//	int isReported;
//	double dark_best, per_best, dev_best, la_best, be_best;
//};

//const int BLOCK_DIM = 128;