//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
//#define __CL_ENABLE_EXCEPTIONS
#pragma pack(8)

#include <CL/cl.h>
#include <CL/cl.hpp>
#include "constants.h"
#include <vector>
#include <iostream>
#include <cstring>


//#define kernel

//#ifdef __GNUC__
//#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
//#endif
//
#ifdef _MSC_VER
//#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#define PACK( __Declaration__ ) __pragma( pack(push, 8) ) __Declaration__ __pragma( pack(pop))
#endif


// global to one thread

//struct __declspec(align(8)) mfreq_context {
struct alignas(8) mfreq_context
{
	//double* Area;
	//double* Dg;
	//double* alpha;
	//double* covar;
	//double* dytemp;
	//double* ytemp;

	 double Area[MAX_N_FAC + 1];
	 double Dg[(MAX_N_FAC + 1) * (MAX_N_PAR + 1)];
	 double alpha[(MAX_N_PAR + 1) * (MAX_N_PAR + 1)];
	 double covar[(MAX_N_PAR + 1) * (MAX_N_PAR + 1)];
	 double dytemp[(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	 double ytemp[POINTS_MAX + 1];

	double beta[MAX_N_PAR + 1];
	double atry[MAX_N_PAR + 1];
	double da[MAX_N_PAR + 1];
	double cg[MAX_N_PAR + 1];
	double Blmat[4][4];
	double Dblm[3][4][4];
	double jp_Scale[POINTS_MAX + 1];
	double jp_dphp_1[POINTS_MAX + 1];
	double jp_dphp_2[POINTS_MAX + 1];
	double jp_dphp_3[POINTS_MAX + 1];
	double e_1[POINTS_MAX + 1];
	double e_2[POINTS_MAX + 1];
	double e_3[POINTS_MAX + 1];
	double e0_1[POINTS_MAX + 1];
	double e0_2[POINTS_MAX + 1];
	double e0_3[POINTS_MAX + 1];
	double de[POINTS_MAX + 1][4][4];
	double de0[POINTS_MAX + 1][4][4];
	double dave[MAX_N_PAR + 1];
	double dyda[MAX_N_PAR + 1];

	double sh_big[BLOCK_DIM];
	double chck[4];
	double pivinv;
	double ave;
	double freq;
	double Alamda;
	double Chisq;
	double Ochisq;
	double rchisq;
	double trial_chisq;
	double iter_diff, dev_old, dev_new;

	cl_int Niter;
	cl_int np, np1, np2;
	cl_int isInvalid, isAlamda, isNiter;
	cl_int icol;
	//double conw_r;

	cl_int ipiv[MAX_N_PAR + 1];
	cl_int indxc[MAX_N_PAR + 1];
	cl_int indxr[MAX_N_PAR + 1];
	cl_int sh_icol[BLOCK_DIM];
	cl_int sh_irow[BLOCK_DIM];
};

//typedef struct __attribute__((packed)) freq_context
//struct __declspec(align(8)) freq_context
//#pragma pack(8)
struct alignas(8) freq_context
{
	double Phi_0;
	double logCl;
	double cl;
	//double logC;
	double lambda_pole[N_POLES + 1];
	double beta_pole[N_POLES + 1];


	double par[4];
	double Alamda_start;
	double Alamda_incr;

	//double cgFirst[MAX_N_PAR + 1];
	double tim[MAX_N_OBS + 1];
	double ee[MAX_N_OBS + 1][3];	// double* ee;
	double ee0[MAX_N_OBS + 1][3];	// double* ee0;
	double Sig[MAX_N_OBS + 1];
	double Weight[MAX_N_OBS + 1];
	double Brightness[MAX_N_OBS + 1];
	double Fc[MAX_N_FAC + 1][MAX_LM + 1];
	double Fs[MAX_N_FAC + 1][MAX_LM + 1];
	double Darea[MAX_N_FAC + 1];
	double Nor[MAX_N_FAC + 1][3];
	double Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	double Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
	double conw_r;

	cl_int ia[MAX_N_PAR + 1];

	cl_int Dg_block;
	cl_int lastone;
	cl_int lastma;
	cl_int ma;
	cl_int Mfit, Mfit1;
	cl_int Mmax, Lmax;
	cl_int n;
	cl_int Ncoef, Ncoef0;
	cl_int Numfac;
	cl_int Numfac1;
	cl_int Nphpar;
	cl_int ndata;
	cl_int Is_Precalc;
};

//extern __declspec(align(4)) freq_context* CUDA_CC2;

// NOTE: Check here https://docs.microsoft.com/en-us/cpp/preprocessor/pack?redirectedfrom=MSDN&view=vs-2019
//#pragma pack(4)
//struct __declspec(align(16)) freq_result
struct alignas(8) freq_result
{
	double dark_best, per_best, dev_best, dev_best_x2, la_best, be_best, freq;
	cl_int isReported, isInvalid, isNiter;
};
