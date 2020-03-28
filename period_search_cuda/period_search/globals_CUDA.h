#pragma once
#include <cuda_runtime_api.h>

//  NOTE Fake declaration to satisfy intellisense. See https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/39990500
#ifndef __CUDACC__
//#define __host__
//#define __device__
//#define __shared__
//#define __constant__
//#define __global__
//#define __host__
#include <device_functions.h>
#include <vector_types.h>
#include <driver_types.h>
#include <texture_types.h>
#include <cuda_texture_types.h>
//#define __CUDACC__
#define __CUDA__
inline void __syncthreads() {};
inline void atomicAdd(int*, int) {};

//template <class T>
//static __device__ T tex1Dfetch(texture<int2, 1> texObject, int x) { return {}; };

__device__ __device_builtin__ double __hiloint2double(int hi, int lo);

//template<class T, int texType = cudaTextureType1D, enum cudaTextureReadMode mode = cudaReadModeElementType>
//struct texture {};
//	int                          norm;
//	enum cudaTextureFilterMode   fMode;
//	enum cudaTextureAddressMode  aMode;
//	struct cudaChannelFormatDesc desc;
//};

//#include <__clang_cuda_builtin_vars.h>
//#include <__clang_cuda_intrinsics.h>
//#include <__clang_cuda_math_forward_declares.h>
//#include <__clang_cuda_complex_builtins.h>
//#include <../../../../../../2019/Professional/VC/Tools/Llvm/lib/clang/9.0.0/include/__clang_cuda_cmath.h>
#endif

//#ifdef __INTELLISENSE__
////#define __device__ \
////			__location__(device)
//#endif

#include "constants.h"


//NOTE: https://devtalk.nvidia.com/default/topic/517801/-34-texture-is-not-a-template-34-error-mvs-2010/

//global to all freq
__constant__ extern int /*CUDA_n,*/CUDA_Ncoef, CUDA_Nphpar, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ extern int CUDA_ia[MAX_N_PAR + 1];
__constant__ extern int CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__device__ extern double CUDA_cg_first[MAX_N_PAR + 1];
__device__ extern double CUDA_beta_pole[N_POLES + 1];
__device__ extern double CUDA_lambda_pole[N_POLES + 1];
__device__ extern double CUDA_par[4];
//__device__ __constant__ extern double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ extern double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ extern int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__device__ extern double CUDA_iter_diff_max;
__constant__ extern double CUDA_Nor[MAX_N_FAC + 1][3];
__constant__ extern double CUDA_conw_r;
__constant__ extern int CUDA_Lmax, CUDA_Mmax;
__device__ extern double CUDA_Fc[MAX_N_FAC + 1][MAX_LM + 1];
__device__ extern double CUDA_Fs[MAX_N_FAC + 1][MAX_LM + 1];
__device__ extern double CUDA_Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
__constant__ extern double CUDA_Darea[MAX_N_FAC + 1];
__device__ extern double CUDA_Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
__device__ extern double* CUDA_ee/*[MAX_N_OBS+1][3]*/;
__device__ extern double* CUDA_ee0/*[MAX_N_OBS+1][3]*/;
__device__ extern double CUDA_tim[MAX_N_OBS + 1];
//__device__ extern double CUDA_brightness[MAX_N_OBS+1];
//__device__ extern double CUDA_sig[MAX_N_OBS+1];
//__device__ extern double *CUDA_Weight/*[MAX_N_OBS+1]*/;
__constant__ extern double CUDA_Phi_0;
__device__ extern int CUDA_End;
__device__ extern int CUDA_Is_Precalc;

//__device__ extern int icol;
//__device__ extern double pivinv;
//__shared__ extern int sh_icol[CUDA_BLOCK_DIM];
//__shared__ extern int sh_irow[CUDA_BLOCK_DIM];
//__shared__ extern double sh_big[CUDA_BLOCK_DIM];



extern texture<int2, 1> texWeight;
extern texture<int2, 1> texbrightness;
extern texture<int2, 1> texsig;

//global to one thread
struct freq_context
{
	//	double Area[MAX_N_FAC+1];
	double* Area;
	//	double Dg[(MAX_N_FAC+1)*(MAX_N_PAR+1)];
	double* Dg;
	//	double alpha[MAX_N_PAR+1][MAX_N_PAR+1];
	double* alpha;
	//	double covar[MAX_N_PAR+1][MAX_N_PAR+1];
	double* covar;
	//	double dytemp[(POINTS_MAX+1)*(MAX_N_PAR+1)]
	double* dytemp;
	//	double ytemp[POINTS_MAX+1],
	double* ytemp;
	double cg[MAX_N_PAR + 1];
	double Ochisq, Chisq, Alamda;
	double atry[MAX_N_PAR + 1], beta[MAX_N_PAR + 1], da[MAX_N_PAR + 1];
	double Blmat[4][4];
	double Dblm[3][4][4];
	//mrqcof locals
	double dyda[MAX_N_PAR + 1], dave[MAX_N_PAR + 1];
	double trial_chisq, ave;
	int np, np1, np2;
	//bright
	double e_1[POINTS_MAX + 1], e_2[POINTS_MAX + 1], e_3[POINTS_MAX + 1], e0_1[POINTS_MAX + 1], e0_2[POINTS_MAX + 1], e0_3[POINTS_MAX + 1], de[POINTS_MAX + 1][4][4], de0[POINTS_MAX + 1][4][4];
	double jp_Scale[POINTS_MAX + 1];
	double jp_dphp_1[POINTS_MAX + 1], jp_dphp_2[POINTS_MAX + 1], jp_dphp_3[POINTS_MAX + 1];
	// gaus
	int indxc[MAX_N_PAR + 1], indxr[MAX_N_PAR + 1], ipiv[MAX_N_PAR + 1];
	//global
	double freq;
	int isNiter;
	double iter_diff, rchisq, dev_old, dev_new;
	int Niter;
	double chck[4];
	int isAlamda; //Alamda<0 for init
	//
	int isInvalid;
	//test
};

extern texture<int2, 1> texArea;
extern texture<int2, 1> texDg;

__device__ extern freq_context* CUDA_CC;

struct freq_result
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};

__device__ extern freq_result* CUDA_FR;
