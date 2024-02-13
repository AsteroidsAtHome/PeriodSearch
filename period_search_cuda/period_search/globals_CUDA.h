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

#define N_BLOCKS 2048

//global to all freq
__constant__ extern int CUDA_Ncoef, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ extern int CUDA_ma, CUDA_mfit, /*CUDA_mfit1,*/ CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__constant__ extern double CUDA_cg_first[MAX_N_PAR + 1];
__constant__ extern int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__constant__ extern double CUDA_iter_diff_max;
__constant__ extern double CUDA_conw_r;
__constant__ extern int CUDA_Lmax, CUDA_Mmax;
__constant__ extern double CUDA_lcl, CUDA_Alamda_start, CUDA_Alamda_incr, CUDA_Alamda_incrr;
__constant__ extern double CUDA_Phi_0;
__constant__ extern double CUDA_beta_pole[N_POLES + 1];
__constant__ extern double CUDA_lambda_pole[N_POLES + 1];

__device__ extern double CUDA_par[4];
__device__ extern int CUDA_ia[MAX_N_PAR + 1];
__device__ extern double CUDA_Nor[3][MAX_N_FAC + 1];
__device__ extern double CUDA_Fc[MAX_LM + 1][MAX_N_FAC + 1];
__device__ extern double CUDA_Fs[MAX_LM + 1][MAX_N_FAC + 1];

__device__ extern double CUDA_Pleg[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
__device__ extern double CUDA_Darea[MAX_N_FAC + 1]; 
__device__ extern double CUDA_Dsph[MAX_N_PAR + 1][MAX_N_FAC + 1];

__device__ extern int CUDA_End;
__device__ extern int CUDA_Is_Precalc;

__device__ extern double CUDA_tim[MAX_N_OBS + 1];
__device__ extern double CUDA_brightness[MAX_N_OBS+1];
__device__ extern double CUDA_sig[MAX_N_OBS+1];
__device__ extern double CUDA_sigr2[MAX_N_OBS+1]; // (1/CUDA_sig^2) /*[MAX_N_OBS+1]*/;
__device__ extern double CUDA_Weight[MAX_N_OBS+1];
__device__ extern double CUDA_ee[3][MAX_N_OBS+1]; 
__device__ extern double CUDA_ee0[3][MAX_N_OBS+1]; 


//global to one thread
struct freq_context
{
  double *Dg;
  //double *alpha;
  double *covar;
  double *dytemp;
  double *ytemp;
  
  //double cg[MAX_N_PAR + 1];
  //double beta[MAX_N_PAR + 1];
  double da[MAX_N_PAR + 1];
};

extern __device__ double *CUDA_Dg;

__device__ extern freq_context *CUDA_CC;

/*
struct freq_result
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};
*/

//__device__ extern freq_result *CUDA_FR;
//LFR
__managed__ extern int isReported[N_BLOCKS];
__managed__ extern double dark_best[N_BLOCKS];
__managed__ extern double per_best[N_BLOCKS];
__managed__ extern double dev_best[N_BLOCKS];
__managed__ extern double la_best[N_BLOCKS];
__managed__ extern double be_best[N_BLOCKS];
