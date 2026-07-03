#pragma once
#include <cuda_runtime_api.h>
#include <vector>

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

//NOTE: MUST BE 128 or 64
#define CUDA_BLOCK_DIM 128

//NOTE: https://devtalk.nvidia.com/default/topic/517801/-34-texture-is-not-a-template-34-error-mvs-2010/

//global to all freq
__constant__ extern int /*CUDA_n,*/CUDA_Ncoef, CUDA_Nphpar, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ extern int CUDA_ia[MAX_N_PAR + 1];
__constant__ extern int CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__device__ extern double* CUDA_cg_first;	//[MAX_N_PAR + 1];
__device__ extern double CUDA_beta_pole[N_POLES + 1];
__device__ extern double CUDA_lambda_pole[N_POLES + 1];
__device__ extern double CUDA_par[4];
__device__ extern double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ extern int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__device__ extern double CUDA_iter_diff_max;
__constant__ extern double CUDA_Nor[MAX_N_FAC + 1][3];
__constant__ extern double CUDA_conw_r;
__constant__ extern int CUDA_Lmax, CUDA_Mmax;
__device__ extern double CUDA_Fc[MAX_N_FAC + 1][MAX_LM + 1];
__device__ extern double CUDA_Fs[MAX_N_FAC + 1][MAX_LM + 1];
__device__ extern double CUDA_Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
__device__ extern double CUDA_Darea[MAX_N_FAC + 1]; //not constant access in curv (depends on threadidx --> slow)
__device__ extern double CUDA_Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
__device__ extern double* CUDA_ee;			/*[MAX_N_OBS+1][3]*/
__device__ extern double* CUDA_ee0;			/*[MAX_N_OBS+1][3]*/
__device__ extern double* CUDA_tim;			/*[MAX_N_OBS + 1]*/
__device__ extern double* CUDA_brightness;	/*[MAX_N_OBS+1]*/
__device__ extern double* CUDA_sig;			/*[MAX_N_OBS+1]*/
__device__ extern double* CUDA_Weight;		/*[MAX_N_OBS+1]*/
__constant__ extern double CUDA_Phi_0;
__device__ extern int CUDA_End;
__device__ extern int CUDA_Is_Precalc;

extern double* d_CUDA_tim;

//global to one thread
struct freq_context
{
	// NOTE: Keep the following pointer members in this exact sequence!

	
	double* Area;		//	former double Area[MAX_N_FAC+1]	
	double* Dg;			//	former double Dg[(MAX_N_FAC+1)*(MAX_N_PAR+1)]	
	double* alpha;		//	former double alpha[MAX_N_PAR+1][MAX_N_PAR+1]	
	double* covar;		//	former double covar[MAX_N_PAR+1][MAX_N_PAR+1]	
	double* dytemp;		//	former double dytemp[(POINTS_MAX+1)*(MAX_N_PAR+1)]	
	double* ytemp;		//	former double ytemp[POINTS_MAX+1]

	//bright
	double* e_1;
	double* e_2;
	double* e_3;
	double* e0_1;
	double* e0_2;
	double* e0_3;
	double* jp_Scale;
	double* jp_dphp_1;
	double* jp_dphp_2;
	double* jp_dphp_3;
	double* de;
	double* de0;
	// End of pointer members sequence

	double cg[MAX_N_PAR + 1];
	double Ochisq, Chisq, Alamda;
	double atry[MAX_N_PAR + 1], beta[MAX_N_PAR + 1], da[MAX_N_PAR + 1];
	double Blmat[4][4];
	double Dblm[3][4][4];

	//mrqcof locals
	double dyda[MAX_N_PAR + 1], dave[MAX_N_PAR + 1];
	double trial_chisq, ave;
	int np, np1, np2;

	// gauss
	int indxc[MAX_N_PAR + 1], indxr[MAX_N_PAR + 1], ipiv[MAX_N_PAR + 1];

	//global
	double freq;
	int isNiter;
	double iter_diff, rchisq, dev_old, dev_new;
	int Niter;
	double chck[4];
	int isAlamda; //Alamda < 0 for init
	//
	int isInvalid;
	//test
};

__device__ extern double *CUDA_Area;
__device__ extern double *CUDA_Dg;
__device__ extern freq_context *CUDA_CC;

struct freq_result
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};

__device__ extern freq_result *CUDA_FR;

/* ---- shared-memory staging for the 2026 warp-cooperative kernels ---- */

/* transposed dytemp row stride: dytempT[(jp-1)*DYT_STRIDE + l], l = 1..ma.
   Requires ma <= DYT_STRIDE-1 (spherical-harmonics degree <= 6, i.e. every
   production workunit); enforced on the host. */
#define DYT_STRIDE 64
#define CURVE2_K 8
#define GEOM_PT_SIZE 26
#define GEO_BATCH 16

#ifdef __CUDACC__
struct brightshare
{
	double wcA[32];                     /* compacted facet weights, point A */
	double wcB[32];                     /* compacted facet weights, point B */
	int    fc[32];                      /* compacted facet indices */
	double geo[GEO_BATCH][GEOM_PT_SIZE];/* per-point geometry */
	double inv[11];                     /* per-curve invariants */
};

struct curve2share
{
	double T[CURVE2_K][DYT_STRIDE];     /* staged dyda tile, rows 1..ma */
	double s2w[CURVE2_K];
	double dws[CURVE2_K];
};

/* bright and curve2 never use their staging at the same time, so they share
   one per-block union: the shared footprint is max(...), not the sum */
union mrqshare
{
	brightshare b;
	curve2share c2;
};

__device__ __forceinline__ mrqshare* mrq_share_block()
{
	__shared__ mrqshare s;
	return &s;
}
#endif
