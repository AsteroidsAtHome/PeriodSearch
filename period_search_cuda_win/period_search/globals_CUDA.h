#include "constants.h"

//global to all freq
__constant__ extern int /*CUDA_n,*/CUDA_Ncoef,CUDA_Nphpar,CUDA_Numfac,CUDA_Numfac1,CUDA_Dg_block;
__constant__ extern int CUDA_ia[MAX_N_PAR+1];
__constant__ extern int CUDA_ma,CUDA_mfit,CUDA_mfit1,CUDA_lastone,CUDA_lastma,CUDA_ncoef0;
__device__ extern double CUDA_cg_first[MAX_N_PAR+1];
__device__ extern double CUDA_beta_pole[N_POLES+1];
__device__ extern double CUDA_lambda_pole[N_POLES+1];
__device__ extern double CUDA_par[4];
__device__ extern double CUDA_cl,CUDA_Alamda_start,CUDA_Alamda_incr;
__device__ extern int CUDA_n_iter_max,CUDA_n_iter_min,CUDA_ndata;
__device__ extern double CUDA_iter_diff_max;
__constant__ extern double CUDA_Nor[MAX_N_FAC+1][3];
__constant__ extern double CUDA_conw_r;
__constant__ extern int CUDA_Lmax,CUDA_Mmax;
__device__ extern double CUDA_Fc[MAX_N_FAC+1][MAX_LM+1];
__device__ extern double CUDA_Fs[MAX_N_FAC+1][MAX_LM+1];
__device__ extern double CUDA_Pleg[MAX_N_FAC+1][MAX_LM+1][MAX_LM+1];
__constant__ extern double CUDA_Darea[MAX_N_FAC+1];
__device__ extern double CUDA_Dsph[MAX_N_FAC+1][MAX_N_PAR+1];
__device__ extern double *CUDA_ee/*[MAX_N_OBS+1][3]*/;
__device__ extern double *CUDA_ee0/*[MAX_N_OBS+1][3]*/;
__device__ extern double CUDA_tim[MAX_N_OBS+1];
//__device__ extern double CUDA_brightness[MAX_N_OBS+1];
//__device__ extern double CUDA_sig[MAX_N_OBS+1];
//__device__ extern double *CUDA_Weight/*[MAX_N_OBS+1]*/;
__constant__ extern double CUDA_Phi_0;
__device__ extern int CUDA_End; 

extern texture<int2,1> texWeight;
extern texture<int2,1> texbrightness;
extern texture<int2,1> texsig;

//global to one thread
struct freq_context
{
//	double Area[MAX_N_FAC+1];
	double *Area;
//	double Dg[(MAX_N_FAC+1)*(MAX_N_PAR+1)];
	double *Dg;
//	double alpha[MAX_N_PAR+1][MAX_N_PAR+1];
	double *alpha;
//	double covar[MAX_N_PAR+1][MAX_N_PAR+1];
	double *covar;
//	double dytemp[(POINTS_MAX+1)*(MAX_N_PAR+1)]
	double *dytemp;
//	double ytemp[POINTS_MAX+1],
	double *ytemp;
	double cg[MAX_N_PAR+1];
	double Ochisq, Chisq, Alamda;
	double atry[MAX_N_PAR+1], beta[MAX_N_PAR+1], da[MAX_N_PAR+1];
	double Blmat[4][4];
	double Dblm[3][4][4];
	//mrqcof locals
	double dyda[MAX_N_PAR+1], dave[MAX_N_PAR+1];
	double trial_chisq,ave;
	int np, np1, np2;
	//bright
    double e_1[POINTS_MAX+1],e_2[POINTS_MAX+1],e_3[POINTS_MAX+1],e0_1[POINTS_MAX+1],e0_2[POINTS_MAX+1],e0_3[POINTS_MAX+1],de[POINTS_MAX+1][4][4],de0[POINTS_MAX+1][4][4];
    double jp_Scale[POINTS_MAX+1];
	double jp_dphp_1[POINTS_MAX+1],jp_dphp_2[POINTS_MAX+1],jp_dphp_3[POINTS_MAX+1];
	// gaus
	int indxc[MAX_N_PAR+1],indxr[MAX_N_PAR+1],ipiv[MAX_N_PAR+1];
	//global
	double freq;
	int isNiter;
	double iter_diff,rchisq,dev_old,dev_new;
	int Niter;
	double chck[4];
	int isAlamda; //Alamda<0 for init
	//
	int isInvalid;
	//test
};

extern texture<int2,1> texArea;
extern texture<int2,1> texDg;

__device__ extern freq_context *CUDA_CC;

struct freq_result
{
	int isReported;
	double dark_best,per_best,dev_best,la_best,be_best;
};

__device__ extern freq_result *CUDA_FR;
