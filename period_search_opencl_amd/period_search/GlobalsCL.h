#pragma OPENCL FP_CONTRACT ON

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

//struct __attribute__((packed)) freq_context
//struct mfreq_context
//struct __attribute__((aligned(8))) mfreq_context
//#ifdef NVIDIA
//struct mfreq_context
//#else
//typedef struct mfreq_context
//#endif
typedef struct mfreq_context
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

	int Niter;
	int np, np1, np2;
	int isInvalid, isAlamda, isNiter;
	int icol;
	//double conw_r;

	int ipiv[MAX_N_PAR + 1];
	int indxc[MAX_N_PAR + 1];
	int indxr[MAX_N_PAR + 1];
	int sh_icol[BLOCK_DIM];
	int sh_irow[BLOCK_DIM];
} CUDA_LCC;

//struct freq_context
//typedef struct __attribute__((aligned(8))) freq_context
//#ifdef NVIDIA
//struct freq_context
//#else
//typedef struct freq_context
//#endif
struct freq_context
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

	int ia[MAX_N_PAR + 1];

	int Dg_block;
	int lastone;
	int lastma;
	int ma;
	int Mfit, Mfit1;
	int Mmax, Lmax;
	int n;
	int Ncoef, Ncoef0;
	int Numfac;
	int Numfac1;
	int Nphpar;
	int ndata;
	int Is_Precalc;
};

//struct freq_result
//struct __attribute__((aligned(8))) freq_result
//#ifdef NVIDIA
//struct freq_result
//#else
//typedef struct freq_result
//#endif
struct freq_result
{
	double dark_best, per_best, dev_best, dev_best_x2, la_best, be_best, freq;
	int isReported, isInvalid, isNiter;
};
