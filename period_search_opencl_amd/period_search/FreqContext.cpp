#include "constants.h"

// NOTE: global to one thread
struct FreqContext
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

struct FreqResult
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};