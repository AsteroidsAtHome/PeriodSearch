#pragma once
int CUDAPrepare(int cudadev, double *beta_pole, double *lambda_pole, double *par, double cl,
		double Alamda_start, double Alamda_incr, double Alamda_incrr, double ee[][MAX_N_OBS+1],
		double ee0[][MAX_N_OBS+1], double *tim, double Phi_0, int checkex, int ndata);

int CUDAStart(int cudadev, int n_start_from,double freq_start,double freq_end,double freq_step,double stop_condition,int n_iter_min,double conw_r,
			  int ndata,int *ia,int *ia_par,double *cg_first,MFILE& mf,double escl,double *sig,int Numfac,double *brightness);

int CUDAPrecalc(int cudadev, double freq_start,double freq_end,double freq_step,double stop_condition,int n_iter_min,double *conw_r,
			  int ndata,int *ia,int *ia_par,int *new_conw,double *cg_first,double *sig,int Numfac,double *brightness);

int DoCheckpoint(MFILE& mf, int nlines, int newConw, double conwr);

void CUDAUnprepare(void);

void GetCUDAOccupancy(const int cudaDevice);
