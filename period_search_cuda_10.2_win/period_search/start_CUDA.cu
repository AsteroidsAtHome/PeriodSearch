#include <cuda.h>
#include "mfile.h"
#include "globals.h"
#include "globals_CUDA.h"
#include "start_CUDA.h"
#include "declarations_CUDA.h"
#include "boinc_api.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_occupancy.h>
#include <device_launch_parameters.h>

#ifdef __GNUC__
#include <time.h>
#endif

//global to all freq
__constant__ int /*CUDA_n,*/CUDA_Ncoef,CUDA_Nphpar,CUDA_Numfac,CUDA_Numfac1,CUDA_Dg_block;
__constant__ int CUDA_ia[MAX_N_PAR+1];
__constant__ int CUDA_ma,CUDA_mfit,CUDA_mfit1,CUDA_lastone,CUDA_lastma,CUDA_ncoef0;
__device__ double CUDA_cg_first[MAX_N_PAR+1];
__device__ double CUDA_beta_pole[N_POLES+1];
__device__ double CUDA_lambda_pole[N_POLES+1];
__device__ double CUDA_par[4];
__device__ double CUDA_cl,CUDA_Alamda_start,CUDA_Alamda_incr;
__device__ int CUDA_n_iter_max,CUDA_n_iter_min,CUDA_ndata;
__device__ double CUDA_iter_diff_max;
__constant__ double CUDA_Nor[MAX_N_FAC+1][3];
__constant__ double CUDA_conw_r;
__constant__ int CUDA_Lmax,CUDA_Mmax;
__device__ double CUDA_Fc[MAX_N_FAC+1][MAX_LM+1];
__device__ double CUDA_Fs[MAX_N_FAC+1][MAX_LM+1];
__device__ double CUDA_Pleg[MAX_N_FAC+1][MAX_LM+1][MAX_LM+1];
__constant__ double CUDA_Darea[MAX_N_FAC+1];
__device__ double CUDA_Dsph[MAX_N_FAC+1][MAX_N_PAR+1];
__device__ double *CUDA_ee/*[MAX_N_OBS+1][3]*/;
__device__ double *CUDA_ee0/*[MAX_N_OBS+1][3]*/;
__device__ double CUDA_tim[MAX_N_OBS+1];
//__device__ double CUDA_brightness[MAX_N_OBS+1];
//__device__ double CUDA_sig[MAX_N_OBS+1];
//__device__ double *CUDA_Weight/*[MAX_N_OBS+1]*/;
__constant__ double CUDA_Phi_0;
__device__ int CUDA_End;

texture<int2,1> texWeight;
texture<int2,1> texbrightness;
texture<int2,1> texsig;

//global to one thread
__device__ freq_context *CUDA_CC;
__device__ freq_result *CUDA_FR;

texture<int2,1> texArea;
texture<int2,1> texDg;

int CUDA_grid_dim;
double *pee,*pee0,*pWeight;

int CUDAPrepare(int cudadev,double *beta_pole,double *lambda_pole,double *par,double cl,double Alamda_start,double Alamda_incr,
	            double ee[][3],double ee0[][3],double *tim,double Phi_0,int checkex,int ndata)
{
	//init gpu
    cudaSetDevice(cudadev);
	cudaSetDeviceFlags(cudaDeviceBlockingSync);
	//determine gridDim
	cudaDeviceProp deviceProp;
	int SMXBlock; // Maximum number of resident thread blocks per multiprocessor
    cudaGetDeviceProperties(&deviceProp, cudadev);
	if (!checkex)
	{
		auto cudaVersion = CUDA_VERSION;
		fprintf(stderr, "CUDA version: %d\n", cudaVersion);
		//fprintf(stderr, "CUDA RC12!!!!!!!!!!\n");
		fprintf(stderr, "CUDA Device number: %d\n",cudadev);
		fprintf(stderr, "CUDA Device: %s\n",deviceProp.name);
		fprintf(stderr, "Compute capability: %d.%d\n",deviceProp.major,deviceProp.minor);
		//fprintf(stderr, "CUDA Device max grid size(x, y, z): %d, %d, %d \n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		fprintf(stderr, "Multiprocessors: %d\n",deviceProp.multiProcessorCount);
	}

	// NOTE: See this https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities , Table 15.
	// NOTE: Also this https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
	// NOTE: NB - Always set MaxUsedRegisters to 32 in order to achieve 100% SM occupancy (project's Configuration properties -> CUDA C/C++ -> Device)
	if (deviceProp.major == 7)
	{
		switch (deviceProp.minor)
		{
			case 0:
			case 2:
				SMXBlock = 32;	// CC 7.0 & 7.2, occupancy 100% = 32 blocks per SMX
			case 5:
				SMXBlock = 16;	// CC 7.5, occupancy 100% = 16 blocks per SMX
			default:
				SMXBlock = 16;	// unknown CC, occupancy unknown, 16 blocks per SMX
		}
	}
	else
	if (deviceProp.major == 6) //CC 6.0, 6.1 & 6.2
	{
		SMXBlock = 32; //occupancy 100% = 32 blocks per SMX
	}
	else
	if (deviceProp.major == 5) //CC 5.0, 5.2 & 5.3
	{
		SMXBlock = 32; //occupancy 100% = 32 blocks per SMX, instead as previous was 16 blocks per SMX which led to only 50%
	}
	else
	if (deviceProp.major == 3) //CC 3.0, 3.2, 3.5 & 3.7
	{
		SMXBlock = 16; //occupancy 100% = 16 blocks per SMX
	}
	/*else
	if (deviceProp.major==2) //CC 2.0 and 2.1
	{
		SMXBlock=8; //occupancy 67% = 8 blocks per SMX
	}
	else
	if ((deviceProp.major==1) && (deviceProp.major==3)) //CC 1.3
	{
		SMXBlock=8; //occupancy 50% = 8 blocks per SMX
		CUDA_BLOCK_DIM=64;
	}*/
	else
	{
		fprintf(stderr, "Unsupported Compute Capability (CC) detected (%d.%d). Supported Compute Capabilities are between 3.0 and 7.5.\n", deviceProp.major, deviceProp.minor);
		return 0;
	}

	CUDA_grid_dim=deviceProp.multiProcessorCount*SMXBlock;

	if (!checkex)
	{
		fprintf(stderr, "Grid dim: %d = %d*%d\n",CUDA_grid_dim,deviceProp.multiProcessorCount,SMXBlock);
		fprintf(stderr, "Block dim: %d\n", CUDA_BLOCK_DIM);
	}

	cudaError_t res;

	//Global parameters
	res=cudaMemcpyToSymbol(CUDA_beta_pole,beta_pole,sizeof(double)*(N_POLES+1));
	res=cudaMemcpyToSymbol(CUDA_lambda_pole,lambda_pole,sizeof(double)*(N_POLES+1));
	res=cudaMemcpyToSymbol(CUDA_par,par,sizeof(double)*4);
	res=cudaMemcpyToSymbol(CUDA_cl,&cl,sizeof(cl));
	res=cudaMemcpyToSymbol(CUDA_Alamda_start,&Alamda_start,sizeof(Alamda_start));
	res=cudaMemcpyToSymbol(CUDA_Alamda_incr,&Alamda_incr,sizeof(Alamda_incr));
	res=cudaMemcpyToSymbol(CUDA_Mmax,&m_max,sizeof(m_max));
	res=cudaMemcpyToSymbol(CUDA_Lmax,&l_max,sizeof(l_max));
	res=cudaMemcpyToSymbol(CUDA_tim,tim,sizeof(double)*(MAX_N_OBS+1));
	res=cudaMemcpyToSymbol(CUDA_Phi_0,&Phi_0,sizeof(Phi_0));

	res=cudaMalloc(&pWeight,(ndata+3+1)*sizeof(double));
	res=cudaMemcpy(pWeight,weight,(ndata+3+1)*sizeof(double),cudaMemcpyHostToDevice);
	res=cudaBindTexture(0, texWeight, pWeight, (ndata+3+1)*sizeof(double));

	res=cudaMalloc(&pee,(ndata+1)*3*sizeof(double));
	res=cudaMemcpy(pee,ee,(ndata+1)*3*sizeof(double),cudaMemcpyHostToDevice);
	res=cudaMemcpyToSymbol(CUDA_ee,&pee,sizeof(void*));

	res=cudaMalloc(&pee0,(ndata+1)*3*sizeof(double));
	res=cudaMemcpy(pee0,ee0,(ndata+1)*3*sizeof(double),cudaMemcpyHostToDevice);
	res=cudaMemcpyToSymbol(CUDA_ee0,&pee0,sizeof(void*));

	if (res==cudaSuccess) return 1; else return 0;
}

void CUDAUnprepare(void)
{
	cudaUnbindTexture(texWeight);
	cudaFree(pee);
	cudaFree(pee0);
	cudaFree(pWeight);
}

__global__ void CUDACalculatePrepare(int n_start,int n_max,double freq_start,double freq_step)
{
	int thidx=blockIdx.x;
	int n=n_start+thidx;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];
	freq_result *CUDA_LFR=&CUDA_FR[thidx];

	//zero context
//	CUDA_CC is zeroed itself as global memory but need to reset between freq TODO
	if (n>n_max)
	{
        (*CUDA_LCC).isInvalid=1;
		return;
	}
	else
	{
		(*CUDA_LCC).isInvalid=0;
	}

	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;

        /* initial poles */
	(*CUDA_LFR).per_best = 0;
	(*CUDA_LFR).dark_best = 0;
	(*CUDA_LFR).la_best = 0;
	(*CUDA_LFR).be_best = 0;
	(*CUDA_LFR).dev_best = 1e40;
}

__global__ void CUDACalculatePreparePole(int m)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];
	freq_result *CUDA_LFR=&CUDA_FR[thidx];
	double prd;
	int i;

	if ((*CUDA_LCC).isInvalid)
	{
		atomicAdd(&CUDA_End,1);
		(*CUDA_LFR).isReported=0; //signal not to read result
		return;
	}

	prd = 1 / (*CUDA_LCC).freq;
            /* starts from the initial ellipsoid */
    for (i = 1; i <= CUDA_Ncoef; i++)
       (*CUDA_LCC).cg[i] = CUDA_cg_first[i];

	(*CUDA_LCC).cg[CUDA_Ncoef+1] = CUDA_beta_pole[m];
	(*CUDA_LCC).cg[CUDA_Ncoef+2] = CUDA_lambda_pole[m];

	/* The formulas use beta measured from the pole */
	(*CUDA_LCC).cg[CUDA_Ncoef+1] = 90 - (*CUDA_LCC).cg[CUDA_Ncoef+1];
	/* conversion of lambda, beta to radians */
    (*CUDA_LCC).cg[CUDA_Ncoef+1] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef+1];
    (*CUDA_LCC).cg[CUDA_Ncoef+2] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef+2];

    /* Use omega instead of period */
	(*CUDA_LCC).cg[CUDA_Ncoef+3] = 24 * 2 * PI / prd;

    for (i = 1; i <= CUDA_Nphpar; i++)
    {
        (*CUDA_LCC).cg[CUDA_Ncoef+3+i] = CUDA_par[i];
//              ia[Ncoef+3+i] = ia_par[i]; moved to global
   	}
        /* Lommel-Seeliger part */
    (*CUDA_LCC).cg[CUDA_Ncoef+3+CUDA_Nphpar+2] = 1;
        /* Use logarithmic formulation for Lambert to keep it positive */
	(*CUDA_LCC).cg[CUDA_Ncoef+3+CUDA_Nphpar+1] = log(CUDA_cl);

    	/* Levenberg-Marquardt loop */
		// moved to global iter_max,iter_min,iter_dif_max
		//
	(*CUDA_LCC).rchisq = -1;
    (*CUDA_LCC).Alamda = -1;
    (*CUDA_LCC).Niter = 0;
    (*CUDA_LCC).iter_diff = 1e40;
    (*CUDA_LCC).dev_old = 1e30;
    (*CUDA_LCC).dev_new = 0;
//	(*CUDA_LCC).Lastcall=0; always ==0
	(*CUDA_LFR).isReported=0;
}

__global__ void CUDACalculateIter1_Begin(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];
	freq_result *CUDA_LFR=&CUDA_FR[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	(*CUDA_LCC).isNiter=(((*CUDA_LCC).Niter < CUDA_n_iter_max) && ((*CUDA_LCC).iter_diff > CUDA_iter_diff_max)) || ((*CUDA_LCC).Niter < CUDA_n_iter_min);

	if ((*CUDA_LCC).isNiter)
    {
		if ((*CUDA_LCC).Alamda<0)
		{
			(*CUDA_LCC).isAlamda=1;
			(*CUDA_LCC).Alamda = CUDA_Alamda_start; /* initial alambda */
		}
		else
			(*CUDA_LCC).isAlamda=0;
	}
	else
	{
        if (!(*CUDA_LFR).isReported)
		{
			atomicAdd(&CUDA_End,1);
			(*CUDA_LFR).isReported=1;
		}
	}

}

__global__ void CUDACalculateIter1_mrqmin1_end(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	/*gauss_err=*/mrqmin_1_end(CUDA_LCC);
}

__global__ void CUDACalculateIter1_mrqmin2_end(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqmin_2_end(CUDA_LCC,CUDA_ia,CUDA_ma);
	(*CUDA_LCC).Niter++;
}

__global__ void CUDACalculateIter1_mrqcof1_start(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_start(CUDA_LCC,(*CUDA_LCC).cg,(*CUDA_LCC).alpha,(*CUDA_LCC).beta);
}

__global__ void CUDACalculateIter1_mrqcof1_matrix(int Lpoints)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_matrix(CUDA_LCC,(*CUDA_LCC).cg,Lpoints);
}

__global__ void CUDACalculateIter1_mrqcof1_curve1(int Inrel,int Lpoints)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1(CUDA_LCC,(*CUDA_LCC).cg,(*CUDA_LCC).alpha,(*CUDA_LCC).beta,Inrel,Lpoints);
}

__global__ void CUDACalculateIter1_mrqcof1_curve1_last(int Inrel,int Lpoints)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1_last(CUDA_LCC,(*CUDA_LCC).cg,(*CUDA_LCC).alpha,(*CUDA_LCC).beta,Inrel,Lpoints);
}

__global__ void CUDACalculateIter1_mrqcof1_end(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	(*CUDA_LCC).Ochisq=mrqcof_end(CUDA_LCC,(*CUDA_LCC).alpha);
}

__global__ void CUDACalculateIter1_mrqcof2_start(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_start(CUDA_LCC,(*CUDA_LCC).atry,(*CUDA_LCC).covar,(*CUDA_LCC).da);
}

__global__ void CUDACalculateIter1_mrqcof2_matrix(int Lpoints)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_matrix(CUDA_LCC,(*CUDA_LCC).atry,Lpoints);
}

__global__ void CUDACalculateIter1_mrqcof2_curve1(int Inrel,int Lpoints)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve1(CUDA_LCC,(*CUDA_LCC).atry,(*CUDA_LCC).covar,(*CUDA_LCC).da,Inrel,Lpoints);
}

__global__ void CUDACalculateIter1_mrqcof2_curve1_last(int Inrel,int Lpoints)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve1_last(CUDA_LCC,(*CUDA_LCC).atry,(*CUDA_LCC).covar,(*CUDA_LCC).da,Inrel,Lpoints);
}

__global__ void CUDACalculateIter1_mrqcof2_end(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	(*CUDA_LCC).Chisq=mrqcof_end(CUDA_LCC,(*CUDA_LCC).covar);
}


__global__ void CUDACalculateIter2(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];
//	freq_result *CUDA_LFR=&CUDA_FR[thidx];
	int i,j;

	if ((*CUDA_LCC).isInvalid) return;

	if ((*CUDA_LCC).isNiter)
    {
        if (((*CUDA_LCC).Niter == 1) || ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq))
        {
			if (threadIdx.x==0)
			{
				(*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
			}
			__syncthreads();

			int brtmph,brtmpl;
			brtmph=CUDA_Numfac/CUDA_BLOCK_DIM;
			if(CUDA_Numfac%CUDA_BLOCK_DIM) brtmph++;
			brtmpl=threadIdx.x*brtmph;
			brtmph=brtmpl+brtmph;
			if (brtmph>CUDA_Numfac) brtmph=CUDA_Numfac;
			brtmpl++;

			curv(CUDA_LCC,(*CUDA_LCC).cg,brtmpl,brtmph);

			if (threadIdx.x==0)
			{
				for (i = 1; i <= 3; i++)
				{
					(*CUDA_LCC).chck[i] = 0;
					for (j = 1; j <= CUDA_Numfac; j++)
						(*CUDA_LCC).chck[i] = (*CUDA_LCC).chck[i] + (*CUDA_LCC).Area[j] * CUDA_Nor[j][i-1];
				}
				(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - (pow((*CUDA_LCC).chck[1],2) + pow((*CUDA_LCC).chck[2],2) + pow((*CUDA_LCC).chck[3],2)) * pow(CUDA_conw_r,2);
			}
        }
		if (threadIdx.x==0)
		{
			(*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / (CUDA_ndata - 3));
			/* only if this step is better than the previous,
				1e-10 is for numeric errors */
			if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
			{
				(*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
				(*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
			}
	//		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
		}
    }
}

__global__ void CUDACalculateFinishPole(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];
	freq_result *CUDA_LFR=&CUDA_FR[thidx];
	double totarea,sum,dark,prd,la_tmp,be_tmp;
	int i;

	if ((*CUDA_LCC).isInvalid) return;

    totarea = 0;
    for (i = 1; i <= CUDA_Numfac; i++)
        totarea = totarea + (*CUDA_LCC).Area[i];
    sum = pow((*CUDA_LCC).chck[1],2) + pow((*CUDA_LCC).chck[2],2) + pow((*CUDA_LCC).chck[3],2);
    dark = sqrt(sum);

    /* period solution */
    prd = 2 * PI / (*CUDA_LCC).cg[CUDA_Ncoef+3];

	/* pole solution */
	la_tmp = RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef+2];
	be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef+1];

	if ((*CUDA_LCC).dev_new < (*CUDA_LFR).dev_best)
	{
	    (*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
	    (*CUDA_LFR).per_best = prd;
	    (*CUDA_LFR).dark_best = dark / totarea * 100;
	    (*CUDA_LFR).la_best = la_tmp;
	    (*CUDA_LFR).be_best = be_tmp;
	}
	//debug
/*	(*CUDA_LFR).dark=dark;
	(*CUDA_LFR).totarea=totarea;
	(*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
	(*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
	(*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
}

__global__ void CUDACalculateFinish(void)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];
	freq_result *CUDA_LFR=&CUDA_FR[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if ((*CUDA_LFR).la_best < 0)
	   (*CUDA_LFR).la_best += 360;

	if (isnan((*CUDA_LFR).dark_best) == 1)
	    (*CUDA_LFR).dark_best = 1.0;
}

int CUDAPrecalc(double freq_start,double freq_end,double freq_step,double stop_condition,int n_iter_min,double *conw_r,
	            int ndata,int *ia,int *ia_par,int *new_conw,double *cg_first,double *sig,int Numfac,double *brightness)
{
    int max_test_periods,iC,theEnd;
	double sum_dark_facet,ave_dark_facet;
	int i,n,m,n_max=(int) ((freq_start - freq_end) / freq_step) + 1;
	int n_iter_max;
	double iter_diff_max;
	freq_result *res;
	void *pcc,*pfr,*pbrightness,*psig;

     max_test_periods = 10;
     sum_dark_facet = 0.0;
     ave_dark_facet = 0.0;

     if (n_max < max_test_periods)
		max_test_periods = n_max;

    for (i = 1; i <= n_ph_par; i++)
    {
        ia[n_coef+3+i] = ia_par[i];
    }

    n_iter_max = 0;
    iter_diff_max = -1;
    if (stop_condition > 1)
    {
        n_iter_max = (int) stop_condition;
        iter_diff_max = 0;
		n_iter_min = 0; /* to not overwrite the n_iter_max value */
    }
    if (stop_condition < 1)
    {
        n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
        iter_diff_max = stop_condition;
    }

	cudaError_t err;

	//here move data to device
	cudaMemcpyToSymbol(CUDA_Ncoef,&n_coef,sizeof(n_coef));
	cudaMemcpyToSymbol(CUDA_Nphpar,&n_ph_par,sizeof(n_ph_par));
	cudaMemcpyToSymbol(CUDA_Numfac,&Numfac,sizeof(Numfac));
	m=Numfac+1;
	cudaMemcpyToSymbol(CUDA_Numfac1,&m,sizeof(m));
	cudaMemcpyToSymbol(CUDA_ia,ia,sizeof(int)*(MAX_N_PAR+1));
	cudaMemcpyToSymbol(CUDA_cg_first,cg_first,sizeof(double)*(MAX_N_PAR+1));
	cudaMemcpyToSymbol(CUDA_n_iter_max,&n_iter_max,sizeof(n_iter_max));
	cudaMemcpyToSymbol(CUDA_n_iter_min,&n_iter_min,sizeof(n_iter_min));
	cudaMemcpyToSymbol(CUDA_ndata,&ndata,sizeof(ndata));
	cudaMemcpyToSymbol(CUDA_iter_diff_max,&iter_diff_max,sizeof(iter_diff_max));
	cudaMemcpyToSymbol(CUDA_conw_r,conw_r,sizeof(conw_r));
	cudaMemcpyToSymbol(CUDA_Nor,normal,sizeof(double)*(MAX_N_FAC+1)*3);
	cudaMemcpyToSymbol(CUDA_Fc,f_c,sizeof(double)*(MAX_N_FAC+1)*(MAX_LM+1));
	cudaMemcpyToSymbol(CUDA_Fs,f_s,sizeof(double)*(MAX_N_FAC+1)*(MAX_LM+1));
	cudaMemcpyToSymbol(CUDA_Pleg,pleg,sizeof(double)*(MAX_N_FAC+1)*(MAX_LM+1)*(MAX_LM+1));
	cudaMemcpyToSymbol(CUDA_Darea,d_area,sizeof(double)*(MAX_N_FAC+1));
	cudaMemcpyToSymbol(CUDA_Dsph,d_sphere,sizeof(double)*(MAX_N_FAC+1)*(MAX_N_PAR+1));

	err=cudaMalloc(&pbrightness,(ndata+1)*sizeof(double));
	err=cudaMemcpy(pbrightness,brightness,(ndata+1)*sizeof(double),cudaMemcpyHostToDevice);
	err=cudaBindTexture(0, texbrightness, pbrightness, (ndata+1)*sizeof(double));

	err=cudaMalloc(&psig,(ndata+1)*sizeof(double));
	err=cudaMemcpy(psig,sig,(ndata+1)*sizeof(double),cudaMemcpyHostToDevice);
	err=cudaBindTexture(0, texsig, psig, (ndata+1)*sizeof(double));

	/* number of fitted parameters */
	int lmfit=0,llastma=0,llastone=1,ma=n_coef+5+n_ph_par;
		 for (m = 1; m <= ma; m++)
		 {
		  if (ia[m])
		  {
			lmfit++;
			llastma=m;
		  }
		 }
		 llastone=1;
		 for (m = 2; m <=llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
		 {
		  if (!ia[m]) break;
		  llastone=m;
		 }
	cudaMemcpyToSymbol(CUDA_ma,&ma,sizeof(ma));
 	cudaMemcpyToSymbol(CUDA_mfit,&lmfit,sizeof(lmfit));
	m=lmfit+1;
	cudaMemcpyToSymbol(CUDA_mfit1,&m,sizeof(m));
	cudaMemcpyToSymbol(CUDA_lastma,&llastma,sizeof(llastma));
	cudaMemcpyToSymbol(CUDA_lastone,&llastone,sizeof(llastone));
	m=ma-2-n_ph_par;
	cudaMemcpyToSymbol(CUDA_ncoef0,&m,sizeof(m));

	int CUDA_Grid_dim_precalc=CUDA_grid_dim;
	if (max_test_periods<CUDA_Grid_dim_precalc) CUDA_Grid_dim_precalc=max_test_periods;

	err=cudaMalloc(&pcc,CUDA_Grid_dim_precalc*sizeof(freq_context));
	cudaMemcpyToSymbol(CUDA_CC,&pcc,sizeof(pcc));
	err=cudaMalloc(&pfr,CUDA_Grid_dim_precalc*sizeof(freq_result));
	cudaMemcpyToSymbol(CUDA_FR,&pfr,sizeof(pfr));

	m=(Numfac+1)*(n_coef+1);
	cudaMemcpyToSymbol(CUDA_Dg_block,&m,sizeof(m));

    double *pa,*pg,*pal,*pco,*pdytemp,*pytemp;

	err=cudaMalloc(&pa,CUDA_Grid_dim_precalc*(Numfac+1)*sizeof(double));
	err=cudaBindTexture(0, texArea, pa, CUDA_Grid_dim_precalc*(Numfac+1)*sizeof(double));
	err=cudaMalloc(&pg,CUDA_Grid_dim_precalc*(Numfac+1)*(n_coef+1)*sizeof(double));
	err=cudaBindTexture(0, texDg, pg, CUDA_Grid_dim_precalc*(Numfac+1)*(n_coef+1)*sizeof(double));
	err=cudaMalloc(&pal,CUDA_Grid_dim_precalc*(lmfit+1)*(lmfit+1)*sizeof(double));
	err=cudaMalloc(&pco,CUDA_Grid_dim_precalc*(lmfit+1)*(lmfit+1)*sizeof(double));
	err=cudaMalloc(&pdytemp,CUDA_Grid_dim_precalc*(max_l_points+1)*(ma+1)*sizeof(double));
	err=cudaMalloc(&pytemp,CUDA_Grid_dim_precalc*(max_l_points+1)*sizeof(double));

	for (m=0;m<CUDA_Grid_dim_precalc;m++)
	{
		freq_context ps;
		ps.Area=&pa[m*(Numfac+1)];
		ps.Dg=&pg[m*(Numfac+1)*(n_coef+1)];
		ps.alpha=&pal[m*(lmfit+1)*(lmfit+1)];
		ps.covar=&pco[m*(lmfit+1)*(lmfit+1)];
		ps.dytemp=&pdytemp[m*(max_l_points+1)*(ma+1)];
		ps.ytemp=&pytemp[m*(max_l_points+1)];
		freq_context *pt=&((freq_context *)pcc)[m];
		err=cudaMemcpy(pt,&ps,sizeof(void *)*6,cudaMemcpyHostToDevice);
	}

	res=(freq_result *)malloc(CUDA_Grid_dim_precalc*sizeof(freq_result));

	for (n=1;n<=max_test_periods;n+=CUDA_Grid_dim_precalc)
	{
        CUDACalculatePrepare<<<CUDA_Grid_dim_precalc,1>>>(n,max_test_periods,freq_start,freq_step);
		err=cudaThreadSynchronize();

		for (m = 1; m <= N_POLES; m++)
		{
			//zero global End signal
			theEnd=0;
			cudaMemcpyToSymbol(CUDA_End,&theEnd,sizeof(theEnd));
			//
			CUDACalculatePreparePole<<<CUDA_Grid_dim_precalc,1>>>(m);
			//
			while (!theEnd)
			{
				CUDACalculateIter1_Begin<<<CUDA_Grid_dim_precalc,1>>>();
				//mrqcof
				CUDACalculateIter1_mrqcof1_start<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>();
				for (iC=1;iC<l_curves;iC++)
				{
						CUDACalculateIter1_mrqcof1_matrix<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(l_points[iC]);
						CUDACalculateIter1_mrqcof1_curve1<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
						CUDACalculateIter1_mrqcof1_curve2<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
				}
				CUDACalculateIter1_mrqcof1_curve1_last<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof1_curve2<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof1_end<<<CUDA_Grid_dim_precalc,1>>>();
				//mrqcof
				CUDACalculateIter1_mrqmin1_end<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>();
				//mrqcof
				CUDACalculateIter1_mrqcof2_start<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>();
				for (iC=1;iC<l_curves;iC++)
				{
						CUDACalculateIter1_mrqcof2_matrix<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(l_points[iC]);
						CUDACalculateIter1_mrqcof2_curve1<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
						CUDACalculateIter1_mrqcof2_curve2<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
				}
				CUDACalculateIter1_mrqcof2_curve1_last<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof2_curve2<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof2_end<<<CUDA_Grid_dim_precalc,1>>>();
				//mrqcof
				CUDACalculateIter1_mrqmin2_end<<<CUDA_Grid_dim_precalc,1>>>();
				CUDACalculateIter2<<<CUDA_Grid_dim_precalc,CUDA_BLOCK_DIM>>>();
				//err=cudaThreadSynchronize(); memcpy is synchro itself
				cudaMemcpyFromSymbol(&theEnd,CUDA_End,sizeof(theEnd));
				theEnd=theEnd==CUDA_Grid_dim_precalc;

				//break;//debug
			}
			CUDACalculateFinishPole<<<CUDA_Grid_dim_precalc,1>>>();
			err=cudaThreadSynchronize();
//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_Grid_dim_precalc);
			//break; //debug
		}

		CUDACalculateFinish<<<CUDA_Grid_dim_precalc,1>>>();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		err=cudaMemcpy(res,pfr,sizeof(freq_result)*CUDA_Grid_dim_precalc,cudaMemcpyDeviceToHost);

		for (m=1; m <= CUDA_Grid_dim_precalc; m++)
		{
		  if (res[m-1].isReported==1)
			sum_dark_facet = sum_dark_facet + res[m-1].dark_best;
		}
   } /* period loop */

	cudaUnbindTexture(texArea);
	cudaUnbindTexture(texDg);
	cudaUnbindTexture(texbrightness);
	cudaUnbindTexture(texsig);
	cudaFree(pa);
	cudaFree(pg);
	cudaFree(pal);
	cudaFree(pco);
	cudaFree(pdytemp);
	cudaFree(pytemp);
	cudaFree(pcc);
	cudaFree(pfr);
	cudaFree(pbrightness);
	cudaFree(psig);

	free((void *)res);

	ave_dark_facet = sum_dark_facet / max_test_periods;

	if ( ave_dark_facet < 1.0 )
		*new_conw = 1; /* new correct conwexity weight */
	if ( ave_dark_facet >= 1.0 )
		*conw_r = *conw_r * 2; /* still not good */

	return 1;
}

void GetCUDAOccupancy(const int cudaDevice)
{
	int numBlocks;        // Occupancy in terms of active blocks
	const auto blockSize = CUDA_BLOCK_DIM;

	//cudaGetDevice(&cudaDevice);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, cudaDevice);

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocks,
		CUDACalculateIter1_mrqcof1_curve2,
		blockSize,
		0);

	const auto activeWarps = numBlocks * blockSize / deviceProp.warpSize;
	const auto maxWarps = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;

	const auto ocupancy = static_cast<double>(activeWarps) / maxWarps * 100;

	fprintf(stderr, "Occupancy for kernel \"CUDACalculateIter1_mrqcof1_curve2\": %f%%\n", ocupancy);

	//std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
}

int CUDAStart(int n_start_from,double freq_start,double freq_end,double freq_step,double stop_condition,int n_iter_min,double conw_r,
	          int ndata,int *ia,int *ia_par,double *cg_first,MFILE& mf,double escl,double *sig,int Numfac,double *brightness)
{
	int retval,i,n,m,iC,n_max=(int) ((freq_start - freq_end) / freq_step) + 1;
	int n_iter_max,theEnd,LinesWritten;
	double iter_diff_max;
	freq_result *res;
	void *pcc,*pfr,*pbrightness,*psig;
	char buf[256];

    for (i = 1; i <= n_ph_par; i++)
    {
        ia[n_coef+3+i] = ia_par[i];
    }

    n_iter_max = 0;
    iter_diff_max = -1;
    if (stop_condition > 1)
    {
        n_iter_max = (int) stop_condition;
        iter_diff_max = 0;
		n_iter_min = 0; /* to not overwrite the n_iter_max value */
    }
    if (stop_condition < 1)
    {
        n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
        iter_diff_max = stop_condition;
    }

	cudaError_t err;

	//here move data to device
	cudaMemcpyToSymbol(CUDA_Ncoef,&n_coef,sizeof(n_coef));
	cudaMemcpyToSymbol(CUDA_Nphpar,&n_ph_par,sizeof(n_ph_par));
	cudaMemcpyToSymbol(CUDA_Numfac,&Numfac,sizeof(Numfac));
	m=Numfac+1;
	cudaMemcpyToSymbol(CUDA_Numfac1,&m,sizeof(m));
	cudaMemcpyToSymbol(CUDA_ia,ia,sizeof(int)*(MAX_N_PAR+1));
	cudaMemcpyToSymbol(CUDA_cg_first,cg_first,sizeof(double)*(MAX_N_PAR+1));
	cudaMemcpyToSymbol(CUDA_n_iter_max,&n_iter_max,sizeof(n_iter_max));
	cudaMemcpyToSymbol(CUDA_n_iter_min,&n_iter_min,sizeof(n_iter_min));
	cudaMemcpyToSymbol(CUDA_ndata,&ndata,sizeof(ndata));
	cudaMemcpyToSymbol(CUDA_iter_diff_max,&iter_diff_max,sizeof(iter_diff_max));
	cudaMemcpyToSymbol(CUDA_conw_r,&conw_r,sizeof(conw_r));
	cudaMemcpyToSymbol(CUDA_Nor,normal,sizeof(double)*(MAX_N_FAC+1)*3);
	cudaMemcpyToSymbol(CUDA_Fc,f_c,sizeof(double)*(MAX_N_FAC+1)*(MAX_LM+1));
	cudaMemcpyToSymbol(CUDA_Fs,f_s,sizeof(double)*(MAX_N_FAC+1)*(MAX_LM+1));
	cudaMemcpyToSymbol(CUDA_Pleg,pleg,sizeof(double)*(MAX_N_FAC+1)*(MAX_LM+1)*(MAX_LM+1));
	cudaMemcpyToSymbol(CUDA_Darea,d_area,sizeof(double)*(MAX_N_FAC+1));
	cudaMemcpyToSymbol(CUDA_Dsph,d_sphere,sizeof(double)*(MAX_N_FAC+1)*(MAX_N_PAR+1));

	err=cudaMalloc(&pbrightness,(ndata+1)*sizeof(double));
	err=cudaMemcpy(pbrightness,brightness,(ndata+1)*sizeof(double),cudaMemcpyHostToDevice);
	err=cudaBindTexture(0, texbrightness, pbrightness, (ndata+1)*sizeof(double));

	err=cudaMalloc(&psig,(ndata+1)*sizeof(double));
	err=cudaMemcpy(psig,sig,(ndata+1)*sizeof(double),cudaMemcpyHostToDevice);
	err=cudaBindTexture(0, texsig, psig, (ndata+1)*sizeof(double));

	/* number of fitted parameters */
	int lmfit=0,llastma=0,llastone=1,ma=n_coef+5+n_ph_par;
		 for (m = 1; m <= ma; m++)
		 {
		  if (ia[m])
		  {
			lmfit++;
			llastma=m;
		  }
		 }
		 llastone=1;
		 for (m = 2; m <=llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
		 {
		  if (!ia[m]) break;
		  llastone=m;
		 }
	cudaMemcpyToSymbol(CUDA_ma,&ma,sizeof(ma));
	cudaMemcpyToSymbol(CUDA_mfit,&lmfit,sizeof(lmfit));
	m=lmfit+1;
	cudaMemcpyToSymbol(CUDA_mfit1,&m,sizeof(m));
	cudaMemcpyToSymbol(CUDA_lastma,&llastma,sizeof(llastma));
	cudaMemcpyToSymbol(CUDA_lastone,&llastone,sizeof(llastone));
	m=ma-2-n_ph_par;
	cudaMemcpyToSymbol(CUDA_ncoef0,&m,sizeof(m));

	err=cudaMalloc(&pcc,CUDA_grid_dim*sizeof(freq_context));
	cudaMemcpyToSymbol(CUDA_CC,&pcc,sizeof(pcc));
	err=cudaMalloc(&pfr,CUDA_grid_dim*sizeof(freq_result));
	cudaMemcpyToSymbol(CUDA_FR,&pfr,sizeof(pfr));

	m=(Numfac+1)*(n_coef+1);
	cudaMemcpyToSymbol(CUDA_Dg_block,&m,sizeof(m));

    double *pa,*pg,*pal,*pco,*pdytemp,*pytemp;

	err=cudaMalloc(&pa,CUDA_grid_dim*(Numfac+1)*sizeof(double));
	err=cudaBindTexture(0, texArea, pa, CUDA_grid_dim*(Numfac+1)*sizeof(double));
	err=cudaMalloc(&pg,CUDA_grid_dim*(Numfac+1)*(n_coef+1)*sizeof(double));
	err=cudaBindTexture(0, texDg, pg, CUDA_grid_dim*(Numfac+1)*(n_coef+1)*sizeof(double));
	err=cudaMalloc(&pal,CUDA_grid_dim*(lmfit+1)*(lmfit+1)*sizeof(double));
	err=cudaMalloc(&pco,CUDA_grid_dim*(lmfit+1)*(lmfit+1)*sizeof(double));
	err=cudaMalloc(&pdytemp,CUDA_grid_dim*(max_l_points+1)*(ma+1)*sizeof(double));
	err=cudaMalloc(&pytemp,CUDA_grid_dim*(max_l_points+1)*sizeof(double));

	for (m=0;m<CUDA_grid_dim;m++)
	{
		freq_context ps;
		ps.Area=&pa[m*(Numfac+1)];
		ps.Dg=&pg[m*(Numfac+1)*(n_coef+1)];
		ps.alpha=&pal[m*(lmfit+1)*(lmfit+1)];
		ps.covar=&pco[m*(lmfit+1)*(lmfit+1)];
		ps.dytemp=&pdytemp[m*(max_l_points+1)*(ma+1)];
		ps.ytemp=&pytemp[m*(max_l_points+1)];
		freq_context *pt=&((freq_context *)pcc)[m];
		err=cudaMemcpy(pt,&ps,sizeof(void *)*6,cudaMemcpyHostToDevice);
	}

	res=(freq_result *)malloc(CUDA_grid_dim*sizeof(freq_result));

	int firstreport=0;//beta debug

	for (n=n_start_from;n<=n_max;n+=CUDA_grid_dim)
	{
		auto fractionDone = (double)n / (double)n_max;
		boinc_fraction_done(fractionDone);

//#if _DEBUG
		float fraction = fractionDone * 100;
			printf("Fraction done: %.2f%%\n", fraction);
//#endif

        CUDACalculatePrepare<<<CUDA_grid_dim,1>>>(n,n_max,freq_start,freq_step);
		err=cudaThreadSynchronize();

		for (m = 1; m <= N_POLES; m++)
		{
			//zero global End signal
			theEnd=0;
			cudaMemcpyToSymbol(CUDA_End,&theEnd,sizeof(theEnd));
			//
			CUDACalculatePreparePole<<<CUDA_grid_dim,1>>>(m);
			//
			while (!theEnd)
			{
				CUDACalculateIter1_Begin<<<CUDA_grid_dim,1>>>();
				//mrqcof
				CUDACalculateIter1_mrqcof1_start<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>();
				for (iC=1;iC<l_curves;iC++)
				{
					CUDACalculateIter1_mrqcof1_matrix<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(l_points[iC]);
					CUDACalculateIter1_mrqcof1_curve1<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
					CUDACalculateIter1_mrqcof1_curve2<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
				}
				CUDACalculateIter1_mrqcof1_curve1_last<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof1_curve2<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof1_end<<<CUDA_grid_dim,1>>>();
				//mrqcof
				CUDACalculateIter1_mrqmin1_end<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>();
				//mrqcof
				CUDACalculateIter1_mrqcof2_start<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>();
				for (iC=1;iC<l_curves;iC++)
				{
					CUDACalculateIter1_mrqcof2_matrix<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(l_points[iC]);
					CUDACalculateIter1_mrqcof2_curve1<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
					CUDACalculateIter1_mrqcof2_curve2<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[iC],l_points[iC]);
				}
				CUDACalculateIter1_mrqcof2_curve1_last<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof2_curve2<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>(in_rel[l_curves],l_points[l_curves]);
				CUDACalculateIter1_mrqcof2_end<<<CUDA_grid_dim,1>>>();
				//mrqcof
				CUDACalculateIter1_mrqmin2_end<<<CUDA_grid_dim,1>>>();
				CUDACalculateIter2<<<CUDA_grid_dim,CUDA_BLOCK_DIM>>>();
				//err=cudaThreadSynchronize(); memcpy is synchro itself
				cudaMemcpyFromSymbol(&theEnd,CUDA_End,sizeof(theEnd));
				theEnd=theEnd==CUDA_grid_dim;

				//break;//debug
			}
			CUDACalculateFinishPole<<<CUDA_grid_dim,1>>>();
			err=cudaThreadSynchronize();
//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_grid_dim);
//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_grid_dim);
			//break; //debug
		}

		CUDACalculateFinish<<<CUDA_grid_dim,1>>>();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		err=cudaMemcpy(res,pfr,sizeof(freq_result)*CUDA_grid_dim,cudaMemcpyDeviceToHost);

		LinesWritten=0;
		for (m = 1; m <=CUDA_grid_dim ; m++)
		{
			if (res[m-1].isReported==1)
			{
				LinesWritten++;
				/* output file */
				if (( n ==1 ) && (m==1))
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m-1].per_best, res[m-1].dev_best, res[m-1].dev_best * res[m-1].dev_best * (ndata - 3), conw_r * escl * escl, res[m-1].la_best, res[m-1].be_best);
				else
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m-1].per_best, res[m-1].dev_best, res[m-1].dev_best * res[m-1].dev_best * (ndata - 3), res[m-1].dark_best, res[m-1].la_best, res[m-1].be_best);
			}
		}
		 if (boinc_time_to_checkpoint() || boinc_is_standalone()) {
			retval = DoCheckpoint(mf, (n-1)+LinesWritten,1,conw_r); //zero lines
			if (retval) {fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval);}
			boinc_checkpoint_completed();
		 }

//		break;//debug
    } /* period loop */

	cudaUnbindTexture(texArea);
	cudaUnbindTexture(texDg);
	cudaUnbindTexture(texbrightness);
	cudaUnbindTexture(texsig);
	cudaFree(pa);
	cudaFree(pg);
	cudaFree(pal);
	cudaFree(pco);
	cudaFree(pdytemp);
	cudaFree(pytemp);
    cudaFree(pcc);
	cudaFree(pfr);
	cudaFree(pbrightness);
	cudaFree(psig);

	free((void *)res);

	return 1;
}
