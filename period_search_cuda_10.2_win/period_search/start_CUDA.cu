#include <cuda.h>
#include <cstdio>
#include "mfile.h"
#include "globals.h"
#include "globals_CUDA.h"
#include "start_CUDA.h"
#include "declarations_CUDA.h"
#include "boinc_api.h"
#include "Start.cuh"
//#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
//#include <cuda_occupancy.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>

#ifdef __GNUC__
#include <time.h>
#endif

//global to all freq
__constant__ int /*CUDA_n,*/CUDA_Ncoef, CUDA_Nphpar, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ int CUDA_ia[MAX_N_PAR + 1];
__constant__ int CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__device__ double CUDA_cg_first[MAX_N_PAR + 1];
__device__ double CUDA_beta_pole[N_POLES + 1];
__device__ double CUDA_lambda_pole[N_POLES + 1];
__device__ double CUDA_par[4];
//__device__ __constant__ double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__device__ double CUDA_iter_diff_max;
__constant__ double CUDA_Nor[MAX_N_FAC + 1][3];
__constant__ double CUDA_conw_r;
__constant__ int CUDA_Lmax, CUDA_Mmax;
__device__ double CUDA_Fc[MAX_N_FAC + 1][MAX_LM + 1];
__device__ double CUDA_Fs[MAX_N_FAC + 1][MAX_LM + 1];
__device__ double CUDA_Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
__constant__ double CUDA_Darea[MAX_N_FAC + 1];
__device__ double CUDA_Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
__device__ double* CUDA_ee/*[MAX_N_OBS+1][3]*/;
__device__ double* CUDA_ee0/*[MAX_N_OBS+1][3]*/;
__device__ double CUDA_tim[MAX_N_OBS + 1];
//__device__ double CUDA_brightness[MAX_N_OBS+1];
//__device__ double CUDA_sig[MAX_N_OBS+1];
//__device__ double *CUDA_Weight/*[MAX_N_OBS+1]*/;
__constant__ double CUDA_Phi_0;
__device__ int CUDA_End;
__device__ int CUDA_Is_Precalc;

//__device__ int icol;
//__device__ double pivinv;
//__shared__ int sh_icol[CUDA_BLOCK_DIM];
//__shared__ int sh_irow[CUDA_BLOCK_DIM];
//__shared__ double sh_big[CUDA_BLOCK_DIM];

texture<int2, 1> texWeight;
texture<int2, 1> texbrightness;
texture<int2, 1> texsig;

//global to one thread
__device__ freq_context* CUDA_CC;
__device__ freq_result* CUDA_FR;

texture<int2, 1> texArea;
texture<int2, 1> texDg;

int CUDA_grid_dim;
double* pee, * pee0, * pWeight;

// NOTE: https://boinc.berkeley.edu/trac/wiki/CudaApps
bool SetCUDABlockingSync(const int device) {
	CUdevice  hcuDevice;
	CUcontext hcuContext;

	CUresult status = cuInit(0);
	if (status != CUDA_SUCCESS)
		return false;

	status = cuDeviceGet(&hcuDevice, device);
	if (status != CUDA_SUCCESS)
		return false;

	status = cuCtxCreate(&hcuContext, 0x4, hcuDevice);
	if (status != CUDA_SUCCESS)
		return false;

	return true;
}

int CUDAPrepare(int cudadev, double* beta_pole, double* lambda_pole, double* par, double cl, double Alamda_start, double Alamda_incr,
	double ee[][3], double ee0[][3], double* tim, double Phi_0, int checkex, int ndata)
{
	//init gpu
	auto initResult = SetCUDABlockingSync(cudadev);
	if(!initResult)
	{
		fprintf(stderr, "%s CUDA: Error while initialising CUDA\n");
		exit(999);
	}

	cudaSetDevice(cudadev);
	// TODO: Check if this is obsolete when calling SetCUDABlockingSync()
	//cudaSetDeviceFlags(cudaDeviceBlockingSync);

	//determine gridDim
	cudaDeviceProp deviceProp;
	int SMXBlock; // Maximum number of resident thread blocks per multiprocessor
	cudaGetDeviceProperties(&deviceProp, cudadev);
	if (!checkex)
	{
		auto cudaVersion = CUDA_VERSION;
		auto totalGlobalMemory = deviceProp.totalGlobalMem / 1048576;
		auto sharedMemorySm = deviceProp.sharedMemPerMultiprocessor;
		auto sharedMemoryBlock = deviceProp.sharedMemPerBlock;

		fprintf(stderr, "CUDA version: %d\n", cudaVersion);
		fprintf(stderr, "CUDA Device number: %d\n", cudadev);
		fprintf(stderr, "CUDA Device: %s %lluMB \n", deviceProp.name, totalGlobalMemory);
		fprintf(stderr, "Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		fprintf(stderr, "Shared memory per Block | per SM: %llu | %llu\n", sharedMemoryBlock, sharedMemorySm);
		fprintf(stderr, "Multiprocessors: %d\n", deviceProp.multiProcessorCount);

	}

	//int cudaBlockDim = CUDA_BLOCK_DIM;
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
			break;
		case 5:
			SMXBlock = 16;	// CC 7.5, occupancy 100% = 16 blocks per SMX
			break;
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

	CUDA_grid_dim = deviceProp.multiProcessorCount * SMXBlock;

	if (!checkex)
	{
		fprintf(stderr, "Resident blocks per multiprocessor: %d\n", SMXBlock);
		fprintf(stderr, "Grid dim: %d = %d*%d\n", CUDA_grid_dim, deviceProp.multiProcessorCount, SMXBlock);
		fprintf(stderr, "Block dim: %d\n", CUDA_BLOCK_DIM);
	}

	cudaError_t res;

	//Global parameters
	res = cudaMemcpyToSymbol(CUDA_beta_pole, beta_pole, sizeof(double) * (N_POLES + 1));
	res = cudaMemcpyToSymbol(CUDA_lambda_pole, lambda_pole, sizeof(double) * (N_POLES + 1));
	res = cudaMemcpyToSymbol(CUDA_par, par, sizeof(double) * 4);
	res = cudaMemcpyToSymbol(CUDA_cl, &cl, sizeof(cl));
	res = cudaMemcpyToSymbol(CUDA_Alamda_start, &Alamda_start, sizeof(Alamda_start));
	res = cudaMemcpyToSymbol(CUDA_Alamda_incr, &Alamda_incr, sizeof(Alamda_incr));
	res = cudaMemcpyToSymbol(CUDA_Mmax, &m_max, sizeof(m_max));
	res = cudaMemcpyToSymbol(CUDA_Lmax, &l_max, sizeof(l_max));
	res = cudaMemcpyToSymbol(CUDA_tim, tim, sizeof(double) * (MAX_N_OBS + 1));
	res = cudaMemcpyToSymbol(CUDA_Phi_0, &Phi_0, sizeof(Phi_0));

	res = cudaMalloc(&pWeight, (ndata + 3 + 1) * sizeof(double));
	res = cudaMemcpy(pWeight, weight, (ndata + 3 + 1) * sizeof(double), cudaMemcpyHostToDevice);
	res = cudaBindTexture(0, texWeight, pWeight, (ndata + 3 + 1) * sizeof(double));

	res = cudaMalloc(&pee, (ndata + 1) * 3 * sizeof(double));
	res = cudaMemcpy(pee, ee, (ndata + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
	res = cudaMemcpyToSymbol(CUDA_ee, &pee, sizeof(void*));

	res = cudaMalloc(&pee0, (ndata + 1) * 3 * sizeof(double));
	res = cudaMemcpy(pee0, ee0, (ndata + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
	res = cudaMemcpyToSymbol(CUDA_ee0, &pee0, sizeof(void*));

	if (res == cudaSuccess)
	{
		return 1;
	}

	else return 0;
}

void CUDAUnprepare(void)
{
	cudaUnbindTexture(texWeight);
	cudaFree(pee);
	cudaFree(pee0);
	cudaFree(pWeight);
}

int CUDAPrecalc(double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
	int ndata, int* ia, int* ia_par, int* new_conw, double* cg_first, double* sig, int Numfac, double* brightness)
{
	//int* endPtr;
	int max_test_periods, iC, theEnd;
	double sum_dark_facet, ave_dark_facet;
	int i, n, m;
	int n_iter_max;
	double iter_diff_max;
	freq_result* res;
	void* pcc, * pfr, * pbrightness, * psig;

	// NOTE: max_test_periods dictates the CUDA_Grid_dim_precalc value which is actual Threads-per-Block
	/*	Cuda Compute profiler gives the following advice for almost every kernel launched:
		"Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 16 threads per block.
		Consequently, some threads in a warp are masked off and those hardware resources are unused. Try changing the number of threads per block to be a multiple of 32 threads.
		Between 128 and 256 threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one large thread block per multiprocessor
		if latency affects performance. This is particularly beneficial to kernels that frequently call __syncthreads().*/

	max_test_periods = 10; //10;
	sum_dark_facet = 0.0;
	ave_dark_facet = 0.0;

//#ifdef _DEBUG
//	int n_max = (int)((freq_start - freq_end) / freq_step) + 1;
//	if (n_max < max_test_periods)
//	{
//		max_test_periods = n_max;
//		fprintf(stderr, "n_max(%d) < max_test_periods (%d)\n", n_max, max_test_periods);
//	}
//	else
//	{
//		fprintf(stderr, "n_max(%d) > max_test_periods (%d)\n", n_max, max_test_periods);
//	}
//
//	fprintf(stderr, "freq_start (%.3f) - freq_end (%.3f) / freq_step (%.3f) = n_max (%d)\n", freq_start, freq_end, freq_step, n_max);
//#endif

	for (i = 1; i <= n_ph_par; i++)
	{
		ia[n_coef + 3 + i] = ia_par[i];
	}

	n_iter_max = 0;
	iter_diff_max = -1;
	if (stop_condition > 1)
	{
		n_iter_max = (int)stop_condition;
		iter_diff_max = 0;
		n_iter_min = 0; /* to not overwrite the n_iter_max value */
	}
	if (stop_condition < 1)
	{
		n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
		iter_diff_max = stop_condition;
	}

	cudaError_t err;
	int isPrecalc = 1;
	/*int i_col, sh_icol_local[CUDA_BLOCK_DIM], sh_irow_local[CUDA_BLOCK_DIM];
	double piv_inv, sh_big_local[CUDA_BLOCK_DIM];*/

	//here move data to device
	cudaMemcpyToSymbol(CUDA_Ncoef, &n_coef, sizeof(n_coef));
	cudaMemcpyToSymbol(CUDA_Nphpar, &n_ph_par, sizeof(n_ph_par));
	cudaMemcpyToSymbol(CUDA_Numfac, &Numfac, sizeof(Numfac));
	m = Numfac + 1;
	cudaMemcpyToSymbol(CUDA_Numfac1, &m, sizeof(m));
	cudaMemcpyToSymbol(CUDA_ia, ia, sizeof(int) * (MAX_N_PAR + 1));
	cudaMemcpyToSymbol(CUDA_cg_first, cg_first, sizeof(double) * (MAX_N_PAR + 1));
	cudaMemcpyToSymbol(CUDA_n_iter_max, &n_iter_max, sizeof(n_iter_max));
	cudaMemcpyToSymbol(CUDA_n_iter_min, &n_iter_min, sizeof(n_iter_min));
	cudaMemcpyToSymbol(CUDA_ndata, &ndata, sizeof(ndata));
	cudaMemcpyToSymbol(CUDA_iter_diff_max, &iter_diff_max, sizeof(iter_diff_max));
	cudaMemcpyToSymbol(CUDA_conw_r, conw_r, sizeof(conw_r));
	cudaMemcpyToSymbol(CUDA_Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
	cudaMemcpyToSymbol(CUDA_Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	cudaMemcpyToSymbol(CUDA_Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	cudaMemcpyToSymbol(CUDA_Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
	cudaMemcpyToSymbol(CUDA_Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
	cudaMemcpyToSymbol(CUDA_Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	cudaMemcpyToSymbol(CUDA_Is_Precalc, &isPrecalc, sizeof isPrecalc, 0, cudaMemcpyHostToDevice);
	/*cudaMemcpyToSymbol(icol, &i_col, sizeof(i_col));
	cudaMemcpyToSymbol(pivinv, &piv_inv, sizeof(piv_inv));
	cudaMemcpyToSymbol(sh_icol, sh_icol_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_irow, sh_irow_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_big, sh_big_local, sizeof(double) * CUDA_BLOCK_DIM);*/

	err = cudaMalloc(&pbrightness, (ndata + 1) * sizeof(double));
	err = cudaMemcpy(pbrightness, brightness, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	err = cudaBindTexture(0, texbrightness, pbrightness, (ndata + 1) * sizeof(double));

	err = cudaMalloc(&psig, (ndata + 1) * sizeof(double));
	err = cudaMemcpy(psig, sig, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	err = cudaBindTexture(0, texsig, psig, (ndata + 1) * sizeof(double));
	if (err) printf("Error: %s\n", cudaGetErrorString(err));

	/* number of fitted parameters */
	int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
	for (m = 1; m <= ma; m++)
	{
		if (ia[m])
		{
			lmfit++;
			llastma = m;
		}
	}
	llastone = 1;
	for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
	{
		if (!ia[m]) break;
		llastone = m;
	}
	cudaMemcpyToSymbol(CUDA_ma, &ma, sizeof(ma));
	cudaMemcpyToSymbol(CUDA_mfit, &lmfit, sizeof(lmfit));
	m = lmfit + 1;
	cudaMemcpyToSymbol(CUDA_mfit1, &m, sizeof(m));
	cudaMemcpyToSymbol(CUDA_lastma, &llastma, sizeof(llastma));
	cudaMemcpyToSymbol(CUDA_lastone, &llastone, sizeof(llastone));
	m = ma - 2 - n_ph_par;
	cudaMemcpyToSymbol(CUDA_ncoef0, &m, sizeof(m));

	int CUDA_Grid_dim_precalc = CUDA_grid_dim;
	if (max_test_periods < CUDA_Grid_dim_precalc)
	{
		CUDA_Grid_dim_precalc = max_test_periods;
//#ifdef _DEBUG
//		fprintf(stderr, "CUDA_Grid_dim_precalc = %d\n", CUDA_Grid_dim_precalc);
//#endif
	}

	err = cudaMalloc(&pcc, CUDA_Grid_dim_precalc * sizeof(freq_context));
	cudaMemcpyToSymbol(CUDA_CC, &pcc, sizeof(pcc));
	err = cudaMalloc(&pfr, CUDA_Grid_dim_precalc * sizeof(freq_result));
	cudaMemcpyToSymbol(CUDA_FR, &pfr, sizeof(pfr));

	m = (Numfac + 1) * (n_coef + 1);
	cudaMemcpyToSymbol(CUDA_Dg_block, &m, sizeof(m));

	double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;

	err = cudaMalloc(&pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	err = cudaBindTexture(0, texArea, pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	err = cudaMalloc(&pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	err = cudaBindTexture(0, texDg, pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	err = cudaMalloc(&pal, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = cudaMalloc(&pco, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = cudaMalloc(&pdytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double));
	err = cudaMalloc(&pytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * sizeof(double));

	for (m = 0; m < CUDA_Grid_dim_precalc; m++)
	{
		freq_context ps;
		ps.Area = &pa[m * (Numfac + 1)];
		ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
		ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
		ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
		ps.dytemp = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
		ps.ytemp = &pytemp[m * (max_l_points + 1)];
		freq_context* pt = &((freq_context*)pcc)[m];
		err = cudaMemcpy(pt, &ps, sizeof(void*) * 6, cudaMemcpyHostToDevice);
	}

	res = (freq_result*)malloc(CUDA_Grid_dim_precalc * sizeof(freq_result));

	for (n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
	{
		CudaCalculatePrepare<<<CUDA_Grid_dim_precalc, 1>>>(n, max_test_periods, freq_start, freq_step);
		err = cudaDeviceSynchronize();

		for (m = 1; m <= N_POLES; m++)
		{
			//zero global End signal
			theEnd = 0;
			cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd), 0, cudaMemcpyHostToDevice);
			//cudaGetSymbolAddress((void**)&endPtr, CUDA_End);
			//
			CudaCalculatePreparePole<<<CUDA_Grid_dim_precalc, 1>>>(m);
			//
#ifdef _DEBUG
			printf(".");
#endif
			auto count = 0;
			while (!theEnd)
			{
				count++;
				CudaCalculateIter1Begin<<<CUDA_Grid_dim_precalc, 1>>>();
				//mrqcof
				CudaCalculateIter1Mrqcof1Start<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
				for (iC = 1; iC < l_curves; iC++)
				{
					CudaCalculateIter1Mrqcof1Matrix<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(l_points[iC]);
					CudaCalculateIter1Mrqcof1Curve1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
					CudaCalculateIter1Mrqcof1Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof1Curve1Last<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1End<<<CUDA_Grid_dim_precalc, 1>>>();
				//mrqcof
				CudaCalculateIter1Mrqmin1End<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
				//mrqcof
				CudaCalculateIter1Mrqcof2Start<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
				for (iC = 1; iC < l_curves; iC++)
				{
					CudaCalculateIter1Mrqcof2Matrix<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(l_points[iC]);
					CudaCalculateIter1Mrqcof2Curve1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
					CudaCalculateIter1Mrqcof2Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof2Curve1Last<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2End<<<CUDA_Grid_dim_precalc, 1>>>();
				//mrqcof
				CudaCalculateIter1Mrqmin2End<<<CUDA_Grid_dim_precalc, 1>>>();
				CudaCalculateIter2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
				//err=cudaThreadSynchronize(); memcpy is synchro itself
				err = cudaDeviceSynchronize();
				//cudaMemcpy(&theEnd, endPtr, sizeof(theEnd), cudaMemcpyDeviceToHost);
				cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd), 0, cudaMemcpyDeviceToHost);
				theEnd = theEnd == CUDA_Grid_dim_precalc;

				//break;//debug
			}
			CudaCalculateFinishPole<<<CUDA_Grid_dim_precalc, 1>>>();
			err = cudaDeviceSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_Grid_dim_precalc);
						//break; //debug
		}
		printf("\n");

		CudaCalculateFinish<<<CUDA_Grid_dim_precalc, 1>>>();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_Grid_dim_precalc, cudaMemcpyDeviceToHost);

		for (m = 1; m <= CUDA_Grid_dim_precalc; m++)
		{
			if (res[m - 1].isReported == 1)
				sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
		}
	} /* period loop */

	isPrecalc = 0;
	cudaMemcpyToSymbol(CUDA_Is_Precalc, &isPrecalc, sizeof(isPrecalc), 0, cudaMemcpyHostToDevice);

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

	free((void*)res);

	ave_dark_facet = sum_dark_facet / max_test_periods;

	if (ave_dark_facet < 1.0)
		*new_conw = 1; /* new correct conwexity weight */
	if (ave_dark_facet >= 1.0)
		*conw_r = *conw_r * 2; /* still not good */

	return 1;
}

int CUDAStart(int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
	int ndata, int* ia, int* ia_par, double* cg_first, MFILE& mf, double escl, double* sig, int Numfac, double* brightness)
{
	int retval, i, n, m, iC, n_max = (int)((freq_start - freq_end) / freq_step) + 1;
	int n_iter_max, theEnd, LinesWritten;
	double iter_diff_max;
	freq_result* res;
	void* pcc, * pfr, * pbrightness, * psig;
	char buf[256];

	for (i = 1; i <= n_ph_par; i++)
	{
		ia[n_coef + 3 + i] = ia_par[i];
	}

	n_iter_max = 0;
	iter_diff_max = -1;
	if (stop_condition > 1)
	{
		n_iter_max = (int)stop_condition;
		iter_diff_max = 0;
		n_iter_min = 0; /* to not overwrite the n_iter_max value */
	}
	if (stop_condition < 1)
	{
		n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
		iter_diff_max = stop_condition;
	}

	cudaError_t err;

	/*int i_col, sh_icol_local[CUDA_BLOCK_DIM], sh_irow_local[CUDA_BLOCK_DIM];
	double piv_inv, sh_big_local[CUDA_BLOCK_DIM];*/

	//here move data to device
	cudaMemcpyToSymbol(CUDA_Ncoef, &n_coef, sizeof(n_coef));
	cudaMemcpyToSymbol(CUDA_Nphpar, &n_ph_par, sizeof(n_ph_par));
	cudaMemcpyToSymbol(CUDA_Numfac, &Numfac, sizeof(Numfac));
	m = Numfac + 1;
	cudaMemcpyToSymbol(CUDA_Numfac1, &m, sizeof(m));
	cudaMemcpyToSymbol(CUDA_ia, ia, sizeof(int) * (MAX_N_PAR + 1));
	cudaMemcpyToSymbol(CUDA_cg_first, cg_first, sizeof(double) * (MAX_N_PAR + 1));
	cudaMemcpyToSymbol(CUDA_n_iter_max, &n_iter_max, sizeof(n_iter_max));
	cudaMemcpyToSymbol(CUDA_n_iter_min, &n_iter_min, sizeof(n_iter_min));
	cudaMemcpyToSymbol(CUDA_ndata, &ndata, sizeof(ndata));
	cudaMemcpyToSymbol(CUDA_iter_diff_max, &iter_diff_max, sizeof(iter_diff_max));
	cudaMemcpyToSymbol(CUDA_conw_r, &conw_r, sizeof(conw_r));
	cudaMemcpyToSymbol(CUDA_Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
	cudaMemcpyToSymbol(CUDA_Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	cudaMemcpyToSymbol(CUDA_Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	cudaMemcpyToSymbol(CUDA_Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
	cudaMemcpyToSymbol(CUDA_Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
	cudaMemcpyToSymbol(CUDA_Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	/*cudaMemcpyToSymbol(icol, &i_col, sizeof(i_col));
	cudaMemcpyToSymbol(pivinv, &piv_inv, sizeof(piv_inv));
	cudaMemcpyToSymbol(sh_icol, sh_icol_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_irow, sh_irow_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_big, sh_big_local, sizeof(double) * CUDA_BLOCK_DIM);*/


	err = cudaMalloc(&pbrightness, (ndata + 1) * sizeof(double));
	err = cudaMemcpy(pbrightness, brightness, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	err = cudaBindTexture(0, texbrightness, pbrightness, (ndata + 1) * sizeof(double));

	err = cudaMalloc(&psig, (ndata + 1) * sizeof(double));
	err = cudaMemcpy(psig, sig, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	err = cudaBindTexture(0, texsig, psig, (ndata + 1) * sizeof(double));
	if (err) printf("Error: %s", cudaGetErrorString(err));

	/* number of fitted parameters */
	int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
	for (m = 1; m <= ma; m++)
	{
		if (ia[m])
		{
			lmfit++;
			llastma = m;
		}
	}
	llastone = 1;
	for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
	{
		if (!ia[m]) break;
		llastone = m;
	}
	cudaMemcpyToSymbol(CUDA_ma, &ma, sizeof(ma));
	cudaMemcpyToSymbol(CUDA_mfit, &lmfit, sizeof(lmfit));
	m = lmfit + 1;
	cudaMemcpyToSymbol(CUDA_mfit1, &m, sizeof(m));
	cudaMemcpyToSymbol(CUDA_lastma, &llastma, sizeof(llastma));
	cudaMemcpyToSymbol(CUDA_lastone, &llastone, sizeof(llastone));
	m = ma - 2 - n_ph_par;
	cudaMemcpyToSymbol(CUDA_ncoef0, &m, sizeof(m));

	err = cudaMalloc(&pcc, CUDA_grid_dim * sizeof(freq_context));
	cudaMemcpyToSymbol(CUDA_CC, &pcc, sizeof(pcc));
	err = cudaMalloc(&pfr, CUDA_grid_dim * sizeof(freq_result));
	cudaMemcpyToSymbol(CUDA_FR, &pfr, sizeof(pfr));

	m = (Numfac + 1) * (n_coef + 1);
	cudaMemcpyToSymbol(CUDA_Dg_block, &m, sizeof(m));

	double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;

	err = cudaMalloc(&pa, CUDA_grid_dim * (Numfac + 1) * sizeof(double));
	err = cudaBindTexture(0, texArea, pa, CUDA_grid_dim * (Numfac + 1) * sizeof(double));
	err = cudaMalloc(&pg, CUDA_grid_dim * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	err = cudaBindTexture(0, texDg, pg, CUDA_grid_dim * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	err = cudaMalloc(&pal, CUDA_grid_dim * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = cudaMalloc(&pco, CUDA_grid_dim * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = cudaMalloc(&pdytemp, CUDA_grid_dim * (max_l_points + 1) * (ma + 1) * sizeof(double));
	err = cudaMalloc(&pytemp, CUDA_grid_dim * (max_l_points + 1) * sizeof(double));

	for (m = 0; m < CUDA_grid_dim; m++)
	{
		freq_context ps;
		ps.Area = &pa[m * (Numfac + 1)];
		ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
		ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
		ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
		ps.dytemp = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
		ps.ytemp = &pytemp[m * (max_l_points + 1)];
		freq_context* pt = &((freq_context*)pcc)[m];
		err = cudaMemcpy(pt, &ps, sizeof(void*) * 6, cudaMemcpyHostToDevice);
	}

	res = (freq_result*)malloc(CUDA_grid_dim * sizeof(freq_result));

	//int firstreport = 0;//beta debug

	for (n = n_start_from; n <= n_max; n += CUDA_grid_dim)
	{
		auto fractionDone = (double)n / (double)n_max;
		boinc_fraction_done(fractionDone);

#if _DEBUG
		float fraction = fractionDone * 100;
		std::time_t t = std::time(nullptr);   // get time now
		std::tm* now = std::localtime(&t);

		printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
		fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
#endif

		CudaCalculatePrepare<<<CUDA_grid_dim, 1>>>(n, n_max, freq_start, freq_step);
		err = cudaDeviceSynchronize();

		for (m = 1; m <= N_POLES; m++)
		{
			//zero global End signal
			theEnd = 0;
			cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd));
			//
			CudaCalculatePreparePole<<<CUDA_grid_dim, 1>>>(m);
			//
			while (!theEnd)
			{
				CudaCalculateIter1Begin<<<CUDA_grid_dim, 1>>>();
				//mrqcof
				CudaCalculateIter1Mrqcof1Start<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();
				for (iC = 1; iC < l_curves; iC++)
				{
					CudaCalculateIter1Mrqcof1Matrix<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(l_points[iC]);
					CudaCalculateIter1Mrqcof1Curve1<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
					CudaCalculateIter1Mrqcof1Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof1Curve1Last<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1End<<<CUDA_grid_dim, 1>>>();
				//mrqcof
				CudaCalculateIter1Mrqmin1End<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();
				//mrqcof
				CudaCalculateIter1Mrqcof2Start<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();
				for (iC = 1; iC < l_curves; iC++)
				{
					CudaCalculateIter1Mrqcof2Matrix<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(l_points[iC]);
					CudaCalculateIter1Mrqcof2Curve1<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
					CudaCalculateIter1Mrqcof2Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof2Curve1Last<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2End<<<CUDA_grid_dim, 1>>>();
				//mrqcof
				CudaCalculateIter1Mrqmin2End<<<CUDA_grid_dim, 1>>>();
				CudaCalculateIter2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();
				//err=cudaThreadSynchronize(); memcpy is synchro itself
				cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));
				theEnd = theEnd == CUDA_grid_dim;

				//break;//debug
			}
			CudaCalculateFinishPole<<<CUDA_grid_dim, 1>>>();
			err = cudaDeviceSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_grid_dim);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_grid_dim);
						//break; //debug
		}

		CudaCalculateFinish<<<CUDA_grid_dim, 1>>>();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim, cudaMemcpyDeviceToHost);

		LinesWritten = 0;
		for (m = 1; m <= CUDA_grid_dim; m++)
		{
			if (res[m - 1].isReported == 1)
			{
				LinesWritten++;
				/* output file */
				if ((n == 1) && (m == 1))
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), conw_r * escl * escl, res[m - 1].la_best, res[m - 1].be_best);
				else
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), res[m - 1].dark_best, res[m - 1].la_best, res[m - 1].be_best);
			}
		}
		if (boinc_time_to_checkpoint() || boinc_is_standalone()) {
			retval = DoCheckpoint(mf, (n - 1) + LinesWritten, 1, conw_r); //zero lines
			if (retval) { fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval); }
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

	free((void*)res);

	return 1;
}
