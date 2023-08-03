#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#pragma OPENCL FP_CONTRACT ON
//#define __CL_ENABLE_EXCEPTIONS  //- redefinition warning - Declared at Preprocessor directives command line
#define FP_64

// _CRT_SECURE_NO_WARNINGS

//#include <CL/cl.h>
#include <CL/cl.hpp>

#ifdef FP_64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
////#else
////#error "Double precision floating point not supported by OpenCL implementation."
#endif

// https://stackoverflow.com/questions/18056677/opencl-double-precision-different-from-cpu-double-precision

// TODO:
//<kernel>:2589 : 10 : warning : incompatible pointer types initializing '__generic double *' with an expression of type '__global float *'
//double* dytemp = &CUDA_Dytemp[blockIdx.x];
//^ ~~~~~~~~~~~~~~~~~~~~~~~~

//#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <algorithm>
#include "mfile.h"
#include "boinc_api.h"

#include "globals.h"
#include "constants.h"
#include "declarations.hpp"
#include "declarations_OpenCl.h"
#include "Start_OpenCl.h"


#ifdef _WIN32
#include "boinc_win.h"
//#include <Shlwapi.h>
#endif

#include "Start_OpenCL.h"
#include "Globals_OpenCl.h"
#include <cstddef>


using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

// NOTE: global to all freq

vector<cl::Platform> platforms;
vector<cl::Device> devices;
cl::Context context;
cl::Context contextCpu;
cl::Program program;
cl::Program programIter1Mrqcof1Start;
cl::CommandQueue queue;
cl::Kernel kernel, kernelDave, kernelSig2wght;
cl::Buffer bufCg, bufArea, bufDarea, bufDg, bufFc, bufFs, bufDsph, bufPleg, bufMmax, bufLmax, bufX, bufY, bufZ;
cl::Buffer bufSig, bufSig2iwght, bufDy, bufWeight, bufYmod;
cl::Buffer bufDave, bufDyda;
cl::Buffer bufD;

int CUDA_grid_dim;
//int CUDA_grid_dim_precalc;

// NOTE: global to one thread
#ifdef __GNUC__
// TODO: Chack compiler version. If  GCC 4.8 or later is used switch to 'alignas(n)'.
//freq_result* CUDA_FR __attribute__((aligned(8)));
freq_context* Fa __attribute__((aligned(8)));
//mfreq_context* Fb __attribute__((aligned(8)));
#else
//__declspec(align(8)) freq_result* CUDA_FR;
//__declspec(align(8)) freq_context* Fa;
//__declspec(align(8)) mfreq_context* Fb;

alignas(8) freq_context* Fa;
//alignas(8) mfreq_context* pcc;
//alignas(8) freq_result* CUDA_FR;
#endif

double* pee, * pee0, * pWeight;


cl_int ClPrepare(cl_int deviceId, cl_double* beta_pole, cl_double* lambda_pole, cl_double* par, cl_double lcoef, cl_double a_lamda_start, cl_double a_lamda_incr,
	cl_double ee[][3], cl_double ee0[][3], cl_double* tim, cl_double Phi_0, cl_int checkex, cl_int ndata)
{
	Fa = static_cast<freq_context*>(malloc(sizeof(freq_context)));

	try {
		cl::Platform::get(&platforms);
		vector<cl::Platform>::iterator iter;
		for (iter = platforms.begin(); iter != platforms.end(); ++iter)
		{
			auto name = (*iter).getInfo<CL_PLATFORM_NAME>();
			auto vendor = (*iter).getInfo<CL_PLATFORM_VENDOR>();
			std::cerr << "Platform vendor: " << vendor << endl;
			if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc."))
			{
				break;
			}
			if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "NVIDIA Corporation"))
			{
				break;
			}
			//if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Intel(R) Corporation"))
			//{
			//	break;
			//}
		}

		// Create an OpenCL context
		cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0 };
		context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
		cl_context_properties cpsAll[3] = { CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0 };
		auto contextAll = cl::Context(CL_DEVICE_TYPE_ALL, cpsAll);

		//cl_context_properties cpsCpu[3] = {CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0};
		//contextCpu = cl::Context(CL_DEVICE_TYPE_CPU, cpsCpu);


		// Detect OpenCL devices
		devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto devicesAll = contextAll.getInfo<CL_CONTEXT_DEVICES>();
		//auto devicesCpu = contextCpu.getInfo<CL_CONTEXT_DEVICES>();
		deviceId = 0;
		const auto device = devices[deviceId];
		const auto openClVersion = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
		const auto clDeviceVersion = device.getInfo<CL_DEVICE_VERSION>();
		const auto clDeviceExtensionCapabilities = device.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>();
		const auto clDeviceGlobalMemSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
		const auto clDeviceLocalMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
		const auto clDeviceMaxConstantArgs = device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>();
		const auto clDeviceMaxConstantBufferSize = device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
		const auto clDeviceMaxParameterSize = device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();
		const auto clDeviceMaxMemAllocSize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
		const auto deviceName = device.getInfo<CL_DEVICE_NAME>();
		const auto deviceMaxWorkItemDims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
		const auto clGlobalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
		const auto globalMemory = clGlobalMemory / 1048576;
		const auto msCount = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		const auto block = device.getInfo<CL_DEVICE_MAX_SAMPLERS>();
		const auto deviceExtensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
		const auto devMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		const auto devMaxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		bool is64CUDA = deviceExtensions.find("cl_khr_fp64") != std::string::npos;
		bool is64AMD = deviceExtensions.find("cl_amd_fp64") == std::string::npos;
		//auto doesNotSupportsFp64 = (deviceExtensions.find("cl_khr_fp64") == std::string::npos) || (deviceExtensions.find("cl_amd_fp64") == std::string::npos);
		//if(doesNotSupportsFp64)
		if (!is64CUDA || !is64AMD)
		{
			fprintf(stderr, "Double precision floating point not supported by OpenCL implementation. Exiting...\n");
			exit(-1);
		}

		std::cerr << "OpenCL version: " << openClVersion << " | " << clDeviceVersion << endl;
		std::cerr << "OpenCL Device number : " << deviceId << endl;
		std::cerr << "OpenCl Device name: " << deviceName << " " << globalMemory << "MB" << endl;
		std::cerr << "Multiprocessors: " << msCount << endl;
		std::cerr << "Max Samplers: " << block << endl;
		std::cerr << "Max work item dimensions: " << deviceMaxWorkItemDims << endl;
#ifdef _DEBUG
		std::cerr << "Debug info:" << endl;
		std::cerr << "CL_DEVICE_EXTENSIONS: " << deviceExtensions << endl;
		std::cerr << "CL_DEVICE_GLOBAL_MEM_SIZE: " << clDeviceGlobalMemSize << " B" << endl;
		std::cerr << "CL_DEVICE_LOCAL_MEM_SIZE: " << clDeviceLocalMemSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_CONSTANT_ARGS: " << clDeviceMaxConstantArgs << endl;
		std::cerr << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << clDeviceMaxConstantBufferSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_PARAMETER_SIZE: " << clDeviceMaxParameterSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << clDeviceMaxMemAllocSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << devMaxWorkGroupSize << endl;
		std::cerr << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << devMaxWorkItemSizes[0] << " | " << devMaxWorkItemSizes[1] << " | " << devMaxWorkItemSizes[2] << endl;
#endif

		auto SMXBlock = block; // 32;
		//CUDA_grid_dim = msCount * SMXBlock; //  24 * 32
		//CUDA_grid_dim = 2 * 6 * 32; //  24 * 32
		CUDA_grid_dim = 2 * msCount * SMXBlock; // 384 (1050Ti), 1536 (Nvidia GTX1660Ti)
		std::cerr << "Resident blocks per multiprocessor: " << SMXBlock << endl;
		std::cerr << "Grid dim (x2): " << CUDA_grid_dim << " = " << msCount * 2 << " * " << SMXBlock << endl;
		std::cerr << "Block dim: " << BLOCK_DIM << endl;

		int err;

		//Global parameters
		err = memcpy_s((*Fa).beta_pole, sizeof((*Fa).beta_pole), beta_pole, sizeof(cl_double) * (N_POLES + 1));
		err = memcpy_s((*Fa).lambda_pole, sizeof((*Fa).lambda_pole), lambda_pole, sizeof(cl_double) * (N_POLES + 1));
		err = memcpy_s((*Fa).par, sizeof((*Fa).par), par, sizeof(cl_double) * 4);
		err = memcpy_s((*Fa).ee, sizeof((*Fa).ee), ee, (ndata + 1) * 3 * sizeof(cl_double));
		err = memcpy_s((*Fa).ee0, sizeof((*Fa).ee0), ee0, (ndata + 1) * 3 * sizeof(cl_double));
		err = memcpy_s((*Fa).tim, sizeof((*Fa).tim), tim, sizeof(double) * (MAX_N_OBS + 1));
		err = memcpy_s((*Fa).Weight, sizeof((*Fa).Weight), weight, (ndata + 3 + 1) * sizeof(double));

		if (err)
		{
			printf("Error executing memcpy_s: r = %d\n", err);
			return err;
		}

		(*Fa).cl = lcoef;
		(*Fa).logCl = log(lcoef);
		(*Fa).Alamda_incr = a_lamda_incr;
		(*Fa).Alamda_start = a_lamda_start;
		(*Fa).Mmax = m_max;
		(*Fa).Lmax = l_max;
		(*Fa).Phi_0 = Phi_0;

		queue = cl::CommandQueue(context, devices[0]);

		return 0;
	}
	catch (cl::Error& err)
	{
		// Catch OpenCL errors and print log if it is a build error
		cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
		cout << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
		if (err.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			const auto str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			cout << "Program Info: " << str << endl;
		}
		//cleanupHost();
		return 1;
	}
	catch (string& msg)
	{
		cerr << "Exception caught in main(): " << msg << endl;
		//cleanupHost();
		return 1;
	}
}

void PrintFreqResult(const int maxItterator, void* pcc, void* pfr)
{
	for (auto l = 0; l < maxItterator; l++)
	{
		//const auto freq = static_cast<freq_context*>(pcc)[l].freq;
		mfreq_context* CC = &((mfreq_context*)pcc)[l];
		//const auto la_best = static_cast<freq_result*>(pfr)[l].la_best;
		freq_result* FR = &((freq_result*)pfr)[l];
		//cerr << "freq[" << l << "] = " << freq << " | la_best[" << l << "] = " << la_best << std::endl;
		cout << "freq[" << l << "] = " << (*CC).freq << " | la_best[" << l << "] = " << (*FR).la_best << std::endl;
	}
}

cl_int ClPrecalc(cl_double freq_start, cl_double freq_end, cl_double freq_step, cl_double stop_condition, cl_int n_iter_min, cl_double* conw_r,
	cl_int ndata, cl_int* ia, cl_int* ia_par, cl_int* new_conw, cl_double* cg_first, cl_double* sig, cl_int Numfac, cl_double* brightness, cl_double lcoef, int Ncoef)
{
	freq_result* res;
	//auto blockDim = BLOCK_DIM;
	int max_test_periods, iC;
	int theEnd = 0;
	double sum_dark_facet, ave_dark_facet;
	int i, n, m;
	int n_iter_max;
	double iter_diff_max;
	auto n_max = static_cast<int>((freq_start - freq_end) / freq_step) + 1;

	auto r = 0;
	int merr;

	int isPrecalc = 1;

	void* pbrightness, * psig;

	max_test_periods = 10;
	sum_dark_facet = 0.0;
	ave_dark_facet = 0.0;

	if (n_max < max_test_periods)
		max_test_periods = n_max;

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

	cl_int* err = nullptr;
	//cudaError_t err;

	(*Fa).conw_r = *conw_r;
	(*Fa).Ncoef = n_coef; //Ncoef;
	(*Fa).Nphpar = n_ph_par;
	(*Fa).Numfac = Numfac;
	m = Numfac + 1;
	(*Fa).Numfac1 = m;
	(*Fa).ndata = ndata;
	(*Fa).Is_Precalc = isPrecalc;

	auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);

	auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
	queue.enqueueWriteBuffer(CUDA_End, CL_TRUE, 0, sizeof(int), &theEnd);

	r = memcpy_s((*Fa).ia, sizeof((*Fa).ia), ia, sizeof(int) * (MAX_N_PAR + 1));
	//r = memcpy_s((*Fa).Weight, sizeof((*Fa).Weight), weight, (ndata + 1) * sizeof(double));				// sizeof(double)* (MAX_N_OBS + 1));
	r = memcpy_s((*Fa).Nor, sizeof((*Fa).Nor), normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
	r = memcpy_s((*Fa).Fc, sizeof((*Fa).Fc), f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	r = memcpy_s((*Fa).Fs, sizeof((*Fa).Fs), f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	r = memcpy_s((*Fa).Pleg, sizeof((*Fa).Pleg), pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
	r = memcpy_s((*Fa).Darea, sizeof((*Fa).Darea), d_area, sizeof(double) * (MAX_N_FAC + 1));
	r = memcpy_s((*Fa).Dsph, sizeof((*Fa).Dsph), d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	r = memcpy_s((*Fa).Brightness, sizeof((*Fa).Brightness), brightness, (ndata + 1) * sizeof(double));		// sizeof(double)* (MAX_N_OBS + 1));
	r = memcpy_s((*Fa).Sig, sizeof((*Fa).Sig), sig, (ndata + 1) * sizeof(double));							// sizeof(double)* (MAX_N_OBS + 1));

	if (r)
	{
		printf("Error executing memcpy_s: r = %d\n", r);
		exit(1);
	}

	/* number of fitted parameters */
	int lmfit = 0;
	int llastma = 0;
	int llastone = 1;
	int ma = n_coef + 5 + n_ph_par;
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

	//(*Fa).Ncoef = n_coef;
	(*Fa).ma = ma;
	(*Fa).Mfit = lmfit;

	m = lmfit + 1;
	(*Fa).Mfit1 = m;

	(*Fa).lastone = llastone;
	(*Fa).lastma = llastma;

	m = ma - 2 - n_ph_par;
	(*Fa).Ncoef0 = m;

	int CUDA_grid_dim_precalc = CUDA_grid_dim;
	if (max_test_periods < CUDA_grid_dim_precalc)
	{
		CUDA_grid_dim_precalc = max_test_periods;
	}

	auto totalWorkItems = CUDA_grid_dim_precalc * BLOCK_DIM;

	m = (Numfac + 1) * (n_coef + 1);
	(*Fa).Dg_block = m;

	//printf("%zu ", offsetof(freq_context, logC));
	//printf("%zu ", offsetof(freq_context, Dg_block));
	//printf("%zu\n", offsetof(freq_context, lastone));

	int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
	//__declspec(align(8)) auto pcc = reinterpret_cast<mfreq_context*>(malloc(pccSize));
	auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
	//pcc = new mfreq_context[CUDA_grid_dim_precalc];

	/*cout << "[Host]: alignof(mfreq_context) = " << alignof(mfreq_context) << endl;
	cout << "[Host]: sizeof(pcc) = " << sizeof(pcc) << endl;
	cout << "[Host]: sizeof(mfreq_context) = " << sizeof(mfreq_context) << endl;*/


	//void* pcc = aligned_alloc(CUDA_grid_dim_precalc, sizeof(mfreq_context)); //[CUDA_grid_dim_precalc] ;
	//pcc = malloc(sizeof(pccSize));

	// NOTE: NOTA BENE - In contrast to Cuda, where global memory is zeroed by itself, here we need to initialize the values in each dimension. GV-26.09.2020
	for (m = 0; m < CUDA_grid_dim_precalc; m++)
	{
		std::fill_n(pcc[m].Area, MAX_N_FAC + 1, 0.0);
		std::fill_n(pcc[m].Dg, (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].alpha, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].covar, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].beta, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].da, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].atry, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].dave, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].dytemp, (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].ytemp, POINTS_MAX + 1, 0.0);
		std::fill_n(pcc[m].sh_big, BLOCK_DIM, 0.0);
		std::fill_n(pcc[m].sh_icol, BLOCK_DIM, 0);
		std::fill_n(pcc[m].sh_irow, BLOCK_DIM, 0);
		//pcc[m].conw_r = 0.0;
		pcc[m].icol = 0;
		pcc[m].pivinv = 0;
	}

	//int dytempSize = CUDA_grid_dim_precalc * (POINTS_MAX + 1) * (MAX_N_PAR + 1) * sizeof(double);
	//int pdytempSize = CUDA_grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double);

	//__declspec(align(8)) double** pdytemp = matrix_double(CUDA_grid_dim_precalc, (max_l_points + 1) * (ma + 1));
	auto alignas(8) pdytemp = new double[CUDA_grid_dim_precalc][(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	int dySize = (POINTS_MAX + 1) * (MAX_N_PAR + 1);
	for (m = 0; m < CUDA_grid_dim_precalc; m++)
	{
		for (int j = 0; j < dySize; j++)
		{
			pdytemp[m][j] = 0.0;
		}
	}

	//auto CUDA_Dytemp = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, CUDA_grid_dim_precalc * dySize * sizeof(double), pdytemp);
	//queue.enqueueWriteBuffer(CUDA_Dytemp, CL_BLOCKING, 0, CUDA_grid_dim_precalc * dySize * sizeof(double), pdytemp);

	auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
	queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);

	int faSize = sizeof(freq_context);
	//__declspec(align(16)) void* pmc = reinterpret_cast<freq_context*>(malloc(pmcSize));
	auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, Fa, err);
	queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);

	// Allocate result space
	//res = (freq_result*)malloc(CUDA_grid_dim_precalc * sizeof(freq_result));

	int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
	//__declspec(align(8)) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	auto alignas(8) pfr = new freq_result[CUDA_grid_dim_precalc];
	//alignas(8) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	//pfr = static_cast<freq_result*>(malloc(frSize));

	for (m = 0; m < CUDA_grid_dim_precalc; m++)
	{
		pfr[m].isInvalid = 0;
		pfr[m].isReported = 0;
		pfr[m].be_best = 0.0;
		pfr[m].dark_best = 0.0;
		pfr[m].dev_best = 0.0;
		pfr[m].freq = 0.0;
		pfr[m].la_best = 0.0;
		pfr[m].per_best = 0.0;
	}

	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, err);
	queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, pfr);

#pragma region Load kernel files
	// Load CL file, build CL program object, create CL kernel object
	std::ifstream constantsFile("period_search/constants.h");
	std::ifstream globalsFile("period_search/GlobalsCL.h");
	std::ifstream intrinsicsFile("period_search/Intrinsics.cl");
	std::ifstream swapFile("period_search/swap.cl");
	std::ifstream blmatrixFile("period_search/blmatrix.cl");
	std::ifstream curvFile("period_search/curv.cl");
	std::ifstream curv2File("period_search/Curv2.cl");
	std::ifstream mrqcofFile("period_search/mrqcof.cl");
	std::ifstream startFile("period_search/Start.cl");
	std::ifstream brightFile("period_search/bright.cl");
	std::ifstream convFile("period_search/conv.cl");
	std::ifstream mrqminFile("period_search/mrqmin.cl");
	std::ifstream gauserrcFile("period_search/gauss_errc.cl");

	// NOTE: The following order is crusial
	std::stringstream st;

	// 1. First load all helper and function Cl files which will be used by the kernels;
	st << constantsFile.rdbuf();
	st << globalsFile.rdbuf();
	st << intrinsicsFile.rdbuf();
	st << swapFile.rdbuf();
	st << blmatrixFile.rdbuf();
	st << curvFile.rdbuf();
	st << curv2File.rdbuf();
	st << brightFile.rdbuf();
	st << convFile.rdbuf();
	st << mrqcofFile.rdbuf();
	st << gauserrcFile.rdbuf();
	st << mrqminFile.rdbuf();

	//2. Load the files that contains all kernels;
	st << startFile.rdbuf();

	auto KernelStart = st.str();
	st.flush();

	constantsFile.close();
	globalsFile.close();
	intrinsicsFile.close();
	startFile.close();
	blmatrixFile.close();
	curvFile.close();
	mrqcofFile.close();
	brightFile.close();
	curv2File.close();
	convFile.close();
	mrqminFile.close();
	gauserrcFile.close();
	swapFile.close();

#pragma endregion

	cl::Program::Sources sources(1, std::make_pair(KernelStart.c_str(), KernelStart.length() + 1));
	program = cl::Program(context, sources, err);

	try
	{
		//int bres = program.build(devices, " -Werror"); // " -w " 
		int bres = program.build(devices);
		for (cl::Device dev : devices)
		{
			std::string name = dev.getInfo<CL_DEVICE_NAME>();
			std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			cl_build_status buildStatus = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
			if (buildlog.length() == 1)
			{
				buildlog.clear();
				buildlog.append("Ok\n");
			}

			std::cerr << "Build log for " << name << ":" << std::endl << buildlog << " (" << buildStatus << ")" << std::endl;
		}
	}
	catch (cl::Error& e)
	{
		if (e.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			for (cl::Device dev : devices)
			{
				// Check the build status
				cl_build_status status1 = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
				//cl_build_status status2 = curv2Program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
				if (status1 != CL_BUILD_ERROR) // && status2 != CL_BUILD_ERROR)
					continue;

				// Get the build log
				std::string name = dev.getInfo<CL_DEVICE_NAME>();
				std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				//fprintf(stderr, "Build log for %s: %s\n", name.c_str(), buildlog.c_str());

				cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
				cout << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
				//buildlog = curv2Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				//std::cerr << buildlog << std::endl;
			}
		}
		else
		{
			for (cl::Device dev : devices)
			{
				std::string name = dev.getInfo<CL_DEVICE_NAME>();
				std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
				fprintf(stderr, "Build log for %s: %s\n", name.c_str(), buildlog.c_str());
			}
			throw e;
		}

		return 2;
	}

#pragma region Kernel creation
	cl_int kerr;
	cl::Kernel kernelCalculatePrepare;
	cl::Kernel kernelCalculatePreparePole;
	cl::Kernel kernelCalculateIter1Begin;
	cl::Kernel kernelCalculateIter1Mrqcof1Start;
	cl::Kernel kernelCalculateIter1Mrqcof1Matrix;
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1;
	cl::Kernel kernelCalculateIter1Mrqcof1Curve2;
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1Last;
	cl::Kernel kernelCalculateIter1Mrqcof1End;
	cl::Kernel kernelCalculateIter1Mrqmin1End;
	cl::Kernel kernelCalculateIter1Mrqcof2Start;
	cl::Kernel kernelCalculateIter1Mrqcof2Matrix;
	cl::Kernel kernelCalculateIter1Mrqcof2Curve1;
	cl::Kernel kernelCalculateIter1Mrqcof2Curve2;
	cl::Kernel kernelCalculateIter1Mrqcof2Curve1Last;
	cl::Kernel kernelCalculateIter1Mrqcof2End;
	cl::Kernel kernelCalculateIter1Mrqmin2End;
	cl::Kernel kernelCalculateIter2;
	cl::Kernel kernelCalculateFinishPole;
	cl::Kernel kernelCalculateFinish;

	try
	{
		kernelCalculatePrepare = cl::Kernel(program, string("ClCalculatePrepare").c_str(), &kerr);
		kernelCalculatePreparePole = cl::Kernel(program, string("ClCalculatePreparePole").c_str(), &kerr);
		kernelCalculateIter1Begin = cl::Kernel(program, string("ClCalculateIter1Begin").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Start = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Start").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Matrix = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Matrix").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Curve1 = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Curve1").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Curve2 = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Curve2").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Curve1Last = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Curve1Last").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1End = cl::Kernel(program, string("ClCalculateIter1Mrqcof1End").c_str(), &kerr);
		kernelCalculateIter1Mrqmin1End = cl::Kernel(program, string("ClCalculateIter1Mrqmin1End").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Start = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Start").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Matrix = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Matrix").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Curve1 = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Curve1").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Curve2 = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Curve2").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Curve1Last = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Curve1Last").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2End = cl::Kernel(program, "ClCalculateIter1Mrqcof2End", &kerr);
		kernelCalculateIter1Mrqmin2End = cl::Kernel(program, "ClCalculateIter1Mrqmin2End", &kerr);
		kernelCalculateIter2 = cl::Kernel(program, "ClCalculateIter2", &kerr);
		kernelCalculateFinishPole = cl::Kernel(program, "ClCalculateFinishPole", &kerr);
		kernelCalculateFinish = cl::Kernel(program, "ClCalculateFinish", &kerr);
	}
	catch (cl::Error& e)
	{
		cerr << "Error " << e.err() << " - " << e.what() << std::endl;
	}
#pragma endregion

#pragma region SetKernelArgs
	kernelCalculatePrepare.setArg(0, CUDA_MCC2);
	kernelCalculatePrepare.setArg(1, CUDA_FR);
	kernelCalculatePrepare.setArg(2, sizeof(freq_start), &freq_start);
	kernelCalculatePrepare.setArg(3, sizeof(freq_step), &freq_step);
	kernelCalculatePrepare.setArg(4, sizeof(max_test_periods), &max_test_periods);

	kernelCalculatePreparePole.setArg(0, CUDA_MCC2);
	kernelCalculatePreparePole.setArg(1, CUDA_CC);
	kernelCalculatePreparePole.setArg(2, CUDA_FR);
	kernelCalculatePreparePole.setArg(3, cgFirst);
	kernelCalculatePreparePole.setArg(4, CUDA_End);
	//kernelCalculatePreparePole.setArg(5, sizeof(double), &lcoef);
	// NOTE: 7th arg 'm' is set a little further as 'm' is an iterator counter

	kernelCalculateIter1Begin.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Begin.setArg(1, CUDA_FR);
	kernelCalculateIter1Begin.setArg(2, CUDA_End);
	kernelCalculateIter1Begin.setArg(3, sizeof(int), &n_iter_min);
	kernelCalculateIter1Begin.setArg(4, sizeof(int), &n_iter_max);
	kernelCalculateIter1Begin.setArg(5, sizeof(double), &iter_diff_max);
	kernelCalculateIter1Begin.setArg(6, sizeof(double), &((*Fa).Alamda_start));
	//kernelCalculateIter1Begin.setArg(6, sizeof(double), &aLambdaStart);

	kernelCalculateIter1Mrqcof1Start.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof1Start.setArg(1, CUDA_CC);
	kernelCalculateIter1Mrqcof1Start.setArg(2, CUDA_FR);
	//kernelCalculateIter1Mrqcof1Start.setArg(3, CUDA_Dytemp);
	//kernelCalculateIter1Mrqcof1Start.setArg(4, CUDA_End);

	kernelCalculateIter1Mrqcof1Matrix.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof1Matrix.setArg(1, CUDA_CC);

	kernelCalculateIter1Mrqcof1Curve1.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof1Curve1.setArg(1, CUDA_CC);
	//kernelCalculateIter1Mrqcof1Curve1.setArg(2, CUDA_Dytemp);

	kernelCalculateIter1Mrqcof1Curve2.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof1Curve2.setArg(1, CUDA_CC);
	//kernelCalculateIter1Mrqcof1Curve2.setArg(2, CUDA_Dytemp);

	kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof1Curve1Last.setArg(1, CUDA_CC);
	//kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, CUDA_Dytemp);

	kernelCalculateIter1Mrqcof1End.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof1End.setArg(1, CUDA_CC);

	kernelCalculateIter1Mrqmin1End.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqmin1End.setArg(1, CUDA_CC);

	kernelCalculateIter1Mrqcof2Start.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof2Start.setArg(1, CUDA_CC);

	kernelCalculateIter1Mrqcof2Matrix.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof2Matrix.setArg(1, CUDA_CC);

	kernelCalculateIter1Mrqcof2Curve1.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof2Curve1.setArg(1, CUDA_CC);
	//kernelCalculateIter1Mrqcof2Curve1.setArg(2, CUDA_Dytemp);

	kernelCalculateIter1Mrqcof2Curve2.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof2Curve2.setArg(1, CUDA_CC);
	//kernelCalculateIter1Mrqcof2Curve2.setArg(2, CUDA_Dytemp);

	kernelCalculateIter1Mrqcof2Curve1Last.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof2Curve1Last.setArg(1, CUDA_CC);
	//kernelCalculateIter1Mrqcof2Curve1Last.setArg(2, CUDA_Dytemp);

	kernelCalculateIter1Mrqcof2End.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof2End.setArg(1, CUDA_CC);

	kernelCalculateIter1Mrqmin2End.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqmin2End.setArg(1, CUDA_CC);

	kernelCalculateIter2.setArg(0, CUDA_MCC2);
	kernelCalculateIter2.setArg(1, CUDA_CC);

	kernelCalculateFinishPole.setArg(0, CUDA_MCC2);
	kernelCalculateFinishPole.setArg(1, CUDA_CC);
	kernelCalculateFinishPole.setArg(2, CUDA_FR);

	kernelCalculateFinish.setArg(0, CUDA_MCC2);
	kernelCalculateFinish.setArg(1, CUDA_FR);
#pragma endregion

	res = (freq_result*)malloc(CUDA_grid_dim * sizeof(freq_result));

	for (n = 1; n <= max_test_periods; n += CUDA_grid_dim_precalc)
	{
		kernelCalculatePrepare.setArg(5, sizeof(n), &n);
		// NOTE: CudaCalculatePrepare(n, max_test_periods, freq_start, freq_step); // << <CUDA_grid_dim_precalc, 1 >> >
		queue.enqueueNDRangeKernel(kernelCalculatePrepare, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));
		queue.enqueueBarrierWithWaitList(); // cuda sync err = cudaThreadSynchronize();

		for (m = 1; m <= N_POLES; m++)
		{
			theEnd = 0; //zero global End signal
			queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
			kernelCalculatePreparePole.setArg(5, sizeof(m), &m);
			// NOTE: CudaCalculatePreparePole(m);										<< <CUDA_grid_dim_precalc, 1 >> >
			queue.enqueueNDRangeKernel(kernelCalculatePreparePole, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));

#ifdef _DEBUG
			printf(".");
#endif
			int count = 0;
			while (!theEnd)
			{
				count++;
				// NOTE: CudaCalculateIter1Begin(); // << <CUDA_grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Begin, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));

				// NOTE: CudaCalculateIter1Mrqcof1Start(); // << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				// NOTE: Global size is the total number of work items we want to run, and the local size is the size of each workgroup.
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Start, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				for (iC = 1; iC < l_curves; iC++)
				{
					kernelCalculateIter1Mrqcof1Matrix.setArg(2, sizeof(l_points[iC]), &(l_points[iC]));
					// NOTE: CudaCalculateIter1Mrqcof1Matrix(l_points[iC]);					//<< <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >>
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Matrix, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof1Curve1.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof1Curve1.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));
					// NOTE: CudaCalculateIter1Mrqcof1Curve1(in_rel[iC], l_points[iC]);		// << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof1Curve2.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof1Curve2.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));
					// NOTE: CudaCalculateIter1Mrqcof1Curve2(in_rel[iC], l_points[iC]);		// << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				}

				//printf("_\n");
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, sizeof in_rel[l_curves], &(in_rel[l_curves]));
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(3, sizeof l_points[l_curves], &(l_points[l_curves]));
				// NOTE: CudaCalculateIter1Mrqcof1Curve1Last(in_rel[l_curves], l_points[l_curves]);	//  << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1Last, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				kernelCalculateIter1Mrqcof1Curve2.setArg(2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
				kernelCalculateIter1Mrqcof1Curve2.setArg(3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
				// NOTE: CudaCalculateIter1Mrqcof1Curve2(in_rel[iC], l_points[iC]);		// << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				// NOTE: CudaCalculateIter1Mrqcof1End();	<< <CUDA_grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1End, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));

				// NOTE: CudaCalculateIter1Mrqmin1End();   << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin1End, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				//queue.enqueueBarrierWithWaitList(); // TEST

				//printf("atry\n");
				// NOTE: CudaCalculateIter1Mrqcof2Start();  	<< <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Start, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				for (iC = 1; iC < l_curves; iC++)
				{
					kernelCalculateIter1Mrqcof2Matrix.setArg(2, sizeof(l_points[iC]), &(l_points[iC]));		// NOTE: CudaCalculateIter1Mrqcof2Matrix(l_points[iC]);	<< <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Matrix, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof2Curve1.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof2Curve1.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));		// NOTE: CudaCalculateIter1Mrqcof2Curve1(in_rel[iC], l_points[iC]);	<< <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve1, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof2Curve2.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof2Curve2.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));		// NOTE: CudaCalculateIter1Mrqcof2Curve2(in_rel[iC], l_points[iC]); << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
					//queue.enqueueBarrierWithWaitList(); // TESTs
				}

				kernelCalculateIter1Mrqcof2Curve1Last.setArg(2, sizeof(in_rel[l_curves]), &in_rel[l_curves]);
				kernelCalculateIter1Mrqcof2Curve1Last.setArg(3, sizeof(l_points[l_curves]), &l_points[l_curves]);		// NOTE: CudaCalculateIter1Mrqcof2Curve1Last(in_rel[l_curves], l_points[l_curves]);	//	 << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve1Last, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				kernelCalculateIter1Mrqcof2Curve2.setArg(2, sizeof(in_rel[l_curves]), &in_rel[l_curves]);
				kernelCalculateIter1Mrqcof2Curve2.setArg(3, sizeof(l_points[l_curves]), &l_points[l_curves]); 			// NOTE: CudaCalculateIter1Mrqcof2Curve2(in_rel[l_curves], l_points[l_curves]);		//	 << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				// NOTE: CudaCalculateIter1Mrqcof2End();	<<<CUDA_grid_dim_precalc, 1 >>>
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2End, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));

				// NOTE: CudaCalculateIter1Mrqmin2End(); <<<CUDA_grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin2End, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));

				// NOTE:CudaCalculateIter2();  <<<CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				queue.enqueueBarrierWithWaitList();  //err=cudaThreadSynchronize(); memcpy is synchro itself

				//cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				theEnd = theEnd == CUDA_grid_dim_precalc;
			}

			// NOTE: CudaCalculateFinishPole();	<<<CUDA_grid_dim_precalc, 1 >> >
			queue.enqueueNDRangeKernel(kernelCalculateFinishPole, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));
			queue.enqueueBarrierWithWaitList(); //err = cudaThreadSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_grid_dim_precalc);
		}

		printf("\n");

		// NOTE: CudaCalculateFinish();	<<<CUDA_grid_dim_precalc, 1 >> >
		queue.enqueueNDRangeKernel(kernelCalculateFinish, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));
		queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, res);
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		//err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim_precalc, cudaMemcpyDeviceToHost);

		for (m = 0; m < CUDA_grid_dim_precalc; m++)
		{
			if (res[m].isReported == 1)
			{
				sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
				//printf("[%3d] res[%3d].dark_best: %10.16f, sum_dark_facet: %10.16f\n", m, m-1, res[m-1].dark_best, sum_dark_facet);
			}
		}


	} /* period loop */

	isPrecalc = 0;
	//queue.enqueueReadBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
	queue.enqueueReadBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);
	(*Fa).Is_Precalc = isPrecalc;
	queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);

	free((void*)res);
	delete[] pfr;
	delete[] pcc;

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
	freq_result* res;
	void* pbrightness, * psig;
	double iter_diff_max;
	int retval, i, n, m, iC;
	int n_iter_max, theEnd, LinesWritten;

	int n_max = (int)((freq_start - freq_end) / freq_step) + 1;

	int isPrecalc = 0;
	auto r = 0;
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

	cl_int* err = nullptr;
	//cudaError_t err;

	(*Fa).conw_r = conw_r;
	(*Fa).Ncoef = n_coef; //Ncoef;
	(*Fa).Nphpar = n_ph_par;
	(*Fa).Numfac = Numfac;
	m = Numfac + 1;
	(*Fa).Numfac1 = m;
	(*Fa).ndata = ndata;
	(*Fa).Is_Precalc = isPrecalc;

	auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);

	auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
	queue.enqueueWriteBuffer(CUDA_End, CL_TRUE, 0, sizeof(int), &theEnd);

	//here move data to device
	r = memcpy_s((*Fa).ia, sizeof((*Fa).ia), ia, sizeof(int) * (MAX_N_PAR + 1));
	//r = memcpy_s((*Fa).Weight, sizeof((*Fa).Weight), weight, (ndata + 1) * sizeof(double));				// sizeof(double)* (MAX_N_OBS + 1));
	r = memcpy_s((*Fa).Nor, sizeof((*Fa).Nor), normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
	r = memcpy_s((*Fa).Fc, sizeof((*Fa).Fc), f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	r = memcpy_s((*Fa).Fs, sizeof((*Fa).Fs), f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	r = memcpy_s((*Fa).Pleg, sizeof((*Fa).Pleg), pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
	r = memcpy_s((*Fa).Darea, sizeof((*Fa).Darea), d_area, sizeof(double) * (MAX_N_FAC + 1));
	r = memcpy_s((*Fa).Dsph, sizeof((*Fa).Dsph), d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	r = memcpy_s((*Fa).Brightness, sizeof((*Fa).Brightness), brightness, (ndata + 1) * sizeof(double));		// sizeof(double)* (MAX_N_OBS + 1));
	r = memcpy_s((*Fa).Sig, sizeof((*Fa).Sig), sig, (ndata + 1) * sizeof(double));							// sizeof(double)* (MAX_N_OBS + 1));

	if (r)
	{
		printf("Error executing memcpy_s: r = %d\n", r);
		return r;
	}

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

	(*Fa).Ncoef = n_coef;
	(*Fa).ma = ma;
	(*Fa).Mfit = lmfit;

	m = lmfit + 1;
	(*Fa).Mfit1 = m;

	(*Fa).lastone = llastone;
	(*Fa).lastma = llastma;

	m = ma - 2 - n_ph_par;
	(*Fa).Ncoef0 = m;


	auto totalWorkItems = CUDA_grid_dim * BLOCK_DIM;
	m = (Numfac + 1) * (n_coef + 1);
	(*Fa).Dg_block = m;

	int pccSize = CUDA_grid_dim * sizeof(mfreq_context);
	//__declspec(align(8)) auto pcc = reinterpret_cast<mfreq_context*>(malloc(pccSize));
	auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];

	for (m = 0; m < CUDA_grid_dim; m++)
	{
		std::fill_n(pcc[m].Area, MAX_N_FAC + 1, 0.0);
		std::fill_n(pcc[m].Dg, (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].alpha, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].covar, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].beta, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].da, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].atry, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].dave, MAX_N_PAR + 1, 0.0);
		std::fill_n(pcc[m].dytemp, (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pcc[m].ytemp, POINTS_MAX + 1, 0.0);
		std::fill_n(pcc[m].sh_big, BLOCK_DIM, 0.0);
		std::fill_n(pcc[m].sh_icol, BLOCK_DIM, 0);
		std::fill_n(pcc[m].sh_irow, BLOCK_DIM, 0);
		pcc[m].icol = 0;
		pcc[m].pivinv = 0;
	}

	auto alignas(8) pdytemp = new double[CUDA_grid_dim][(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	int dySize = (POINTS_MAX + 1) * (MAX_N_PAR + 1);

	for (m = 0; m < CUDA_grid_dim; m++)
	{
		for (int j = 0; j < dySize; j++)
		{
			pdytemp[m][j] = 0.0;
		}
	}

	//auto alignas(8) pdytemp = new double[CUDA_grid_dim_precalc][(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	//int dySize = (POINTS_MAX + 1) * (MAX_N_PAR + 1);

	//auto CUDA_Dytemp = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, CUDA_grid_dim * dySize * sizeof(double), pdytemp);
	//queue.enqueueWriteBuffer(CUDA_Dytemp, CL_BLOCKING, 0, CUDA_grid_dim * dySize * sizeof(double), pdytemp);

	auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
	queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);

	int faSize = sizeof(freq_context);
	//__declspec(align(16)) void* pmc = reinterpret_cast<freq_context*>(malloc(pmcSize));
	auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, Fa, err);
	queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);

	// Allocate result space
	res = (freq_result*)malloc(CUDA_grid_dim * sizeof(freq_result));

	int frSize = CUDA_grid_dim * sizeof(freq_result);
	//__declspec(align(8)) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	auto alignas(8) pfr = new freq_result[CUDA_grid_dim];
	//alignas(8) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	//pfr = static_cast<freq_result*>(malloc(frSize));

	for (m = 0; m < CUDA_grid_dim; m++)
	{
		pfr[m].isInvalid = 0;
		pfr[m].isReported = 0;
		pfr[m].be_best = 0.0;
		pfr[m].dark_best = 0.0;
		pfr[m].dev_best = 0.0;
		pfr[m].freq = 0.0;
		pfr[m].la_best = 0.0;
		pfr[m].per_best = 0.0;
	}

	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, err);
	queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, pfr);

#pragma region Kernels
	cl_int kerr = 0;
	cl::Kernel kernelCalculatePrepare;
	cl::Kernel kernelCalculatePreparePole;
	cl::Kernel kernelCalculateIter1Begin;
	cl::Kernel kernelCalculateIter1Mrqcof1Start;
	cl::Kernel kernelCalculateIter1Mrqcof1Matrix;
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1;
	cl::Kernel kernelCalculateIter1Mrqcof1Curve2;
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1Last;
	cl::Kernel kernelCalculateIter1Mrqcof1End;
	cl::Kernel kernelCalculateIter1Mrqmin1End;
	cl::Kernel kernelCalculateIter1Mrqcof2Start;
	cl::Kernel kernelCalculateIter1Mrqcof2Matrix;
	cl::Kernel kernelCalculateIter1Mrqcof2Curve1;
	cl::Kernel kernelCalculateIter1Mrqcof2Curve2;
	cl::Kernel kernelCalculateIter1Mrqcof2Curve1Last;
	cl::Kernel kernelCalculateIter1Mrqcof2End;
	cl::Kernel kernelCalculateIter1Mrqmin2End;
	cl::Kernel kernelCalculateIter2;
	cl::Kernel kernelCalculateFinishPole;
	cl::Kernel kernelCalculateFinish;

	try
	{
		kernelCalculatePrepare = cl::Kernel(program, string("ClCalculatePrepare").c_str(), &kerr);
		kernelCalculatePreparePole = cl::Kernel(program, string("ClCalculatePreparePole").c_str(), &kerr);
		kernelCalculateIter1Begin = cl::Kernel(program, string("ClCalculateIter1Begin").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Start = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Start").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Matrix = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Matrix").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Curve1 = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Curve1").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Curve2 = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Curve2").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1Curve1Last = cl::Kernel(program, string("ClCalculateIter1Mrqcof1Curve1Last").c_str(), &kerr);
		kernelCalculateIter1Mrqcof1End = cl::Kernel(program, string("ClCalculateIter1Mrqcof1End").c_str(), &kerr);

		kernelCalculateIter1Mrqmin1End = cl::Kernel(program, string("ClCalculateIter1Mrqmin1End").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Start = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Start").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Matrix = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Matrix").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Curve1 = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Curve1").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Curve2 = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Curve2").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2Curve1Last = cl::Kernel(program, string("ClCalculateIter1Mrqcof2Curve1Last").c_str(), &kerr);
		kernelCalculateIter1Mrqcof2End = cl::Kernel(program, "ClCalculateIter1Mrqcof2End", &kerr);
		kernelCalculateIter1Mrqmin2End = cl::Kernel(program, "ClCalculateIter1Mrqmin2End", &kerr);
		kernelCalculateIter2 = cl::Kernel(program, "ClCalculateIter2", &kerr);
		kernelCalculateFinishPole = cl::Kernel(program, "ClCalculateFinishPole", &kerr);
		kernelCalculateFinish = cl::Kernel(program, "ClCalculateFinish", &kerr);

#pragma endregion

#pragma region SetKernelArguments
		kernelCalculatePrepare.setArg(0, CUDA_MCC2);
		kernelCalculatePrepare.setArg(1, CUDA_FR);
		kernelCalculatePrepare.setArg(2, sizeof(freq_start), &freq_start);
		kernelCalculatePrepare.setArg(3, sizeof(freq_step), &freq_step);
		kernelCalculatePrepare.setArg(4, sizeof(n_max), &n_max);

		kernelCalculatePreparePole.setArg(0, CUDA_MCC2);
		kernelCalculatePreparePole.setArg(1, CUDA_CC);
		kernelCalculatePreparePole.setArg(2, CUDA_FR);
		kernelCalculatePreparePole.setArg(3, cgFirst);
		kernelCalculatePreparePole.setArg(4, CUDA_End);
		//kernelCalculatePreparePole.setArg(5, sizeof(double), &lcoef);
		// NOTE: 7th arg 'm' is set a little further as 'm' is an iterator counter

		kernelCalculateIter1Begin.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Begin.setArg(1, CUDA_FR);
		kernelCalculateIter1Begin.setArg(2, CUDA_End);
		kernelCalculateIter1Begin.setArg(3, sizeof(int), &n_iter_min);
		kernelCalculateIter1Begin.setArg(4, sizeof(int), &n_iter_max);
		kernelCalculateIter1Begin.setArg(5, sizeof(double), &iter_diff_max);
		kernelCalculateIter1Begin.setArg(6, sizeof(double), &((*Fa).Alamda_start));

		kernelCalculateIter1Mrqcof1Start.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof1Start.setArg(1, CUDA_CC);
		kernelCalculateIter1Mrqcof1Start.setArg(2, CUDA_FR);
		//kernelCalculateIter1Mrqcof1Start.setArg(3, CUDA_Dytemp);
		//kernelCalculateIter1Mrqcof1Start.setArg(4, CUDA_End);

		kernelCalculateIter1Mrqcof1Matrix.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof1Matrix.setArg(1, CUDA_CC);

		kernelCalculateIter1Mrqcof1Curve1.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof1Curve1.setArg(1, CUDA_CC);
		//kernelCalculateIter1Mrqcof1Curve1.setArg(2, CUDA_Dytemp);

		kernelCalculateIter1Mrqcof1Curve2.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof1Curve2.setArg(1, CUDA_CC);
		//kernelCalculateIter1Mrqcof1Curve2.setArg(2, CUDA_Dytemp);

		kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof1Curve1Last.setArg(1, CUDA_CC);
		//kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, CUDA_Dytemp);

		kernelCalculateIter1Mrqcof1End.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof1End.setArg(1, CUDA_CC);

		kernelCalculateIter1Mrqmin1End.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqmin1End.setArg(1, CUDA_CC);

		kernelCalculateIter1Mrqcof2Start.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof2Start.setArg(1, CUDA_CC);

		kernelCalculateIter1Mrqcof2Matrix.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof2Matrix.setArg(1, CUDA_CC);

		kernelCalculateIter1Mrqcof2Curve1.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof2Curve1.setArg(1, CUDA_CC);
		//kernelCalculateIter1Mrqcof2Curve1.setArg(2, CUDA_Dytemp);

		kernelCalculateIter1Mrqcof2Curve2.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof2Curve2.setArg(1, CUDA_CC);
		//kernelCalculateIter1Mrqcof2Curve2.setArg(2, CUDA_Dytemp);

		kernelCalculateIter1Mrqcof2Curve1Last.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof2Curve1Last.setArg(1, CUDA_CC);
		//kernelCalculateIter1Mrqcof2Curve1Last.setArg(2, CUDA_Dytemp);

		kernelCalculateIter1Mrqcof2End.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqcof2End.setArg(1, CUDA_CC);

		kernelCalculateIter1Mrqmin2End.setArg(0, CUDA_MCC2);
		kernelCalculateIter1Mrqmin2End.setArg(1, CUDA_CC);

		kernelCalculateIter2.setArg(0, CUDA_MCC2);
		kernelCalculateIter2.setArg(1, CUDA_CC);

		kernelCalculateFinishPole.setArg(0, CUDA_MCC2);
		kernelCalculateFinishPole.setArg(1, CUDA_CC);
		kernelCalculateFinishPole.setArg(2, CUDA_FR);

		kernelCalculateFinish.setArg(0, CUDA_MCC2);
		kernelCalculateFinish.setArg(1, CUDA_FR);
#pragma endregion
	}
	catch (cl::Error& e)
	{
		cerr << "Error " << e.err() << " - " << e.what() << std::endl;
	}

	//int firstreport = 0;//beta debug
	auto oldFractionDone = 0.0001;

	for (n = n_start_from; n <= n_max; n += CUDA_grid_dim)
	{
		auto fractionDone = (double)n / (double)n_max;
		//boinc_fraction_done(fractionDone);

		//#if _DEBUG
		//		float fraction = fractionDone * 100;
		//		std::time_t t = std::time(nullptr);   // get time now
		//		std::tm* now = std::localtime(&t);
		//
		//		printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
		//		fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
		//#endif

		kernelCalculatePrepare.setArg(5, sizeof(n), &n); // NOTE: CudaCalculatePrepare << <CUDA_grid_dim, 1 >> > (n, n_max, freq_start, freq_step);
		queue.enqueueNDRangeKernel(kernelCalculatePrepare, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));

		for (m = 1; m <= N_POLES; m++)
		{
			auto mid = (double(fractionDone) - double(oldFractionDone));
			auto inner = (double(mid) / double(N_POLES) * (double(m)));
			//printf("mid: %.4f, inner: %.4f\n", mid, inner);
			auto fractionDone2 = oldFractionDone + inner;
			boinc_fraction_done(fractionDone2);

#ifdef _DEBUG
			float fraction2 = fractionDone2 * 100;
			//float fraction = fractionDone * 100;
			std::time_t t = std::time(nullptr);   // get time now
			std::tm* now = std::localtime(&t);

			printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
			fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
#endif

			theEnd = 0;  //zero global End signal
			//cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd));
			kernelCalculatePreparePole.setArg(5, sizeof(m), &m);
			queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);		// CudaCalculatePreparePole << <CUDA_grid_dim, 1 >> > (m);
			queue.enqueueNDRangeKernel(kernelCalculatePreparePole, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));

			while (!theEnd)
			{
				// CudaCalculateIter1Begin << <CUDA_grid_dim, 1 >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Begin, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));

				//mrqcof
					//CudaCalculateIter1Mrqcof1Start << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Start, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				for (iC = 1; iC < l_curves; iC++)
				{
					kernelCalculateIter1Mrqcof1Matrix.setArg(2, sizeof(l_points[iC]), &(l_points[iC]));	//CudaCalculateIter1Mrqcof1Matrix << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Matrix, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof1Curve1.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof1Curve1.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));	//CudaCalculateIter1Mrqcof1Curve1 << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof1Curve2.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof1Curve2.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));	//CudaCalculateIter1Mrqcof1Curve2 << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				}

				kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, sizeof in_rel[l_curves], &(in_rel[l_curves]));
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(3, sizeof l_points[l_curves], &(l_points[l_curves]));	//CudaCalculateIter1Mrqcof1Curve1Last << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1Last, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				kernelCalculateIter1Mrqcof1Curve2.setArg(2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
				kernelCalculateIter1Mrqcof1Curve2.setArg(3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				//CudaCalculateIter1Mrqcof1End << <CUDA_grid_dim, 1 >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1End, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));
				//mrqcof

					//CudaCalculateIter1Mrqmin1End << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin1End, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				/*if (!if_freq_measured && nvml_enabled && n == n_start_from && m == N_POLES)
					{
						GetPeakClock(cudadev);
					}*/

					//mrqcof

					//CudaCalculateIter1Mrqcof2Start << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Start, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				//clFinish(queue());

				for (iC = 1; iC < l_curves; iC++)
				{
					kernelCalculateIter1Mrqcof2Matrix.setArg(2, sizeof(l_points[iC]), &(l_points[iC]));		//CudaCalculateIter1Mrqcof2Matrix << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Matrix, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof2Curve1.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof2Curve1.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));		//CudaCalculateIter1Mrqcof2Curve1 << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve1, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

					kernelCalculateIter1Mrqcof2Curve2.setArg(2, sizeof(in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof2Curve2.setArg(3, sizeof(l_points[iC]), &(l_points[iC]));		//CudaCalculateIter1Mrqcof2Curve2 << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				}

				kernelCalculateIter1Mrqcof2Curve1Last.setArg(2, sizeof(in_rel[l_curves]), &in_rel[l_curves]);
				kernelCalculateIter1Mrqcof2Curve1Last.setArg(3, sizeof(l_points[l_curves]), &l_points[l_curves]); //CudaCalculateIter1Mrqcof2Curve1Last << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve1Last, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));


				kernelCalculateIter1Mrqcof2Curve2.setArg(2, sizeof(in_rel[l_curves]), &in_rel[l_curves]);
				kernelCalculateIter1Mrqcof2Curve2.setArg(3, sizeof(l_points[l_curves]), &l_points[l_curves]);		//CudaCalculateIter1Mrqcof2Curve2 << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				//CudaCalculateIter1Mrqcof2End << <CUDA_grid_dim, 1 >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2End, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));
				//mrqcof

					//CudaCalculateIter1Mrqmin2End << <CUDA_grid_dim, 1 >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin2End, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));

				//CudaCalculateIter2 << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				queue.enqueueBarrierWithWaitList(); // err = cudaDeviceSynchronize();

				//err=cudaThreadSynchronize(); memcpy is synchro itself
				//err = cudaDeviceSynchronize();
				//cudaMemcpyFromSymbolAsync(&theEnd, CUDA_End, sizeof theEnd, 0, cudaMemcpyDeviceToHost);
				//cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));

				theEnd = theEnd == CUDA_grid_dim;

				//break;//debug
			}

			printf("."); fflush(stdout);
			//CudaCalculateFinishPole << <CUDA_grid_dim, 1 >> > ();
			queue.enqueueNDRangeKernel(kernelCalculateFinishPole, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));

			//err = cudaThreadSynchronize();
			//err = cudaDeviceSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_grid_dim);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_grid_dim);
			//break; //debug
		}

		//CudaCalculateFinish << <CUDA_grid_dim, 1 >> > ();
		queue.enqueueNDRangeKernel(kernelCalculateFinish, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));
		////err=cudaThreadSynchronize(); memcpy is synchro itself
		// 
		//read results here synchronously
		//err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim, cudaMemcpyDeviceToHost);
		queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, res);

		oldFractionDone = fractionDone;
		LinesWritten = 0;
		for (m = 0; m < CUDA_grid_dim; m++)
		{
			mf.printf("%4d %3d  %.8f  %.6f  %.6f %4.1f %4.0f %4.0f | %d %d %d\n", 
				n, m, 24 * res[m].per_best, res[m].dev_best, res[m].dev_best * res[m].dev_best * (ndata - 3), conw_r * escl * escl, 
				round(res[m].la_best), round(res[m].be_best), res[m].isReported, res[m].isInvalid, res[m].isNiter);

			//if (res[m - 1].isReported == 1)
			//{
			//	//LinesWritten++;
			//	/* output file */
			//	if (n == 1 && m == 1)
			//	{
			//		//mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), conw_r * escl * escl, round(res[m - 1].la_best), round(res[m - 1].be_best));
			//		//mf.printf("%4d %3d %.8f  %.6f  %.6f %4.1f %4.0f %4.0f | %d %d\n", n, m, 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), conw_r * escl * escl, round(res[m - 1].la_best), round(res[m - 1].be_best), res[m - 1].isReported, res[m - 1].isInvalid);
			//	}
			//	else
			//	{
			//		//mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), res[m - 1].dark_best, round(res[m - 1].la_best), round(res[m - 1].be_best));
			//		//mf.printf("%4d %3d %.8f  %.6f  %.6f %4.1f %4.0f %4.0f | %d %d\n", n, m, res[m - 1].isInvalid, res[m - 1].isReported, 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), res[m - 1].dark_best, round(res[m - 1].la_best), round(res[m - 1].be_best), res[m - 1].isReported, res[m - 1].isInvalid);
			//	}
			//}
			LinesWritten++;
		}

		if (boinc_time_to_checkpoint() || boinc_is_standalone())
		{
			retval = DoCheckpoint(mf, (n - 1) + LinesWritten, 1, conw_r); //zero lines
			if (retval) { fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval); }
			boinc_checkpoint_completed();
		}
		//		break;//debug

		printf("\n");
		fflush(stdout);
	} /* period loop */

	printf("\n");

	//cudaFree(pa);
	//cudaFree(pg);
	//cudaFree(pal);
	//cudaFree(pco);
	//cudaFree(pdytemp);
	//cudaFree(pytemp);
	//cudaFree(pcc);
	//cudaFree(pfr);
	//cudaFree(pbrightness);
	//cudaFree(psig);

	free((freq_context*)Fa);
	free((void*)res);
	//free((freq_result*)pfr);
	delete[] pfr;
	delete[] pcc;

	return 1;
}

