#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include "globals.h"
#include "Start_OpenCl.h"
//#include "Globals_OpenCl.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "declarations.hpp"


using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

// NOTE: global to all freq

// TODO: Define "Texture" like Texture<int2> from this:
vector<vector<int, int>> texWeight;

int CUDA_grid_dim;

// NOTE: global to one thread
FreqResult* CUDA_FR;

double* pee, * pee0, * pWeight;

cl_int ClPrepare(int device, double* beta_pole, double* lambda_pole, double* par, double cl, double Alambda_start, double Alambda_incr,
    double ee[][3], double ee0[][3], double* tim, double Phi_0, int checkex, int ndata)
{
    try {
        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator iter;
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

			std::cerr << "Supported extensions by platform:" << endl;
			auto extensions = (*iter).getInfo<CL_PLATFORM_EXTENSIONS>();
			auto lenght = extensions.length();
			for(auto i = 0; i < lenght; i++)
			{
				std::cerr << extensions[i] << "\t";
			}
			std::cerr << endl;

        }

        // Create an OpenCL context
        cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0 };
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

        // Detect OpenCL devices
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        const auto deviceId = 0;
        const cl::Device device = devices[deviceId];
        const auto openClVersion = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
        const auto clDeviceVersion = device.getInfo<CL_DEVICE_VERSION>();
        const auto deviceName = device.getInfo<CL_DEVICE_NAME>();
        const auto deviceMaxWorkItemDims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
        const auto clGlobalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        const auto globalMemory = clGlobalMemory / 1048576;
        const auto msCount = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        const auto block = device.getInfo<CL_DEVICE_MAX_SAMPLERS>();
		//const auto t = device.getInfo<CL_PL>();

        std::cerr << "OpenCL version: " << openClVersion << " | " << clDeviceVersion << endl;
        std::cerr << "OpenCL Device number : " << deviceId << endl;
        std::cerr << "OpenCl Device name: " << deviceName << " " << globalMemory << "MB" << endl;
        std::cerr << "Multiprocessors: " << msCount << endl;
        std::cerr << "Max Samplers: " << block << endl;
        std::cerr << "Max work item dimensions: " << deviceMaxWorkItemDims << endl;

		// TODO: Calculate this:
		auto SMXBlock = 128;
		CUDA_grid_dim = msCount * SMXBlock;

        cl_int* err = nullptr;
        auto clBetaPole = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N_POLES + 1), beta_pole, err);
        auto clLambdaPole = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N_POLES + 1), lambda_pole, err);
        auto clPar = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * 4, par, err);
        auto clCl = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof cl, &cl, err);
        auto clAlambdaStart = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Alambda_start, &Alambda_start, err);
        auto clAlambdaIncr = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Alambda_incr, &Alambda_incr, err);
        auto clMmax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m_max, &m_max, err);
        auto clLmax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof l_max, &l_max, err);
        auto clTim = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_OBS + 1), tim, err);
        auto clPhi0 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Phi_0, &Phi_0);

        queue = cl::CommandQueue(context, devices[0]);

        auto pWsize = (ndata + 3 + 1) * sizeof(double);
        auto pWeightBuf = cl::Buffer(context, CL_MEM_READ_ONLY, pWsize, err);
        double* pWeight; //???
        //queue.enqueueWriteBuffer(pWeightBuf, CL_BLOCKING, 0, pWsize, texWeight);
        // texWeight = &pWeight; ??

        /*res = cudaMalloc(&pWeight, (ndata + 3 + 1) * sizeof(double));
        res = cudaMemcpy(pWeight, weight, (ndata + 3 + 1) * sizeof(double), cudaMemcpyHostToDevice);
        res = cudaBindTexture(0, texWeight, pWeight, (ndata + 3 + 1) * sizeof(double)); */

        auto pEeSize = (ndata + 1) * 3 * sizeof(double);
        //auto pEeBuf = cl::Buffer(context, CL_MEM_READ_ONLY, pEeSize, err);
        //queue.enqueueWriteBuffer(pEeBuf, CL_BLOCKING, 0, pEeSize, ee);
        auto clEe = cl::Buffer(context, CL_MEM_READ_ONLY, pEeSize, ee, err);
        auto clEe0 = cl::Buffer(context, CL_MEM_READ_ONLY, pEeSize, ee, err);

        /*
        res = cudaMalloc(&pee, (ndata + 1) * 3 * sizeof(double));
        res = cudaMemcpy(pee, ee, (ndata + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
        res = cudaMemcpyToSymbol(CUDA_ee, &pee, sizeof(void*));0

        res = cudaMalloc(&pee0, (ndata + 1) * 3 * sizeof(double));
        res = cudaMemcpy(pee0, ee0, (ndata + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
        res = cudaMemcpyToSymbol(CUDA_ee0, &pee0, sizeof(void*));

        if (res == cudaSuccess) return 1; else return 0;*/
        cl_int result = *err;
        return result;
    }
    catch (cl::Error &err)
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
    catch (string &msg)
    {
        cerr << "Exception caught in main(): " << msg << endl;
        //cleanupHost();
        return 1;
    }
}

cl_int ClPrecalc(double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
    int ndata, int* ia, int* ia_par, int* new_conw, double* cg_first, double* sig, int Numfac, double* brightness)
{
	int max_test_periods, iC, theEnd;
	double sum_dark_facet, ave_dark_facet;
	int i, n, m, n_max = (int)((freq_start - freq_end) / freq_step) + 1;
	int n_iter_max;
	double iter_diff_max;
	FreqResult* res;
	void* pcc, * pfr, * pbrightness, * psig;

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

	auto clNCoef = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_coef, &n_coef, err);
	auto clNPhPar = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_ph_par, &n_ph_par, err);
	auto clNumfac = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Numfac, &Numfac, err);
	m = Numfac + 1;
	auto clNumfac1 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);
	auto clIa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * (MAX_N_PAR + 1), ia);
	auto clCgFirst = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	auto clNIterMax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_iter_max, &n_iter_max, err);
	auto clNIterMin = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_iter_min, &n_iter_min, err);
	auto clNdata = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof ndata, &ndata, err);
	auto clIterDiffMax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof iter_diff_max, &iter_diff_max, err);
	auto clConwR = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof conw_r, conw_r, err);
	auto clNor = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * 3, normal, err);
	auto clFc = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), f_c, err);
	auto clFs = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), f_s, err);
	auto clPleg = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1), pleg, err);
	auto clDArea = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1), d_area, err);
	auto clDSphere = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1), d_sphere, err);

	auto brightnesSize = sizeof(double) * (ndata + 1);
	auto clBrightness = cl::Buffer(context, CL_MEM_READ_ONLY, brightnesSize , brightness, err);
	queue.enqueueWriteBuffer(clBrightness, CL_BLOCKING, 0, brightnesSize, brightness);

	//  err = cudaMalloc(&pbrightness, (ndata + 1) * sizeof(double));
	//  err = cudaMemcpy(pbrightness, brightness, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	//  err = cudaBindTexture(0, texbrightness, pbrightness, (ndata + 1) * sizeof(double));

	auto sigSize = sizeof(double) * (ndata + 1);
	auto clSig = cl::Buffer(context, CL_MEM_READ_ONLY, sigSize, sig, err);
	queue.enqueueWriteBuffer(clBrightness, CL_BLOCKING, 0, sigSize, sig);

	//  err = cudaMalloc(&psig, (ndata + 1) * sizeof(double));
	//  err = cudaMemcpy(psig, sig, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	//  err = cudaBindTexture(0, texsig, psig, (ndata + 1) * sizeof(double));
	//

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

	auto clMa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof ma, &ma, err);
	auto clMFit = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof lmfit, &lmfit, err);
	m = lmfit + 1;
	auto clMFit1 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);
	auto clLastMa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof llastma, &llastma, err);
	auto clMLastOne = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof llastone, &llastone, err);
	m = ma - 2 - n_ph_par;
	auto clNCoef0 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);

	//		cudaMemcpyToSymbol(CUDA_ma, &ma, sizeof(ma));
	//		cudaMemcpyToSymbol(CUDA_mfit, &lmfit, sizeof(lmfit));
	//		m = lmfit + 1;
	//		cudaMemcpyToSymbol(CUDA_mfit1, &m, sizeof(m));
	//		cudaMemcpyToSymbol(CUDA_lastma, &llastma, sizeof(llastma));
	//		cudaMemcpyToSymbol(CUDA_lastone, &llastone, sizeof(llastone));
	//		m = ma - 2 - n_ph_par;
	//		cudaMemcpyToSymbol(CUDA_ncoef0, &m, sizeof(m));

	auto CUDA_Grid_dim_precalc = CUDA_grid_dim;
	if (max_test_periods < CUDA_Grid_dim_precalc) CUDA_Grid_dim_precalc = max_test_periods;

	auto gdpcSize = CUDA_Grid_dim_precalc * sizeof(FreqContext);
	auto clPcc = cl::Buffer(context, CL_MEM_READ_ONLY, gdpcSize, &pcc, err);
	queue.enqueueWriteBuffer(clPcc, CL_BLOCKING, 0, gdpcSize, pcc);

	// err = cudaMalloc(&pcc, CUDA_Grid_dim_precalc * sizeof(freq_context));
	//	cudaMemcpyToSymbol(CUDA_CC, &pcc, sizeof(pcc));

	auto frSize = CUDA_Grid_dim_precalc * sizeof(FreqResult);
	auto clFr = cl::Buffer(context, CL_MEM_READ_ONLY, frSize, &pfr, err);
	queue.enqueueWriteBuffer(clFr, CL_BLOCKING, 0, frSize, pfr);

	//err = cudaMalloc(&pfr, CUDA_Grid_dim_precalc * sizeof(freq_result));
	//cudaMemcpyToSymbol(CUDA_FR, &pfr, sizeof(pfr));

	m = (Numfac + 1) * (n_coef + 1);
	auto clDbBlock = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);

	/*m = (Numfac + 1) * (n_coef + 1);
	cudaMemcpyToSymbol(CUDA_Dg_block, &m, sizeof(m));*/

	double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;


	//err = cudaMalloc(&pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	//err = cudaBindTexture(0, texArea, pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	//err = cudaMalloc(&pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	//err = cudaBindTexture(0, texDg, pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	//err = cudaMalloc(&pal, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	//err = cudaMalloc(&pco, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	//err = cudaMalloc(&pdytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double));
	//err = cudaMalloc(&pytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * sizeof(double));

	//for (m = 0; m < CUDA_Grid_dim_precalc; m++)
	//{
	//	FreqContext ps;
	//	ps.Area = &pa[m * (Numfac + 1)];
	//	ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
	//	ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
	//	ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
	//	ps.dytemp = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
	//	ps.ytemp = &pytemp[m * (max_l_points + 1)];
	//	FreqContext* pt = &((FreqContext*)pcc)[m];

	//	err = cudaMemcpy(pt, &ps, sizeof(void*) * 6, cudaMemcpyHostToDevice);
	//}

	//res = (freq_result*)malloc(CUDA_Grid_dim_precalc * sizeof(freq_result));

	//for (n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
	//{
	//	CUDACalculatePrepare << <CUDA_Grid_dim_precalc, 1 >> > (n, max_test_periods, freq_start, freq_step);
	//	err = cudaThreadSynchronize();

	//	for (m = 1; m <= N_POLES; m++)
	//	{
	//		//zero global End signal
	//		theEnd = 0;
	//		cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd));
	//		//
	//		CUDACalculatePreparePole << <CUDA_Grid_dim_precalc, 1 >> > (m);
	//		//
	//		while (!theEnd)
	//		{
	//			CUDACalculateIter1_Begin << <CUDA_Grid_dim_precalc, 1 >> > ();
	//			//mrqcof
	//			CUDACalculateIter1_mrqcof1_start << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > ();
	//			for (iC = 1; iC < l_curves; iC++)
	//			{
	//				CUDACalculateIter1_mrqcof1_matrix << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (l_points[iC]);
	//				CUDACalculateIter1_mrqcof1_curve1 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
	//				CUDACalculateIter1_mrqcof1_curve2 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
	//			}
	//			CUDACalculateIter1_mrqcof1_curve1_last << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
	//			CUDACalculateIter1_mrqcof1_curve2 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
	//			CUDACalculateIter1_mrqcof1_end << <CUDA_Grid_dim_precalc, 1 >> > ();
	//			//mrqcof
	//			CUDACalculateIter1_mrqmin1_end << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > ();
	//			//mrqcof
	//			CUDACalculateIter1_mrqcof2_start << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > ();
	//			for (iC = 1; iC < l_curves; iC++)
	//			{
	//				CUDACalculateIter1_mrqcof2_matrix << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (l_points[iC]);
	//				CUDACalculateIter1_mrqcof2_curve1 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
	//				CUDACalculateIter1_mrqcof2_curve2 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[iC], l_points[iC]);
	//			}
	//			CUDACalculateIter1_mrqcof2_curve1_last << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
	//			CUDACalculateIter1_mrqcof2_curve2 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
	//			CUDACalculateIter1_mrqcof2_end << <CUDA_Grid_dim_precalc, 1 >> > ();
	//			//mrqcof
	//			CUDACalculateIter1_mrqmin2_end << <CUDA_Grid_dim_precalc, 1 >> > ();
	//			CUDACalculateIter2 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> > ();
	//			//err=cudaThreadSynchronize(); memcpy is synchro itself
	//			cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));
	//			theEnd = theEnd == CUDA_Grid_dim_precalc;

	//			//break;//debug
	//		}
	//		CUDACalculateFinishPole << <CUDA_Grid_dim_precalc, 1 >> > ();
	//		err = cudaThreadSynchronize();
	//		//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
	//		//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_Grid_dim_precalc);
	//					//break; //debug
	//	}

	//	CUDACalculateFinish << <CUDA_Grid_dim_precalc, 1 >> > ();
	//	//err=cudaThreadSynchronize(); memcpy is synchro itself

	//	//read results here
	//	err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_Grid_dim_precalc, cudaMemcpyDeviceToHost);

	//	for (m = 1; m <= CUDA_Grid_dim_precalc; m++)
	//	{
	//		if (res[m - 1].isReported == 1)
	//			sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
	//	}
	//} /* period loop */

	//cudaUnbindTexture(texArea);
	//cudaUnbindTexture(texDg);
	//cudaUnbindTexture(texbrightness);
	//cudaUnbindTexture(texsig);
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

	//free((void*)res);

	//ave_dark_facet = sum_dark_facet / max_test_periods;

	//if (ave_dark_facet < 1.0)
	//	*new_conw = 1; /* new correct conwexity weight */
	//if (ave_dark_facet >= 1.0)
	//	*conw_r = *conw_r * 2; /* still not good */

	return 1;
}