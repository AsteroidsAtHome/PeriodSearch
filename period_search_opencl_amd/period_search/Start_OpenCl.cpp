#if defined __GNUC__
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#endif
#include <CL/cl.hpp>
#include "opencl_helper.h"

// https://stackoverflow.com/questions/18056677/opencl-double-precision-different-from-cpu-double-precision

// TODO:
//<kernel>:2589 : 10 : warning : incompatible pointer types initializing '__generic double *' with an expression of type '__global float *'
//double* dytemp = &CUDA_Dytemp[blockIdx.x];
//^ ~~~~~~~~~~~~~~~~~~~~~~~~

//#include <vector>
#include <cmath>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <algorithm>
#include <ctime>
#include <mfile.h>"
#include <boinc_api.h>"

#include "globals.h"
#include "constants.h"
#include "declarations.hpp"
#include "declarations_OpenCl.h"
#include "Start_OpenCl.h"


#ifdef _WIN32
#include "boinc_win.h"
//#include <Shlwapi.h>
#else
#endif

#include "Globals_OpenCl.h"
#include <cstddef>
#include <numeric>

using namespace std;
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

cl::Kernel kernelClCheckEnd;
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

int CUDA_grid_dim;
//int CUDA_grid_dim_precalc;

// NOTE: global to one thread
#ifdef __GNUC__
// TODO: Check compiler version. If  GCC 4.8 or later is used switch to 'alignas(n)'.
#if defined (INTEL)
cl_uint faOptimizedSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)aligned_alloc(4096, faOptimizedSize);
#else
freq_context* Fa __attribute__((aligned(8)));
#endif
#else

#if defined (INTEL)
cl_uint faOptimizedSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)_aligned_malloc(faOptimizedSize, 4096);
#else
alignas(8) freq_context* Fa;
#endif
#endif

double* pee, * pee0, * pWeight;


cl_int ClPrepare(cl_int deviceId, cl_double* beta_pole, cl_double* lambda_pole, cl_double* par, cl_double lcoef, cl_double a_lamda_start, cl_double a_lamda_incr,
	cl_double ee[][3], cl_double ee0[][3], cl_double* tim, cl_double Phi_0, cl_int checkex, cl_int ndata)
{
#ifndef INTEL
	Fa = static_cast<freq_context*>(malloc(sizeof(freq_context)));
#endif

	//try {
	cl::Platform::get(&platforms);
	vector<cl::Platform>::iterator iter;
	#if defined __GNUC__
	cl::string name;
	cl::string vendor;
	#else
	cl::STRING_CLASS name;
	cl::STRING_CLASS vendor;
	#endif

	for (iter = platforms.begin(); iter != platforms.end(); ++iter)
	{
		auto name = (*iter).getInfo<CL_PLATFORM_NAME>();
		vendor = (*iter).getInfo<CL_PLATFORM_VENDOR>();
		std::cerr << "Platform name: " << name << endl;
		std::cerr << "Platform vendor: " << vendor << endl;
#if defined (AMD)
		if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.") ||
			!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Mesa"))
		{
			break;
		}
#elif defined(NVIDIA)
		if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "NVIDIA Corporation"))
		{
			break;
		}
#endif defined(INTEL)
		if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Intel(R) Corporation"))
		{
			break;
		}
	}

	auto platform = (*iter)();
	cl_int errNum;
	cl_uint numDevices;
	//cl_device_id deviceIds = new int[numDevices];
	cl_device_id* deviceIds;
	errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

	if (numDevices < 1)
	{
		cerr << "No GPU device found for platform " << vendor << "(" << name << ")" << endl;
		return (1);
	}
	if (numDevices > 0)
	{
		deviceIds = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id)); // << GNUC? alloca
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, deviceIds, NULL);
	}

	//errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &deviceIds[deviceId], NULL);
	//errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, deviceIds, NULL);
	auto dev1 = deviceIds[deviceId];
	auto device = cl::Device(dev1);
	for (int i = 0; i < numDevices; i++)
	{
		devices.push_back(cl::Device(deviceIds[i]));
	}



	// Create an OpenCL context
	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	cl_context clContext;
	clContext = clCreateContext(properties, numDevices, deviceIds, NULL, NULL, NULL);
	context = cl::Context(clContext);

	//cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0 };
	//context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

	//cl_context_properties cpsAll[3] = { CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0 };
	//auto contextAll = cl::Context(CL_DEVICE_TYPE_ALL, cpsAll);

	//cl_context_properties cpsCpu[3] = {CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0};
	//contextCpu = cl::Context(CL_DEVICE_TYPE_CPU, cpsCpu);


	// Detect OpenCL devices
	//devices = context.getInfo<CL_CONTEXT_DEVICES>();
	//auto devicesAll = contextAll.getInfo<CL_CONTEXT_DEVICES>();
	//auto devicesCpu = contextCpu.getInfo<CL_CONTEXT_DEVICES>();
	deviceId = 0;
	//const auto device = devices[deviceId];
	//const auto dev = devices[deviceId]();
	const auto deviceName = device.getInfo<CL_DEVICE_NAME>();
	//const auto devicePlatform = device.getInfo<CL_DEVICE_PLATFORM>();
	const auto deviceVendor = device.getInfo<CL_DEVICE_VENDOR>();
	const auto driverVersion = device.getInfo<CL_DRIVER_VERSION>();
	const auto openClVersion = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
	const auto clDeviceVersion = device.getInfo<CL_DEVICE_VERSION>();
	const auto clDeviceExtensionCapabilities = device.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>();
	const auto deviceDoubleFpConfig = device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>();
	const auto clDeviceGlobalMemSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	const auto clDeviceLocalMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
	const auto clDeviceMaxConstantArgs = device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>();
	const auto clDeviceMaxConstantBufferSize = device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
	const auto clDeviceMaxParameterSize = device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();
	const auto clDeviceMaxMemAllocSize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
	const auto deviceMaxWorkItemDims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
	const auto clGlobalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	const auto globalMemory = clGlobalMemory / 1048576;
	const auto msCount = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	const auto block = device.getInfo<CL_DEVICE_MAX_SAMPLERS>();
	const auto deviceExtensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
	const auto devMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	const auto devMaxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();


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

	bool isFp64 = deviceExtensions.find("cl_khr_fp64") != std::string::npos;
	bool doesNotSupportsFp64 = !isFp64;
	if (doesNotSupportsFp64)
	{
		fprintf(stderr, "Double precision floating point not supported by OpenCL implementation on current device34. Exiting...\n");
		exit(-1);
	}

	auto SMXBlock = block; // 32;
	//CUDA_grid_dim = msCount * SMXBlock; //  24 * 32
	//CUDA_grid_dim = 2 * 6 * 32; //  24 * 32
	CUDA_grid_dim = 2 * msCount * SMXBlock; // 384 (1050Ti), 1536 (Nvidia GTX1660Ti), 768 (Intel Graphics HD)
	std::cerr << "Resident blocks per multiprocessor: " << SMXBlock << endl;
	std::cerr << "Grid dim (x2): " << CUDA_grid_dim << " = " << msCount * 2 << " * " << SMXBlock << endl;
	std::cerr << "Block dim: " << BLOCK_DIM << endl;

	int err;

	//Global parameters
	memcpy((*Fa).beta_pole, beta_pole, sizeof(cl_double) * (N_POLES + 1));
	memcpy((*Fa).lambda_pole, lambda_pole, sizeof(cl_double) * (N_POLES + 1));
	memcpy((*Fa).par, par, sizeof(cl_double) * 4);
	memcpy((*Fa).ee, ee, (ndata + 1) * 3 * sizeof(cl_double));
	memcpy((*Fa).ee0, ee0, (ndata + 1) * 3 * sizeof(cl_double));
	memcpy((*Fa).tim, tim, sizeof(double) * (MAX_N_OBS + 1));
	memcpy((*Fa).Weight, weight, (ndata + 3 + 1) * sizeof(double));

	(*Fa).cl = lcoef;
	(*Fa).logCl = log(lcoef);
	(*Fa).Alamda_incr = a_lamda_incr;
	(*Fa).Alamda_start = a_lamda_start;
	(*Fa).Mmax = m_max;
	(*Fa).Lmax = l_max;
	(*Fa).Phi_0 = Phi_0;

#pragma region Load kernel files
	string kernelSourceFile = "kernelSource.bin";
	const char* kernelFileName = "kernels.bin";
#if defined (_DEBUG)
#if defined __GNUC__
	// Load CL file, build CL program object, create CL kernel object
	std::ifstream constantsFile("constants.h");
	std::ifstream globalsFile("GlobalsCL.h");
	std::ifstream intrinsicsFile("Intrinsics.cl");
	std::ifstream swapFile("swap.cl");
	std::ifstream blmatrixFile("blmatrix.cl");
	std::ifstream curvFile("curv.cl");
	std::ifstream curv2File("Curv2.cl");
	std::ifstream mrqcofFile("mrqcof.cl");
	std::ifstream startFile("Start.cl");
	std::ifstream brightFile("bright.cl");
	std::ifstream convFile("conv.cl");
	std::ifstream mrqminFile("mrqmin.cl");
	std::ifstream gauserrcFile("gauss_errc.cl");
	std::ifstream testFile("test.cl");
#else
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
    std::ifstream testFile("period_search/test.cl");
#endif
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
	st << testFile.rdbuf();
	//2. Load the files that contains all kernels;
	st << startFile.rdbuf();

	auto kernel_code = st.str();
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
	testFile.close();

	std::ofstream out(kernelSourceFile);
	out << kernel_code;
	out.close();
#endif

	std::ifstream f(kernelFileName);
	bool kernelExist = f.good();

	bool readsource = false;
#if defined (_DEBUG)
	readsource = true;
#endif

	cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.length() + 1));
	if (!kernelExist || readsource)
	{
		//#if defined (NDEBUG)
		//		cerr << "Kernel binary is missing. Exiting..." << endl;
		//		return(3);
		//#endif

		std::ifstream sourcefile(kernelSourceFile);
		string source;
		string str;
		while (std::getline(sourcefile, str))
		{
			source += str;
			source.push_back('\n');
		}
		sourcefile.close();

#pragma endregion
		cl_int* perr = nullptr;

		//cl::Program::Sources sources(1, std::make_pair(source.c_str(), source.length() + 1));
// 		cl::Program binProgram = cl::Program(context, sources, perr);

// 		//#if _DEBUG
// 		//		if (binProgram.build(devices, "-cl-kernel-arg-info") != CL_SUCCESS)
// 		//#else
// 		//		if (binProgram.build(devices) != CL_SUCCESS)
// 		//#endif
// #if defined (AMD)
// 		// binProgram.build(devices, "-D AMD -w -cl-std=CL1.2");
// 		binProgram.build(devices, "-D AMD -w -cl-std=CL1.1");
// #elif defined (NVIDIA)
// 		binProgram.build(devices, "-D NVIDIA -w -cl-std=CL1.2"); // "-w" "-Werror"
// #elif defined (INTEL)
// 		binProgram.build(devices, "-D INTEL -cl-std=CL1.2");
// #endif

// 		//if (binProgram.build(devices, "-w -x clc++") != CL_SUCCESS) // inhibit all warnings
// 		//{
// 		//	std::cout << " Error building: " << binProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << "\n";
// 		//	exit(1);
// 		//}

// #if defined (NDEBUG)
// 		std::remove(kernelSourceFile.c_str());
// #endif

// 		for (cl::Device dev : devices)
// 		{
// 			std::string name = dev.getInfo<CL_DEVICE_NAME>();
// 			std::string buildlog = binProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
// 			cl_build_status buildStatus = binProgram.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
// 			std::cerr << "Binary build log for " << name << ":" << std::endl << buildlog << " (" << buildStatus << ")" << std::endl;
// 		}

// 		vector<size_t> binSizes = binProgram.getInfo<CL_PROGRAM_BINARY_SIZES>();
// 		vector<char*> output = binProgram.getInfo<CL_PROGRAM_BINARIES>();
// 		std::vector<char> binData(std::accumulate(binSizes.begin(), binSizes.end(), 0));
// 		char* binChunk = &binData[0];
// 		vector<char*> binaries;

// 		for (unsigned int i = 0; i < binSizes.size(); ++i) {
// 			binaries.push_back(binChunk);
// 			binChunk += binSizes[i];
// 		}


// 		binProgram.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);
// 		binProgram.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);
// 		std::ofstream binaryfile(kernelFileName, std::ios::binary);
// 		for (unsigned int i = 0; i < binaries.size(); ++i)
// 			binaryfile.write(binaries[i], binSizes[i]);

// 		binaryfile.close();
 	}

	try
	{
		std::ifstream file(kernelFileName, std::ios::binary | std::ios::in | std::ios::ate);

		uint32_t size = file.tellg();
		file.seekg(0, std::ios::beg);
		char* buffer = new char[size];
		file.read(buffer, size);
		file.close();
		cl::Program::Binaries bin{{buffer, size}};

		std::vector<cl_int> binaryStatus;
		err = 0;
		//cl::Program
		// program = cl::Program(context, devices, bin, &binaryStatus, &err);
		program = cl::Program(context, sources, &err);

#if defined (AMD)
		program.build(devices); //, "-g -x cl -cl-std=CL1.2 -Werror"); // "-Werror" "-w" "-cl-std=CL1.2"
#elif defined (NVIDIA)
        program.build(devices); //, "-D NVIDIA -w -cl-std=CL1.2"); // "-Werror" "-w"
#elif defined (INTEL)
		program.build(devices, "-D INTEL -cl-std=CL1.2");
#endif

		queue = cl::CommandQueue(context, devices[0]);
		if (err != CL_SUCCESS) {
			std::cerr << " Error loading" << cl_error_to_str(err) << "\n";
			exit(1);
		}
		for (std::vector<cl_int>::const_iterator bE = binaryStatus.begin(); bE != binaryStatus.end(); bE++) {
			std::cerr << "Bynary status: " << *bE << std::endl;
		}


		//int bres = program.build(devices, " -Werror"); // " -w "
		//int bres = program.build(devices);
		for (cl::Device dev : devices)
		{
			std::string name = dev.getInfo<CL_DEVICE_NAME>();
			std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			cl_build_status buildStatus = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);

#if _DEBUG
			auto kernels = program.getInfo<CL_PROGRAM_NUM_KERNELS>();
			auto kernelNames = program.getInfo<CL_PROGRAM_KERNEL_NAMES>();
			cerr << "Kernels: " << kernels << endl;
			cerr << "Kernel names: " << endl << kernelNames << endl;
			std::string buildOptions = program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(dev);
			std::cerr << "Build options: " << buildOptions << std::endl;
			std::string programSource = program.getInfo<CL_PROGRAM_SOURCE>();
			// std::cerr << "Program source: " << std::endl;
			// 	std::cerr << programSource << std::endl;

#endif
			if (buildlog.length() == 1)
			{
				buildlog.clear();
				buildlog.append("Ok\n");
			}

			std::cerr << "Build log for " << name << ":" << std::endl << buildlog << " (" << buildStatus << ")" << std::endl;

		}
	}

	catch (cl::Error &e)
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
				std::string deviceDriver = dev.getInfo<CL_DRIVER_VERSION>();
				std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				std::string programSource = program.getInfo<CL_PROGRAM_SOURCE>();
				std::string buildOptions = program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(dev);
				cl_build_status buildStatus = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
				std::cerr << "OpenCL error: " << cl_error_to_str(e.err()) << "(" << e.err() << ")" << std::endl;
				std::cerr << buildStatus << std::endl;
				std::cerr << "Device driver: " << deviceDriver << std::endl;
				std::cerr << "Build options: " << buildOptions << std::endl;
				std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
				std::cerr << "Program source: " << std::endl;
				std::cerr << programSource << std::endl;
				// fprintf(stderr, "Build log for %s: %s\n", name.c_str(), buildlog.c_str());
			}
			throw e;
		}

		return 2;
	}

#pragma region Kernel creation
	cl_int kerr;
	try
	{
		kernelClCheckEnd = cl::Kernel(program, "ClCheckEnd", &kerr);
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
		cerr << "Error creating kernel: \"" << cl_error_to_str(e.err()) << "\"(" << e.err() << ") - " << e.what() <<  " | " << cl_error_to_str(kerr) <<
			" (" << kerr << ")" << std::endl;
		cout << "Error while creating kernel. Check stderr.txt for details." << endl;
		return(4);
	}
#pragma endregion

	return 0;
	//}
	//catch (cl::Error& e)
	//{
	//	// Catch OpenCL errors and print log if it is a build error
	//	cerr << "ERROR: " << e.what() << "(" << e.err() << ")" << endl;
	//	cout << "ERROR: " << e.what() << "(" << e.err() << ")" << endl;
	//	if (e.err() == CL_BUILD_PROGRAM_FAILURE)
	//	{
	//		const auto str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
	//		cout << "Program Info: " << str << endl;
	//	}
	//	//cleanupHost();
	//	return 1;
	//}
	//catch (string& msg)
	//{
	//	cerr << "Exception caught in main(): " << msg << endl;
	//	//cleanupHost();
	//	return 1;
	//}
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
	//freq_result* res;
	//auto blockDim = BLOCK_DIM;
	cl_int max_test_periods, iC;
	cl_int theEnd = -100;
	double sum_dark_facet, ave_dark_facet;
	cl_int i, n, m;
	cl_int n_iter_max;
	double iter_diff_max;
	auto n_max = static_cast<int>((freq_start - freq_end) / freq_step) + 1;

	//void* pcc;

	auto r = 0;
	//int merr;

	cl_int isPrecalc = 1;

	//void* pbrightness, * psig;

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

	memcpy((*Fa).ia, ia, sizeof(cl_int) * (MAX_N_PAR + 1));
	memcpy((*Fa).Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
	memcpy((*Fa).Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	memcpy((*Fa).Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	memcpy((*Fa).Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
	memcpy((*Fa).Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
	memcpy((*Fa).Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	memcpy((*Fa).Brightness, brightness, (ndata + 1) * sizeof(double));		// sizeof(double)* (MAX_N_OBS + 1));
	memcpy((*Fa).Sig, sig, (ndata + 1) * sizeof(double));							// sizeof(double)* (MAX_N_OBS + 1));

	/* number of fitted parameters */
	cl_int lmfit = 0;
	cl_int llastma = 0;
	cl_int llastone = 1;
	cl_int ma = n_coef + 5 + n_ph_par;
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

	cl_int CUDA_grid_dim_precalc = CUDA_grid_dim;
	if (max_test_periods < CUDA_grid_dim_precalc)
	{
		CUDA_grid_dim_precalc = max_test_periods;
	}

	cl_int totalWorkItems = CUDA_grid_dim_precalc * BLOCK_DIM;

	m = (Numfac + 1) * (n_coef + 1);
	(*Fa).Dg_block = m;

	//printf("%zu ", offsetof(freq_context, logC));
	//printf("%zu ", offsetof(freq_context, Dg_block));
	//printf("%zu\n", offsetof(freq_context, lastone));

	////__declspec(align(8)) void* pcc = reinterpret_cast<mfreq_context*>(malloc(pccSize));
	//
	//int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
	//auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
	//auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);


	/*cout << "[Host]: alignof(mfreq_context) = " << alignof(mfreq_context) << endl;
	cout << "[Host]: sizeof(pcc) = " << sizeof(pcc) << endl;
	cout << "[Host]: sizeof(mfreq_context) = " << sizeof(mfreq_context) << endl;*/

#if defined (INTEL)
	auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
#else
	auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);
#endif

#if defined __GNUC__
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)aligned_alloc(4096, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif NVIDIA
    int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#else // WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif NVIDIA
    int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#endif

//#if defined(INTEL)
//	cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
//#if definded __GNUC__
//	auto pcc = (mfreq_context*)aligned_alloc(4096, optimizedSize);
//#else
//	auto pcc = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
//#endif
//	auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
//#elif AMD
//	cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
//#if defined __GNUC__
//	auto pcc = (mfreq_context*)aligned_alloc(8, optimizedSize);
//#else
//	auto pcc = (mfreq_context*)_aligned_malloc(optimizedSize, 8);
//#endif
//	auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
//#else
//	int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
//	auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
//
//	/*cout << "[Host]: alignof(mfreq_context) = " << alignof(mfreq_context) << endl;
//	cout << "[Host]: sizeof(pcc) = " << sizeof(pcc) << endl;
//	cout << "[Host]: sizeof(mfreq_context) = " << sizeof(mfreq_context) << endl;*/
//
//	auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
//
//#endif

	//void* pcc = aligned_alloc(CUDA_grid_dim_precalc, sizeof(mfreq_context)); //[CUDA_grid_dim_precalc] ;
	//pcc = malloc(sizeof(pccSize));

	// NOTE: NOTA BENE - In contrast to Cuda, where global memory is zeroed by itself, here we need to initialize the values in each dimension. GV-26.09.2020
	// <<<<<<<<<<<
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

	// >>>>>>>>>>>>>>>>

	//double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;

	//pa = (double*)malloc(CUDA_grid_dim_precalc * (Numfac + 1) * sizeof(double));
	//pg = (double*)malloc(CUDA_grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	//pal = (double*)malloc(CUDA_grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	//pco = (double*)malloc(CUDA_grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	//pdytemp = (double*)malloc(CUDA_grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double));
	//pytemp = (double*)malloc(CUDA_grid_dim_precalc * (max_l_points + 1) * sizeof(double));

	//for (m = 0; m < CUDA_grid_dim_precalc; m++)
	//{
	//	mfreq_context ps;
	//	ps.Area = &pa[m * (Numfac + 1)];
	//	ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
	//	ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
	//	ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
	//	ps.dytemp = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
	//	ps.ytemp = &pytemp[m * (max_l_points + 1)];

	//	std::fill_n(ps.Area, Numfac + 1, 0.0);
	//	std::fill_n(ps.Dg, (Numfac + 1)* (n_coef + 1), 0.0);
	//	std::fill_n(ps.alpha, (lmfit + 1)* (lmfit + 1), 0.0);
	//	std::fill_n(ps.covar, (lmfit + 1)* (lmfit + 1), 0.0);
	//	std::fill_n(ps.dytemp, (max_l_points + 1)* (ma + 1), 0.0);
	//	std::fill_n(ps.ytemp, (max_l_points + 1), 0.0);

	//	mfreq_context* pt = &((mfreq_context*)pcc)[m];
	//	r = memcpy_s(pt, sizeof(void*) * 6, & ps, sizeof(void*) * 6);
	//	//err = cudaMemcpy(pt, &ps, sizeof(void*) * 6, cudaMemcpyHostToDevice);
	//}

	//// <<<<<<<<<<<<<<<<<

	//for (m = 0; m < CUDA_grid_dim_precalc; m++)
	//{
	//	std::fill_n(((mfreq_context*)pcc)[m].Area, Numfac + 1, 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].Dg, (Numfac + 1) * (n_coef + 1), 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].alpha, (lmfit + 1) * (lmfit + 1), 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].covar, (lmfit + 1) * (lmfit + 1), 0.0);
	//	//std::fill_n(((mfreq_context*)pcc)[m].Area, MAX_N_FAC + 1, 0.0);
	//	//std::fill_n(((mfreq_context*)pcc)[m].Dg, (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
	//	//std::fill_n(((mfreq_context*)pcc)[m].alpha, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
	//	//std::fill_n(((mfreq_context*)pcc)[m].covar, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].beta, MAX_N_PAR + 1, 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].da, MAX_N_PAR + 1, 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].atry, MAX_N_PAR + 1, 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].dave, MAX_N_PAR + 1, 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].dytemp, (max_l_points + 1) * (ma + 1), 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].ytemp, (max_l_points + 1), 0.0);
	//	//std::fill_n(((mfreq_context*)pcc)[m].dytemp, (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
	//	//std::fill_n(((mfreq_context*)pcc)[m].ytemp, POINTS_MAX + 1, 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].sh_big, BLOCK_DIM, 0.0);
	//	std::fill_n(((mfreq_context*)pcc)[m].sh_icol, BLOCK_DIM, 0);
	//	std::fill_n(((mfreq_context*)pcc)[m].sh_irow, BLOCK_DIM, 0);
	//	((mfreq_context*)pcc)[m].icol = 0;
	//	((mfreq_context*)pcc)[m].pivinv = 0;
	//}

	auto t = sizeof(struct mfreq_context); // 351160 B / 384 * 351160 = 134,845,440 (128.6MB)   // 4232760 B / 384 * 4232760 = 1,625,379,840 ( 1.6 GB)
	auto tt = t * CUDA_grid_dim_precalc;

	//void* test;
	//int testSize = 10 * sizeof(double);
	//test = malloc(testSize);

	//auto CUDA_TEST = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 10 * sizeof(double), test, err);
	//auto clTest = queue.enqueueMapBuffer(CUDA_TEST, CL_NON_BLOCKING, CL_MAP_WRITE, 0, 10 * sizeof(double), NULL, NULL, err);

#if defined (INTEL)
	queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#elif defined AMD
	queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
	queue.flush();
#elif defined NVIDIA
	queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
#endif

	//auto clPcc = queue.enqueueMapBuffer(CUDA_MCC2, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, pccSize, NULL, NULL, &r);
	//queue.enqueueUnmapMemObject(CUDA_MCC2, clPcc);

	if (r != CL_SUCCESS) {
		cout << " Error creating mapping" << *(int*)err << "\n";
		exit(1);
	}

#if defined (INTEL)
	auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
	int faSize = sizeof(freq_context);
	auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, Fa, err);
	queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);
#endif

	cl_int* end = (int*)malloc(sizeof(int));
	*end = -90;

	//int end;

	//auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
	//auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(int), end, err);
	//auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(int), &theEnd, err);
	//auto clEnd = queue.enqueueMapBuffer(CUDA_End, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(int));

	//auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_int), end, err);
	//auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int), &theEnd, err);
	//auto clEnd = queue.enqueueMapBuffer(CUDA_End, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_int));

#if defined (INTEL)
	auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int), &theEnd, err);
#else
	auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
	queue.enqueueWriteBuffer(CUDA_End, CL_NON_BLOCKING, 0, sizeof(int), &theEnd);
#endif
	//__declspec(align(8)) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	//auto alignas(8) pfr = new freq_result[CUDA_grid_dim_precalc];
	//alignas(8) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	//pfr = static_cast<freq_result*>(malloc(frSize));

	//auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, err);
	//int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
	//void* memIn = (void*)_aligned_malloc(frSize, 256);

	//__declspec(align(8)) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	//auto alignas(8) pfr = new freq_result[CUDA_grid_dim_precalc];
	//alignas(8) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
	//pfr = static_cast<freq_result*>(malloc(frSize));

	//auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, err);
	//void* memIn = (void*)_aligned_malloc(frSize, 256);

#if defined __GNUC__
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context *)aligned_alloc(4096, optimizedSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void *memIn = (void *)aligned_alloc(8, frSize);
#elif NVIDIA
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void *memIn = (void *)aligned_alloc(8, frSize);
#endif // NVIDIA
#else // WIN
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context *)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void *memIn = (void *)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void *pfr;
#elif NVIDIA
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void *memIn = (void *)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void *pfr;
#endif // NViDIA
#endif // WIN

//#if defined (INTEL)
//	cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
//#if defined __GNUC__
//	auto pfr = (mfreq_context*)aligned_alloc(4096, optimizedSize);
//#else
//	auto pfr = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
//#endif
//	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
//#else
//	int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
//#if defined __GNUC__
//	void* memIn = (void*)aligned_alloc(8, frSize);
//#else
//	void* memIn = (void*)_aligned_malloc(frSize, 256);
//#endif
//	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
//	void* pfr;
//#endif
	//pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
	//queue.flush();

#pragma region SetKernelArgs
	kernelClCheckEnd.setArg(0, CUDA_End);
	//kernelClCheckEnd.SetArg(1, sizeof(theEnd), &theEnd);

	kernelCalculatePrepare.setArg(0, CUDA_MCC2);
	kernelCalculatePrepare.setArg(1, CUDA_FR);
	kernelCalculatePrepare.setArg(2, CUDA_End);
	kernelCalculatePrepare.setArg(3, sizeof(freq_start), &freq_start);
	kernelCalculatePrepare.setArg(4, sizeof(freq_step), &freq_step);
	kernelCalculatePrepare.setArg(5, sizeof(max_test_periods), &max_test_periods);

	kernelCalculatePreparePole.setArg(0, CUDA_MCC2);
	kernelCalculatePreparePole.setArg(1, CUDA_CC);
	kernelCalculatePreparePole.setArg(2, CUDA_FR);
	kernelCalculatePreparePole.setArg(3, cgFirst);

	//kernelCalculatePreparePole.setArg(5, sizeof(double), &lcoef);
	// NOTE: 7th arg 'm' is set a little further as 'm' is an iterator counter

	kernelCalculateIter1Begin.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Begin.setArg(1, CUDA_FR);

	kernelCalculateIter1Begin.setArg(3, sizeof(int), &n_iter_min);
	kernelCalculateIter1Begin.setArg(4, sizeof(int), &n_iter_max);
	kernelCalculateIter1Begin.setArg(5, sizeof(double), &iter_diff_max);
	kernelCalculateIter1Begin.setArg(6, sizeof(double), &((*Fa).Alamda_start));
	//kernelCalculateIter1Begin.setArg(6, sizeof(double), &aLambdaStart);

	kernelCalculateIter1Mrqcof1Start.setArg(0, CUDA_MCC2);
	kernelCalculateIter1Mrqcof1Start.setArg(1, CUDA_CC);
	//kernelCalculateIter1Mrqcof1Start.setArg(2, CUDA_FR);
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

	// Allocate result space
	//res = (freq_result*)malloc(CUDA_grid_dim * sizeof(freq_result));
	freq_result* fres; // = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);

	for (n = 1; n <= max_test_periods; n += CUDA_grid_dim_precalc)
	{
#ifndef INTEL
		pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
		queue.flush();
#endif
		for (m = 0; m < CUDA_grid_dim_precalc; m++)
		{
			((freq_result*)pfr)[m].isInvalid = 1;
			((freq_result*)pfr)[m].isReported = 0;
			((freq_result*)pfr)[m].be_best = 0.0;
			((freq_result*)pfr)[m].dark_best = 0.0;
			((freq_result*)pfr)[m].dev_best = 0.0;
			((freq_result*)pfr)[m].freq = 0.0;
			((freq_result*)pfr)[m].la_best = 0.0;
			((freq_result*)pfr)[m].per_best = 0.0;
			((freq_result*)pfr)[m].dev_best_x2 = 0.0;
		}

#if defined (INTEL)
		queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frOptimizedSize, pfr);
#else
		queue.enqueueUnmapMemObject(CUDA_FR, pfr);
		queue.flush();
#endif

		kernelCalculatePrepare.setArg(6, sizeof(n), &n);
		// NOTE: CudaCalculatePrepare(n, max_test_periods, freq_start, freq_step); // << <CUDA_grid_dim_precalc, 1 >> >
		queue.enqueueNDRangeKernel(kernelCalculatePrepare, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));
		queue.enqueueBarrierWithWaitList(); // cuda sync err = cudaThreadSynchronize();

		for (m = 1; m <= N_POLES; m++)
		{
			theEnd = 0; //zero global End signal
			*end = 0;
			//kernelClCheckEnd.setArg(1, sizeof(theEnd), &theEnd);
			//queue.enqueueNDRangeKernel(kernelClCheckEnd, cl::NDRange(), cl::NDRange(1), cl::NDRange(1));
			//queue.enqueueTask(kernelClCheckEnd);
			//*(int*)clEnd = 0;
			//*(int*)clEnd = -5;
//#ifndef INTEL
			queue.enqueueWriteBuffer(CUDA_End, CL_NON_BLOCKING, 0, sizeof(int), &theEnd);   //   <<<<<<<<<<<<<
			//#endif
						//*(int*)clEnd = theEnd;
						//queue.enqueueMapBuffer(CUDA_End, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(int));
						//clEnd = theEnd;  //   <<<<<<<<<<<<<
						//memcpy(clEnd, &theEnd, sizeof(int));
						//*(int*)clEnd = -5;

			kernelCalculatePreparePole.setArg(4, CUDA_End);
			kernelCalculatePreparePole.setArg(5, sizeof(m), &m);
			// NOTE: CudaCalculatePreparePole(m);										<< <CUDA_grid_dim_precalc, 1 >> >
			queue.enqueueNDRangeKernel(kernelCalculatePreparePole, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));

#ifdef _DEBUG
			printf(".");
#endif
			int count = 0;
			while (!theEnd)
				//while (!(theEnd == CUDA_grid_dim_precalc))
				//while (!((*end) == CUDA_grid_dim_precalc))
			{
				count++;
				kernelCalculateIter1Begin.setArg(2, CUDA_End);
				// NOTE: CudaCalculateIter1Begin(); // << <CUDA_grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Begin, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));
				//queue.enqueueMapBuffer(CUDA_End, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_int), NULL, NULL, &r);
				//queue.enqueueMapBuffer(CUDA_End, CL_BLOCKING, CL_MAP_READ, 0, sizeof(cl_int), NULL, NULL, &r);
				//queue.enqueueBarrierWithWaitList(); // TEST

				// NOTE: CudaCalculateIter1Mrqcof1Start(); // << <CUDA_grid_dim_precalc, CUDA_BLOCK_DIM >> >
				// NOTE: Global size is the total number of work items we want to run, and the local size is the size of each workgroup.
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Start, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				//queue.enqueueBarrierWithWaitList(); // TEST
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
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
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
//#ifndef INTEL
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);   //<<<<<<<<<<<<<<<<<<<<
				//#endif
								//queue.enqueueReadBuffer(CUDA_End, CL_NON_BLOCKING, 0, sizeof(int), end);   //<<<<<<<<<<<<<<<<<<<<
								//theEnd = static_cast<int>(reinterpret_cast<intptr_t>(clEnd));
								//theEnd = *(int*)clEnd;
								//theEnd = theEnd == CUDA_grid_dim_precalc;
								//memcpy(&theEnd, end, sizeof(int));

								//queue.enqueueMapBuffer(CUDA_End, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(int));
				theEnd = theEnd == CUDA_grid_dim_precalc;
				//theEnd = *end == CUDA_grid_dim_precalc;
			}

			// NOTE: CudaCalculateFinishPole();	<<<CUDA_grid_dim_precalc, 1 >> >
			queue.enqueueNDRangeKernel(kernelCalculateFinishPole, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));
			queue.enqueueBarrierWithWaitList(); //err = cudaThreadSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_grid_dim_precalc);
		}

		printf("\n");

		// NOTE: CudaCalculateFinish();	<<<CUDA_grid_dim_precalc, 1 >> >
		queue.enqueueNDRangeKernel(kernelCalculateFinish, cl::NDRange(), cl::NDRange(CUDA_grid_dim_precalc), cl::NDRange(1));
		//queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, res);

#if defined (INTEL)
		fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
		queue.finish();
#else
		pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
		queue.flush();
#endif
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		//err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim_precalc, cudaMemcpyDeviceToHost);
#if defined (INTEL)
		auto res = (freq_result*)fres;
#else
		auto res = (freq_result*)pfr;
#endif

		for (m = 1; m <= CUDA_grid_dim_precalc; m++)
		{
			if (res[m - 1].isReported == 1)
			{
				sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
				//printf("[%3d] res[%3d].dark_best: %10.16f, sum_dark_facet: %10.16f\n", m, m-1, res[m-1].dark_best, sum_dark_facet);
			}
		}
#if defined (INTEL)
		queue.enqueueUnmapMemObject(CUDA_FR, fres);
		queue.flush();
#else
		queue.enqueueUnmapMemObject(CUDA_FR, pfr);
		queue.flush();
#endif
	} /* period loop */

#if defined __GNUC__
#if defined INTEL
    free(pcc);
#elif defined AMD
	free(memIn);
	free(pcc);
    delete[] pcc;
#elif defined NVIDIA
    free(memIn);
    free(pcc);
    delete[] pcc;
#endif
#else // WIN
	_aligned_free(pfr);  // res does not need to be freed as it's just a pointer to *pfr.
#if defined (INTEL)
	_aligned_free(pcc);
#elid defined AMD
	delete[] pcc;
#elif defined NVIDIA
    delete[] pcc;
#endif
#endif // WIN

	ave_dark_facet = sum_dark_facet / max_test_periods;

	if (ave_dark_facet < 1.0)
		*new_conw = 1; /* new correct conwexity weight */
	if (ave_dark_facet >= 1.0)
		*conw_r = *conw_r * 2;

	//printf("ave_dark_facet: %10.17f\n", ave_dark_facet);
	//printf("conw_r:         %10.17f\n", *conw_r);

	return 1;
}

int CUDAStart(int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
	int ndata, int* ia, int* ia_par, double* cg_first, MFILE& mf, double escl, double* sig, int Numfac, double* brightness)
{
	//freq_result* res;
	//void* pbrightness, * psig;
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

	//auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	//queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);

	//auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
	//queue.enqueueWriteBuffer(CUDA_End, CL_TRUE, 0, sizeof(int), &theEnd);

	//here move data to device
	memcpy((*Fa).ia, ia, sizeof(int) * (MAX_N_PAR + 1));
	memcpy((*Fa).Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
	memcpy((*Fa).Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	memcpy((*Fa).Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	memcpy((*Fa).Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
	memcpy((*Fa).Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
	memcpy((*Fa).Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	memcpy((*Fa).Brightness, brightness, (ndata + 1) * sizeof(double));		// sizeof(double)* (MAX_N_OBS + 1));
	memcpy((*Fa).Sig, sig, (ndata + 1) * sizeof(double));							// sizeof(double)* (MAX_N_OBS + 1));

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

	auto totalWorkItems = CUDA_grid_dim * BLOCK_DIM; // 768 * 128 = 98304
	m = (Numfac + 1) * (n_coef + 1);
	(*Fa).Dg_block = m;

	//__declspec(align(8)) void* pcc = reinterpret_cast<mfreq_context*>(malloc(pccSize));
	//auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, pccSize, pcc, err);
	//auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pccSize, pcc, err);
	//auto clPcc = queue.enqueueMapBuffer(CUDA_MCC2, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, pccSize);
	//r = memcpy_s(clPcc, pccSize, pcc, pccSize);

	//int pccSize = CUDA_grid_dim * sizeof(mfreq_context);
	//auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
	//auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);

#if defined (INTEL)
	auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
#else
	auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);
#endif

    #if defined __GNUC__
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)aligned_alloc(4096, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif NVIDIA
    int pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#else  // WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif NVIDIA
    int pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#endif


//#if defined (INTEL)
//	cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
//#if defined __GNUC__
//	auto pcc = (mfreq_context*)_aligned_malloc(4096, optimizedSize);
//#else
//	auto pcc = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
//#endif
//	auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, optimizedSize, pcc, err);
//#else
//	int pccSize = CUDA_grid_dim * sizeof(mfreq_context);
//	auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
//
//	/*cout << "[Host]: alignof(mfreq_context) = " << alignof(mfreq_context) << endl;
//	cout << "[Host]: sizeof(pcc) = " << sizeof(pcc) << endl;
//	cout << "[Host]: sizeof(mfreq_context) = " << sizeof(mfreq_context) << endl;*/
//
//	auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
//#endif

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
		//pcc[m].conw_r = 0.0;
		pcc[m].icol = 0;
		pcc[m].pivinv = 0;
	}

#if defined (INTEL)
	queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#else
	queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
#endif
	//queue.enqueueUnmapMemObject(CUDA_MCC2, clPcc);

#if defined (INTEL)
	auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
	int faSize = sizeof(freq_context);
	auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, Fa, err);
	queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);
#endif

#if defined (INTEL)
	auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int), &theEnd, err);
#else
	auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
	queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
#endif

#if defined __GNUC__
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context *)aligned_alloc(4096, optimizedSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    int frSize = CUDA_grid_dim * sizeof(freq_result);
    void *memIn = (void *)aligned_alloc(8, frSize);
#elif NVIDIA
    int frSize = CUDA_grid_dim * sizeof(freq_result);
    void *memIn = (void *)aligned_alloc(8, frSize);
#endif // NVIDIA
#else  // WIN
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context *)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    int frSize = CUDA_grid_dim * sizeof(freq_result);
    void *memIn = (void *)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void *pfr;
#elif NVIDIA
    int frSize = CUDA_grid_dim * sizeof(freq_result);
    void *memIn = (void *)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void *pfr;
#endif // NViDIA
#endif // WIN

//#if defined (INTEL)
//	cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
//#if defined __GNUC__
//	auto pfr = (mfreq_context*)aligned_alloc(4096, frOptimizedSize);
//#else
//	auto pfr = (mfreq_context*)_aligned_malloc(frOptimizedSize, 4096);
//#endif
//	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
//#else
//	int frSize = CUDA_grid_dim * sizeof(freq_result);
//	//__declspec(align(8)) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
//	//auto alignas(8) pfr = new freq_result[CUDA_grid_dim];
//	//alignas(8) void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
//	//pfr = static_cast<freq_result*>(malloc(frSize));
//
//	//auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, err);
//#if defined __GNUC__
//	void* memIn = (void*)aligned_alloc(8, frSize);
//#else
//	void* memIn = (void*)_aligned_malloc(frSize, 256);
//#endif
//	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
//	void* pfr;
//#endif

	//pfr = queue.enqueueMapBuffer(CUDA_FR, CL_NON_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
	//queue.flush();

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

	//try
	//{
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
		kernelCalculatePrepare.setArg(2, CUDA_End);
		kernelCalculatePrepare.setArg(3, sizeof(freq_start), &freq_start);
		kernelCalculatePrepare.setArg(4, sizeof(freq_step), &freq_step);
		kernelCalculatePrepare.setArg(5, sizeof(n_max), &n_max);

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
		//kernelCalculateIter1Mrqcof1Start.setArg(2, CUDA_FR);
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
//	}
	// catch (cl::Error& e)
	// {
	// 	cerr << "Error " << e.err() << " - " << e.what() << std::endl;
	// }

	//int firstreport = 0;//beta debug
	auto oldFractionDone = 0.0001;
	int count = 0;

	freq_result* fres;

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

#ifndef INTEL
		pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
		queue.flush();
#endif
		for (int j = 0; j < CUDA_grid_dim; j++)
		{
			((freq_result*)pfr)[j].isInvalid = 1;
			((freq_result*)pfr)[j].isReported = 0;
			((freq_result*)pfr)[j].be_best = 0.0;
			((freq_result*)pfr)[j].dark_best = 0.0;
			((freq_result*)pfr)[j].dev_best = 0.0;
			((freq_result*)pfr)[j].freq = 0.0;
			((freq_result*)pfr)[j].la_best = 0.0;
			((freq_result*)pfr)[j].per_best = 0.0;
		}

#if defined (INTEL)
		queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frOptimizedSize, pfr);
#else
		queue.enqueueUnmapMemObject(CUDA_FR, pfr);
		queue.flush();
#endif

		kernelCalculatePrepare.setArg(6, sizeof(n), &n); // NOTE: CudaCalculatePrepare << <CUDA_grid_dim, 1 >> > (n, n_max, freq_start, freq_step);
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
			queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(cl_int), &theEnd);	 /// <<<<<<<<<<<<<<<	// CudaCalculatePreparePole << <CUDA_grid_dim, 1 >> > (m);
			queue.enqueueNDRangeKernel(kernelCalculatePreparePole, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));

			count = 0;

			while (!theEnd)
			{
				count++;
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

				kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(3, sizeof(l_points[l_curves]), &(l_points[l_curves]));	//CudaCalculateIter1Mrqcof1Curve1Last << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > (in_rel[l_curves], l_points[l_curves]);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1Last, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				kernelCalculateIter1Mrqcof1Curve2.setArg(2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
				kernelCalculateIter1Mrqcof1Curve2.setArg(3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

				//CudaCalculateIter1Mrqcof1End << <CUDA_grid_dim, 1 >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1End, cl::NDRange(), cl::NDRange(CUDA_grid_dim), cl::NDRange(1));
				//mrqcof

					//CudaCalculateIter1Mrqmin1End << <CUDA_grid_dim, CUDA_BLOCK_DIM >> > ();
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin1End, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));

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
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);    // <<<<<<<<<<<<<<<<<<

				queue.enqueueBarrierWithWaitList(); // err = cudaDeviceSynchronize();

				//err=cudaThreadSynchronize(); memcpy is synchro itself
				//err = cudaDeviceSynchronize();
				//cudaMemcpyFromSymbolAsync(&theEnd, CUDA_End, sizeof theEnd, 0, cudaMemcpyDeviceToHost);
				//cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));
				printf("[%d][%d][%d] END: %d\n", n, m, count, theEnd);

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
		//queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, res);

#if defined (INTEL)
		fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
		queue.finish();
#else
		pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
		queue.flush();
#endif
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		//err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim_precalc, cudaMemcpyDeviceToHost);

		oldFractionDone = fractionDone;
		LinesWritten = 0;
#if defined (INTEL)
		auto res = (freq_result*)fres;
#else
		auto res = (freq_result*)pfr;
#endif
		for (m = 1; m <= CUDA_grid_dim; m++)
		{
			//mf.printf("%4d %3d  %.8f  %.6f  %.6f %4.1f %4.0f %4.0f | %d %d %d\n",
			//	n, m, 24 * res[m].per_best, res[m].dev_best, res[m].dev_best * res[m].dev_best * (ndata - 3), conw_r * escl * escl,
			//	round(res[m].la_best), round(res[m].be_best), res[m].isReported, res[m].isInvalid, res[m].isNiter);

			if (res[m - 1].isReported == 1)
			{
				//LinesWritten++;
				/* output file */
				if (n == 1 && m == 1)
				{
					//mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), conw_r * escl * escl, round(res[m - 1].la_best), round(res[m - 1].be_best));
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best_x2, conw_r * escl * escl, round(res[m - 1].la_best), round(res[m - 1].be_best));
				}
				else
				{
					// period_best, deviation_best, x2
					//mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), res[m - 1].dark_best, round(res[m - 1].la_best), round(res[m - 1].be_best));
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best_x2, res[m - 1].dark_best, round(res[m - 1].la_best), round(res[m - 1].be_best));
				}
			}
			LinesWritten++;
		}

#if defined (INTEL)
		queue.enqueueUnmapMemObject(CUDA_FR, fres);
		queue.flush();
#else
		queue.enqueueUnmapMemObject(CUDA_FR, pfr);
		queue.flush();
#endif

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

	//free((void*)res);
	//free((freq_result*)pfr);
	//delete[] pfr;
	//delete[] pcc;

#if defined __GNUC__
#if defined INTEL
    free(pcc);
#elif defined AMD
    free(memIn);
    free(pcc);
    delete[] pcc;
#elif defined NVIDIA
    free(memIn);
    free(pcc);
    delete[] pcc;
#endif
#else // WIN
    _aligned_free(pfr); // res does not need to be freed as it's just a pointer to *pfr.
#if defined(INTEL)
    _aligned_free(pcc);
#elid defined AMD
    delete[] pcc;
#elif defined NVIDIA
    delete[] pcc;
#endif
#endif // WIN


	return 1;
}
