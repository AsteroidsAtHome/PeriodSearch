#if !defined INTEL

#if !defined _WIN32
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_CL_1_1_DEFAULT_BUILD
// #define CL_API_SUFFIX__VERSION_1_0 CL_API_SUFFIX_COMMON
#define CL_BLOCKING 	CL_TRUE
#else // WIN32
#define CL_TARGET_OPENCL_VERSION 120
// #define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS
typedef unsigned int uint;
#endif



// #include <CL/opencl.hpp>
#include <CL/cl.h>
#include "opencl_helper.h"

// https://stackoverflow.com/questions/18056677/opencl-double-precision-different-from-cpu-double-precision

// TODO:
// <kernel>:2589 : 10 : warning : incompatible pointer types initializing '__generic double *' with an expression of type '__global float *'
// double* dytemp = &CUDA_Dytemp[blockIdx.x];
// ~~~~~~~~~~~~~~~~~~~~~~~~

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
#include "boinc_api.h"
#include "mfile.h"

#include "globals.h"
#include "constants.h"
#include "declarations.hpp"
#include "Start_OpenCl.h"
#include "kernels.cpp"


#ifdef _WIN32
#include "boinc_win.h"
//#include <Shlwapi.h>
#else
#endif

#include "Globals_OpenCl.h"
#include <cstddef>
#include <numeric>

//using namespace std;
using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

// NOTE: global to all freq

// cl_platform_id *platforms;
cl_device_id* devices;
cl_context context;
cl_context contextCpu;
cl_program binProgram, program;
cl_command_queue queue;
cl_kernel kernel, kernelDave, kernelSig2wght;
cl_mem bufCg, bufArea, bufDarea, bufDg, bufFc, bufFs, bufDsph, bufPleg, bufMmax, bufLmax, bufX, bufY, bufZ;
cl_mem bufSig, bufSig2iwght, bufDy, bufWeight, bufYmod;
cl_mem bufDave, bufDyda;
cl_mem bufD;

cl_kernel kernelClCheckEnd;
cl_kernel kernelCalculatePrepare;
cl_kernel kernelCalculatePreparePole;
cl_kernel kernelCalculateIter1Begin;
cl_kernel kernelCalculateIter1Mrqcof1Start;
cl_kernel kernelCalculateIter1Mrqcof1Matrix;
cl_kernel kernelCalculateIter1Mrqcof1Curve1;
cl_kernel kernelCalculateIter1Mrqcof1Curve2;
cl_kernel kernelCalculateIter1Mrqcof1Curve1Last;
cl_kernel kernelCalculateIter1Mrqcof1End;
cl_kernel kernelCalculateIter1Mrqmin1End;
cl_kernel kernelCalculateIter1Mrqcof2Start;
cl_kernel kernelCalculateIter1Mrqcof2Matrix;
cl_kernel kernelCalculateIter1Mrqcof2Curve1;
cl_kernel kernelCalculateIter1Mrqcof2Curve2;
cl_kernel kernelCalculateIter1Mrqcof2Curve1Last;
cl_kernel kernelCalculateIter1Mrqcof2End;
cl_kernel kernelCalculateIter1Mrqmin2End;
cl_kernel kernelCalculateIter2;
cl_kernel kernelCalculateFinishPole;

size_t CUDA_grid_dim;
//int CUDA_grid_dim_precalc;

// NOTE: global to one thread
#if !defined _WIN32
// TODO: Check compiler version. If  GCC 4.8 or later is used switch to 'alignas(n)'.
#if defined (INTEL)
cl_uint faOptimizedSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)aligned_alloc(4096, faOptimizedSize);
#else
// freq_context* Fa; // __attribute__((aligned(8)));
cl_uint faSize = (sizeof(freq_context) / 128 + 1) * 128;
auto Fa = (freq_context*)aligned_alloc(128, faSize);
// freq_context* Fa __attribute__((aligned(8))) = static_cast<freq_context*>(malloc(sizeof(freq_context)));
#endif
#else // WIN32

#if defined INTEL
cl_uint faOptimizedSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)_aligned_malloc(faOptimizedSize, 4096);
#elif defined AMD
//cl_uint faSize = sizeof(freq_context);
//alignas(8) freq_context* Fa;
cl_uint faSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)_aligned_malloc(faSize, 128);
#elif defined NVIDIA
#endif

#endif

double* pee, * pee0, * pWeight;

unsigned char* GetKernelBinaries(cl_program binProgram, const size_t binary_size)
{
    auto binary = new unsigned char[binary_size];
    cl_int err = clGetProgramInfo(binProgram, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, NULL);

    return binary;
}

cl_int SaveKernelsToBinary(cl_program binProgram, const char* kernelFileName)
{

    size_t binary_size;
    clGetProgramInfo(binProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
    auto binary = GetKernelBinaries(binProgram, binary_size);

    FILE* fp = fopen(kernelFileName, "wb+");
    if (!fp) {
        cerr << "Error while saving kernels binary file." << endl;
        return 1;
    }

    fwrite(binary, binary_size, 1, fp);
    fclose(fp);
    free(binary);

    //std::ofstream file(kernelFileName, std::ios::binary);
    ////size_t binary_size = file.tellg();
    ////file.seekg(0, std::ios::beg);
    ////char* binary = new char[*binary_size];
    //file.write(binary, binary_size);
    //file.close();

    return 0;
}

cl_int ClPrepare(cl_int deviceId, cl_double* beta_pole, cl_double* lambda_pole, cl_double* par, cl_double lcoef, cl_double a_lamda_start, cl_double a_lamda_incr,
    cl_double ee[][3], cl_double ee0[][3], cl_double* tim, cl_double Phi_0, cl_int checkex, cl_int ndata)
{
#if !defined _WIN32

#else
#ifndef INTEL
    //Fa = static_cast<freq_context*>(malloc(sizeof(freq_context)));
#else

#endif
#endif

    //try {
    cl_int err_num;
    cl_uint num_platforms_available;
    err_num = clGetPlatformIDs(0, NULL, &num_platforms_available);

    auto platforms = new cl_platform_id[num_platforms_available];
    err_num = clGetPlatformIDs(num_platforms_available, platforms, NULL);
    // clGetPlatformIDs(1, platforms, &num_platforms);
    // vector<cl::Platform>::iterator iter;
    cl_platform_id platform = nullptr;
#if !defined _WIN32
    char name[1024];
    char vendor[1024];
#else
#if defined AMD
    auto name = new char[1024];
    auto vendor = new char[1024];
#else
    cl::STRING_CLASS name;
    cl::STRING_CLASS vendor;
#endif
#endif

    //for (iter = platforms.begin(); iter != platforms.end(); ++iter)
    for (uint i = 0; i < num_platforms_available; i++)
    {
        platform = platforms[i];
        err_num = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 1024, name, NULL);
        err_num = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 1024, vendor, NULL);
        if (!strcmp(name, "Clover")) {
            continue;
        }
#if defined (AMD)
        if (!strcmp(vendor, "Advanced Micro Devices, Inc."))
        {
            break;
        }
#elif defined (NVIDIA)
        if (!strcmp(vendor, "NVIDIA Corporation"))
        {
            break;
        }
#elif defined (INTEL)
        if (!strcmp(vendor, "Intel(R) Corporation"))
        {
            break;
        }
#endif
        if (!strcmp(name, "rusticl"))
        {
            break;
        }
    }

    std::cerr << "Platform name: " << name << endl;
    std::cerr << "Platform vendor: " << vendor << endl;

    // auto platform = (*iter)();
    // cl_int errNum;
    //cl_device_id deviceIds = new int[numDevices];
    // cl_device_id* deviceIds;

    // Detect OpenCL devices
    cl_uint numDevices;
    err_num = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

    cl_device_id* devices = new cl_device_id[numDevices];
    err_num = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);


    if (numDevices < 1)
    {
        cerr << "No GPU device found for platform " << vendor << "(" << name << ")" << endl;
        return (1);
    }

    // if (numDevices > 0)
    // {
    // 	// TODO: Tedt with CL_DEVICE_TYPE_ALL & CL_DEVICE_TYPE_CPU
    // 	deviceIds = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id)); // << GNUC? alloca
    // 	err_num = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, deviceIds, NULL);
    // }


    // auto dev1 = deviceIds[deviceId];
    // auto device = cl::Device(dev1);
    // for (int i = 0; i < numDevices; i++)
    // {
    // 	devices.push_back(cl::Device(deviceIds[i]));
    // }


    // cl_device_id device = devices[1]; // RX 550

    cl_device_id device = devices[deviceId];
    // Create an OpenCL context for the choosen device
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    // cl_context clContext;
    context = clCreateContext(properties, 1, &device, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS) {
        cerr << "Error: Failed to create a device group! " << cl_error_to_str(err_num) << " (" << err_num << ")" << endl;
        return EXIT_FAILURE;
    }

    // deviceId = 0;
    const uint strBufSize = 1024;
    // char deviceName[strBufSize];
    char deviceVendor[strBufSize];
    char driverVersion[strBufSize];

#if !defined _WIN32
#if defined INTEL
#else
    char deviceName[strBufSize]; // Another AMD thing... Don't ask
    err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), &deviceName, NULL);
#endif
#else
#if defined INTEL
    size_t nameSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &nameSize);
    //auto deviceNameChars = new char[nameSize];
    auto deviceName = (char*)malloc(nameSize);
    //char deviceName[strBufSize];
    err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char) * nameSize, &deviceName, NULL);
#else
    char deviceName[strBufSize]; // Another AMD thing... Don't ask
    err_num = clGetDeviceInfo(device, CL_DEVICE_BOARD_NAME_AMD, sizeof(deviceName), &deviceName, NULL);
#endif

#endif

    //const auto devicePlatform = device.getInfo<CL_DEVICE_PLATFORM>();
    err_num = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor), &deviceVendor, NULL);
    err_num = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driverVersion), &driverVersion, NULL);

    //size_t valueSize;

    char openClVersion[strBufSize];
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, strBufSize, openClVersion, NULL);

    char clDeviceVersion[strBufSize];
    clGetDeviceInfo(device, CL_DEVICE_VERSION, strBufSize, clDeviceVersion, NULL);

    // cl_device_exec_capabilities
    char clDeviceExtensionCapabilities[strBufSize];
    err_num = clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES, strBufSize, &clDeviceExtensionCapabilities, NULL);

    cl_device_fp_config deviceDoubleFpConfig;
    err_num = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &deviceDoubleFpConfig, NULL);

    cl_ulong clDeviceGlobalMemSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &clDeviceGlobalMemSize, NULL);

    cl_ulong clDeviceLocalMemSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &clDeviceLocalMemSize, NULL);

    uint clDeviceMaxConstantArgs;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(uint), &clDeviceMaxConstantArgs, NULL);

    unsigned long long clDeviceMaxConstantBufferSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(unsigned long long), &clDeviceMaxConstantBufferSize, NULL);

    size_t clDeviceMaxParameterSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &clDeviceMaxParameterSize, NULL);

    unsigned long long clDeviceMaxMemAllocSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(unsigned long long), &clDeviceMaxMemAllocSize, NULL);

    cl_ulong clGlobalMemory;
    err_num = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &clGlobalMemory, NULL);
    cl_ulong globalMemory = clGlobalMemory / 1048576;

    cl_uint msCount;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &msCount, NULL);

    uint block;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_SAMPLERS, sizeof(uint), &block, NULL);

    cl_uint baseAddrAlign;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &baseAddrAlign, NULL);

    cl_uint minDataTypeAlignSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(cl_uint), &minDataTypeAlignSize, NULL);

    size_t extSize;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &extSize);
    auto deviceExtensions = new char[extSize];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extSize, deviceExtensions, NULL);

    size_t devMaxWorkGroupSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &devMaxWorkGroupSize, NULL);

    cl_uint devMaxWorkItemDims;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &devMaxWorkItemDims, NULL);

    //size_t *devWorkItemSizes = new size_t[devMaxWorkItemDims];
    //auto devWorkItemSizes = new size_t[devMaxWorkItemDims]{ 0,0,0 };

    //auto devWorkItemSizes = (size_t*)malloc(devMaxWorkItemDims);
    size_t devWorkItemSizes[3];
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(devWorkItemSizes), &devWorkItemSizes, NULL);

    cerr << "OpenCL device C version: " << openClVersion << " | " << clDeviceVersion << endl;
    cerr << "OpenCL device Id: " << deviceId << endl;
    string sufix = "MB";
    if (globalMemory > 1024) {
        globalMemory /= 1024;
        sufix = "GB";
    }
    cerr << "OpenCL device name: " << deviceName << " " << globalMemory << sufix << endl;
    cerr << "Device driver version: " << driverVersion << endl;
    cerr << "Multiprocessors: " << msCount << endl;
    cerr << "Max Samplers: " << block << endl;
    cerr << "Max work item dimensions: " << devMaxWorkItemDims << endl;
#ifdef _DEBUG
    cerr << "Debug info:" << endl;
    cerr << "CL_DEVICE_EXTENSIONS: " << deviceExtensions << endl;
    cerr << "CL_DEVICE_GLOBAL_MEM_SIZE: " << globalMemory << sufix << endl;
    cerr << "CL_DEVICE_LOCAL_MEM_SIZE: " << clDeviceLocalMemSize << " B" << endl;
    cerr << "CL_DEVICE_MAX_CONSTANT_ARGS: " << clDeviceMaxConstantArgs << endl;
    cerr << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << clDeviceMaxConstantBufferSize << " B" << endl;
    cerr << "CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << baseAddrAlign << endl;
    cerr << "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: " << minDataTypeAlignSize << endl;
    cerr << "CL_DEVICE_MAX_PARAMETER_SIZE: " << clDeviceMaxParameterSize << " B" << endl;
    cerr << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << clDeviceMaxMemAllocSize << " B" << endl;
    cerr << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << devMaxWorkGroupSize << endl;
    cerr << "CL_DEVICE_MAX_WORK_ITEM_SIZES: {";

    for (size_t work_item_dim = 0; work_item_dim < devMaxWorkItemDims; work_item_dim++) {
        if (work_item_dim > 0) cerr << ", ";
        cerr << (long int)devWorkItemSizes[work_item_dim];
    }
    cerr << "}" << endl;

#endif
    // cl_khr_fp64 || cl_amd_fp64
    bool isFp64 = string(deviceExtensions).find("cl_khr_fp64") != std::string::npos
        || string(deviceExtensions).find("cl_amd_fp64") != std::string::npos;

    bool doesNotSupportsFp64 = !isFp64;
    if (doesNotSupportsFp64)
    {
        fprintf(stderr, "Double precision floating point not supported by OpenCL implementation on current device. Exiting...\n");
        return (1);
    }

    auto SMXBlock = block; // 32;
    //CUDA_grid_dim = msCount * SMXBlock; //  24 * 32
    //CUDA_grid_dim = 8 * 32 = 256; 6 * 32 = 192
    CUDA_grid_dim = 2 * msCount * SMXBlock; // 256 (RX 550), 384 (1050Ti), 1536 (Nvidia GTX1660Ti), 768 (Intel Graphics HD)
    std::cerr << "Resident blocks per multiprocessor: " << SMXBlock << endl;
    std::cerr << "Grid dim: " << CUDA_grid_dim << " = 2 * " << msCount << " * " << SMXBlock << endl;
    std::cerr << "Block dim: " << BLOCK_DIM << endl;

    //int err;

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

    string kernelSourceFile = "kernelSource.cl";
    const char* kernelFileName = "kernels.bin";
#if defined (_DEBUG)
#if !defined _WIN32
    // Load CL file, build CL program object, create CL kernel object
    std::ifstream constantsFile("constants.h", std::ios::in | std::ios::binary);
    std::ifstream globalsFile("GlobalsCL.h", std::ios::in | std::ios::binary);
    std::ifstream intrinsicsFile("Intrinsics.cl", std::ios::in | std::ios::binary);
    std::ifstream swapFile("swap.cl", std::ios::in | std::ios::binary);
    std::ifstream blmatrixFile("blmatrix.cl", std::ios::in | std::ios::binary);
    std::ifstream curvFile("curv.cl", std::ios::in | std::ios::binary);
    std::ifstream curv2File("Curv2.cl", std::ios::in | std::ios::binary);
    std::ifstream mrqcofFile("mrqcof.cl", std::ios::in | std::ios::binary);
    std::ifstream startFile("Start.cl", std::ios::in | std::ios::binary);
    std::ifstream brightFile("bright.cl", std::ios::in | std::ios::binary);
    std::ifstream convFile("conv.cl", std::ios::in | std::ios::binary);
    std::ifstream mrqminFile("mrqmin.cl", std::ios::in | std::ios::binary);
    std::ifstream gauserrcFile("gauss_errc.cl", std::ios::in | std::ios::binary);
    std::ifstream testFile("test.cl", std::ios::in | std::ios::binary);
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

    auto kernel_code = st.str(); //.c_str();
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

    // cerr << kernel_code << endl;
    std::ofstream out(kernelSourceFile, std::ios::out | std::ios::binary);
    out << kernel_code;
    out.close();
#endif

    std::ifstream f(kernelFileName);
    bool kernelExist = f.good();

    bool readsource = false;
#if defined (_DEBUG)
    readsource = true;
#endif

    // cl::Program::Sources sources(1, std::make_pair(kernel_code.c_str(), kernel_code.length()));	// cl::Program::Sources sources;
    // sources.push_back({kernel_code.c_str(), kernel_code.length()});

    // program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &err_num);

    if (!kernelExist || readsource)
    {
        //auto kernelSource = ocl_src_kernelSource.c_str();
        binProgram = clCreateProgramWithSource(context, 1, (const char**)&ocl_src_kernelSource, NULL, &err_num);
        if (!binProgram || err_num != CL_SUCCESS)
        {
            cerr << "Error: Failed to create compute program! " << cl_error_to_str(err_num) << " (" << err_num << ")" << endl;
            return EXIT_FAILURE;
        }

        //char options[]{ "-Werror" };
        char options[]{ "-w" };
#if defined (AMD)
        err_num = clBuildProgram(binProgram, 1, &device, options, NULL, NULL); // "-Werror -cl-std=CL1.1"
#elif defined (NVIDIA)
        binProgram.build(devices, "-D NVIDIA -w -cl-std=CL1.2"); // "-w" "-Werror"
#elif defined (INTEL)
        binProgram.build(devices, "-D INTEL -cl-std=CL1.2");
#endif

#if defined (NDEBUG)
        std::ifstream fs(kernelFileName);
        bool kernelExist = fs.good();
        if (kernelExist) {
            std::remove(kernelSourceFile.c_str());
        }
#endif

        size_t len;
        err_num = clGetProgramBuildInfo(binProgram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        //char* buildlog = (char*)calloc(len, sizeof(char));
        auto buildlog = new char[len];
        err_num = clGetProgramBuildInfo(binProgram, device, CL_PROGRAM_BUILD_LOG, len, buildlog, NULL);

        cl_build_status buildStatus;
        err_num = clGetProgramBuildInfo(binProgram, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);

        std::string buildlogStr = buildlog;
        if (buildStatus == 0)
        {
            //strcpy(buildlog, "Ok");
            buildlogStr.append("OK");
        }

        cerr << "Binary build log for " << deviceName << ":" << std::endl << buildlogStr << " (" << buildStatus << ")" << endl;
        delete[] buildlog;
        err_num = SaveKernelsToBinary(binProgram, kernelFileName);
        if (err_num > 0)
        {
            return err_num;
        }
    }

    try
    {
        std::ifstream file(kernelFileName, std::ios::binary | std::ios::in | std::ios::ate);
        size_t binary_size = file.tellg();
        file.seekg(0, std::ios::beg);
        char* binary = new char[binary_size];
        file.read(binary, binary_size);
        //file.close();

        //size_t* binary_size = (size_t*)malloc(sizeof(size_t));
        //FILE* fp = fopen(kernelFileName, "rb");
        //if (!fp) {
        //    cerr << "Error while reading kernels binary file." << endl;
        //}

        //fseek(fp, 0, SEEK_END);
        //*binary_size = ftell(fp);
        //fseek(fp, 0, SEEK_SET);

        //char* binary = (char*)malloc(*binary_size);
        //if (!binary) {
        //    fclose(fp);
        //    cerr << "Error while reading kernels binary file." << endl;
        //}

        //fread(binary, *binary_size, 1, fp);
        //fclose(fp);

        //free(fp);

        //auto kSource = kernel_code.c_str();
        //program = clCreateProgramWithSource(context, 1, (const char**)&kSource, NULL, &err_num);

        //program = clCreateProgramWithSource(context, 1, (const char**)&ocl_src_kernelSource, NULL, &err_num);
        cl_int binary_status;
        program = clCreateProgramWithBinary(context, 1, &device, &binary_size, (const unsigned char**)&binary, &binary_status, &err_num);

        //char options[]{ "-Werror" };
        char options[]{ "-w" };
#if defined (AMD)
        err_num = clBuildProgram(program, 1, &device, options, NULL, NULL); // "-Werror -cl-std=CL1.1" "-g -x cl -cl-std=CL1.2 -Werror"
#elif defined (NVIDIA)
        program.build(devices); //, "-D NVIDIA -w -cl-std=CL1.2"); // "-Werror" "-w"
#elif defined (INTEL)
        program.build(devices, "-D INTEL -cl-std=CL1.2");
#endif
        if (err_num != CL_SUCCESS)
        {
            size_t len;
            //size_t* len = (size_t*)malloc(sizeof(size_t));
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            //char* buffer = (char*)calloc(len, sizeof(char));
            auto buffer = new char[len];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
            cerr << "Build log: " << name << " | " << deviceName << ":" << endl << buffer << endl;
            std::cerr << " Error creating queue: " << cl_error_to_str(err_num) << "(" << err_num << ")\n";
            delete[] buffer;
            //free(len);
            //free(binary);
            //free(binary_size);

            return(1);
        }

        //cl_command_queue_properties properties;
        queue = clCreateCommandQueue(context, device, 0, &err_num);
        if (err_num != CL_SUCCESS) {
            std::cerr << " Error creating queue: " << cl_error_to_str(err_num) << "(" << err_num << ")\n";
            return(1);
        }

        //free(binary);
        //free(binary_size);

        char* buildlog = new char[strBufSize];
        err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, strBufSize, buildlog, NULL);
        cl_build_status buildStatus;
        err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, NULL);

#if _DEBUG
#if CL_TARGET_OPENCL_VERSION > 110
        size_t bufSize;
        size_t numKernels;
        err_num = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(numKernels), &numKernels, NULL);
        //auto kernels = program.getInfo<CL_PROGRAM_NUM_KERNELS>();
        err_num = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &bufSize);
        char* kernelNames = new char[bufSize];
        err_num = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, bufSize, kernelNames, NULL);
        //auto kernelNames = program.getInfo<CL_PROGRAM_KERNEL_NAMES>();
        cerr << "Kernels: " << numKernels << endl;
        cerr << "Kernel names: " << endl << kernelNames << endl;
#endif
        char buildOptions[1024];
        err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, sizeof(buildOptions), buildOptions, NULL);
        std::cerr << "Build options: " << buildOptions << std::endl;
        //std::string programSource = program.getInfo<CL_PROGRAM_SOURCE>();
        // std::cerr << "Program source: " << std::endl;
        // 	std::cerr << programSource << std::endl;
#endif
        std::string buildlogStr = buildlog;
        if (buildStatus == 0)
        {
            // strcpy(buildlog, "Ok");
            buildlogStr.append("OK");
        }

        //char deviceName[128]; // Another AMD thing... Don't ask
        err_num = clGetDeviceInfo(device, CL_DEVICE_BOARD_NAME_AMD, sizeof(deviceName), &deviceName, NULL);

        cerr << "Program build log for " << deviceName << ":" << std::endl << buildlogStr << " (" << buildStatus << ")" << endl;
    }

    catch (Error& e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            // Check the build status
            cl_build_status status1;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status1), &status1, NULL);
            if (status1 == CL_BUILD_ERROR) // && status2 != CL_BUILD_ERROR)
            {
                // Get the build log
                char name[1024];
                err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
                char* buildlog = new char[strBufSize];
                err_num = clGetDeviceInfo(device, CL_PROGRAM_BUILD_LOG, sizeof(char) * 1024, buildlog, NULL);
                cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
            }
        }
        else
        {
            char* buildlog = new char[strBufSize];
            err_num = clGetDeviceInfo(device, CL_PROGRAM_BUILD_LOG, sizeof(char) * 1024, buildlog, NULL);
            cl_build_status buildStatus;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, NULL);
            std::cerr << "OpenCL error: " << cl_error_to_str(e.err()) << "(" << e.err() << ")" << std::endl;
            std::cerr << buildStatus << std::endl;
            // std::cerr << "Device driver: " << deviceDriver << std::endl;
            // std::cerr << "Build options: " << buildOptions << std::endl;
            std::cerr << "Build log for " << name << " | " << deviceName << ":" << std::endl << buildlog << std::endl;
            // std::cerr << "Program source: " << std::endl;
            // std::cerr << programSource << std::endl;
            // fprintf(stderr, "Build log for %s: %s\n", name.c_str(), buildlog.c_str());
        }

        return 2;
    }

#pragma region Kernel creation
    cl_int kerr;
    try
    {
        kernelClCheckEnd = clCreateKernel(program, "ClCheckEnd", &kerr);
        kernelCalculatePrepare = clCreateKernel(program, string("ClCalculatePrepare").c_str(), &kerr);
        kernelCalculatePreparePole = clCreateKernel(program, string("ClCalculatePreparePole").c_str(), &kerr);
        kernelCalculateIter1Begin = clCreateKernel(program, string("ClCalculateIter1Begin").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Start = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Start").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Matrix = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Matrix").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Curve1 = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Curve1").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Curve2 = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Curve2").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Curve1Last = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Curve1Last").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1End = clCreateKernel(program, string("ClCalculateIter1Mrqcof1End").c_str(), &kerr);
        kernelCalculateIter1Mrqmin1End = clCreateKernel(program, string("ClCalculateIter1Mrqmin1End").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Start = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Start").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Matrix = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Matrix").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Curve1 = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Curve1").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Curve2 = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Curve2").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Curve1Last = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Curve1Last").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2End = clCreateKernel(program, "ClCalculateIter1Mrqcof2End", &kerr);
        kernelCalculateIter1Mrqmin2End = clCreateKernel(program, "ClCalculateIter1Mrqmin2End", &kerr);
        kernelCalculateIter2 = clCreateKernel(program, "ClCalculateIter2", &kerr);
        kernelCalculateFinishPole = clCreateKernel(program, "ClCalculateFinishPole", &kerr);
    }
    catch (Error& e)
    {
        cerr << "Error creating kernel: \"" << cl_error_to_str(e.err()) << "\"(" << e.err() << ") - " << e.what() << " | " << cl_error_to_str(kerr) <<
            " (" << kerr << ")" << std::endl;
        cout << "Error while creating kernel. Check stderr.txt for details." << endl;
        return(4);
    }
#pragma endregion

#ifndef CL_PROGRAM_NUM_KERNELS
#define CL_PROGRAM_NUM_KERNELS                      0x1167
#define CL_PROGRAM_KERNEL_NAMES                     0x1168
#endif
#if defined _DEBUG
    // size_t numKernels;
    // err = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t), &numKernels, NULL);
    // size_t kernelNamesSize;
    // err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &kernelNamesSize);
    // char kernelNamesChars[kernelNamesSize];
    // // auto kernelNames = (char*) malloc(numKernels);
    // err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, sizeof(kernelNamesChars), kernelNamesChars, NULL);
    // cerr << "Kernel names: " << kernelNamesChars << endl;
    // std::vector<char*> kernelNames;
    // char* kernel_chars = strtok(kernelNamesChars, ";");
    // while(kernel_chars)
    // {
    //     kernelNames.push_back(kernel_chars);
    //     kernel_chars = strtok(NULL, ";");
    // }

    // cerr << "Prefered kernel work group size - kernel | size:" << endl;
    // size_t preferedWGS[numKernels];
    // for(int k = 0; k < numKernels; k++){
    // 	cl_kernel kernel = clCreateKernel(program, kernelNames[k], &kerr);
    // 	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferedWGS[k], NULL);
    // 	cerr << kernelNames[k] << " | " << preferedWGS[k] << endl;
    // }
#endif

    size_t preferedWGS;
    err_num = clGetKernelWorkGroupInfo(kernelCalculateIter1Mrqmin1End, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferedWGS, NULL);
    if (err_num != CL_SUCCESS) {
        std::cerr << " Error in clGetKernelWorkGroupInfo: " << cl_error_to_str(err_num) << "(" << err_num << ")\n";
        return(1);
    }
    cerr << "Prefered kernel work group size multiple: " << preferedWGS << endl;


    if (CUDA_grid_dim > devMaxWorkGroupSize) {
        CUDA_grid_dim = devMaxWorkGroupSize;
        cerr << "Setting Grid Dim to " << CUDA_grid_dim << endl;
    }

    return 0;
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

    cl_int err = 0;
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

    size_t CUDA_grid_dim_precalc = CUDA_grid_dim;
    if (max_test_periods < CUDA_grid_dim_precalc)
    {
        CUDA_grid_dim_precalc = max_test_periods;
    }

    /* totalWorkItems = CUDA_grid_dim_precalc * BLOCK_DIM */
    size_t totalWorkItems = CUDA_grid_dim_precalc * BLOCK_DIM;

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
    //auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
     //queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);

    cl_mem cgFirst = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (MAX_N_PAR + 1), cg_first, &err);
    clEnqueueWriteBuffer(queue, cgFirst, CL_BLOCKING, 0, sizeof(cl_double)* (MAX_N_PAR + 1), cg_first, 0, NULL, NULL);

    //cl_mem cgFirst = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_double) * (MAX_N_PAR + 1), cg_first, &err);
#endif

#if !defined _WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    // cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    // auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);

    // cl_uint pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    // void* pcc = reinterpret_cast<mfreq_context*>(malloc(pccSize));

    // auto pcc __attribute__((aligned(8))) = new mfreq_context[CUDA_grid_dim_precalc];
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
    // auto mcc __attribute__((aligned(8))) = new mfreq_context[CUDA_grid_dim_precalc];

    // cl_uint pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    // auto memPcc = (mfreq_context *)aligned_alloc(128, pccSize);
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pccSize, pcc, err);
    
    // cl_uint pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    // auto pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc) / 128 + 1) * 128;
    auto pcc = (mfreq_context*)aligned_alloc(128, pccSize);

    // auto pcc = queue.enqueueMapBuffer(CUDA_MCC2, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, pccSize, NULL, NULL, err);
    // queue.flush();
    // void* pcc = clEnqueueMapBuffer(queue, CUDA_MCC2, CL_BLOCKING, CL_MAP_WRITE, 0, pccSize, 0, NULL, NULL, &err);

#elif NVIDIA
    int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#else // WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    //cl_uint pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    //auto memPcc = (mfreq_context*)_aligned_malloc(pccSize, 128);
    //cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pccSize, memPcc, &err);
    //void *pcc = clEnqueueMapBuffer(queue, CUDA_MCC2, CL_BLOCKING, CL_MAP_WRITE, 0, pccSize, 0, NULL, NULL, &err);

    // 18-SEP-2023
    //size_t pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    //auto pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc) / 128 + 1) * 128;
    auto pcc = (mfreq_context*)_aligned_malloc(pccSize, 128);
#elif NVIDIA
    int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#endif

    // NOTE: NOTA BENE - In contrast to Cuda, where global memory is zeroed by itself, here we need to initialize the values in each dimension. GV-26.09.2020
    // <<<<<<<<<<<
    for (m = 0; m < CUDA_grid_dim_precalc; m++)
    {
        // std::fill_n(&((mfreq_context*)pcc)[m].Area, MAX_N_FAC + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Area), MAX_N_FAC + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Dg), (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].alpha), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].covar), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].beta), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].da), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].atry), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].dave), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].dytemp), (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].ytemp), POINTS_MAX + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_big), BLOCK_DIM, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_icol), BLOCK_DIM, 0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_irow), BLOCK_DIM, 0);
        //pcc[m].conw_r = 0.0;
        ((mfreq_context*)pcc)[m].icol = 0;
        ((mfreq_context*)pcc)[m].pivinv = 0;
    }

    //auto t = sizeof(struct mfreq_context); // 351160 B / 384 * 351160 = 134,845,440 (128.6MB)   // 4232760 B / 384 * 4232760 = 1,625,379,840 ( 1.6 GB)
    //auto tt = t * CUDA_grid_dim_precalc;

    //void* test;
    //int testSize = 10 * sizeof(double);
    //test = malloc(testSize);

    //auto CUDA_TEST = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 10 * sizeof(double), test, err);
    //auto clTest = queue.enqueueMapBuffer(CUDA_TEST, CL_NON_BLOCKING, CL_MAP_WRITE, 0, 10 * sizeof(double), NULL, NULL, err);

#if defined (INTEL)
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#elif defined AMD
    // queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
    // queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
    // err = clEnqueueWriteBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
    // queue.enqueueUnmapMemObject(CUDA_MCC2, pcc);
    // queue.flush();

     //clEnqueueUnmapMemObject(queue, CUDA_MCC2, pcc, 0, NULL, NULL);
     //clFlush(queue);

    // 18-SEP-2023
    cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, &err);
    clEnqueueWriteBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
#elif defined NVIDIA
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
#endif

    //auto clPcc = queue.enqueueMapBuffer(CUDA_MCC2, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, pccSize, NULL, NULL, &r);
    //queue.enqueueUnmapMemObject(CUDA_MCC2, clPcc);

#if !defined _WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    // int faSize = sizeof(freq_context);
    // cl_int faSize = sizeof(freq_context);
    // cl_uint faSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
    // auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, err);
    // auto pFa = queue.enqueueMapBuffer(CUDA_CC, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, faSize);
    
    // auto memFa = (freq_context*)aligned_alloc(128, faSize);
    // cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, &err);
    // void* pFa = clEnqueueMapBuffer(queue, CUDA_CC, CL_BLOCKING, CL_MAP_WRITE, 0, faSize, 0, NULL, NULL, &err);
    // memcpy(pFa, Fa, faSize);
    // clEnqueueUnmapMemObject(queue, CUDA_CC, pFa, 0, NULL, NULL);
    // clFlush(queue);
    auto pFa = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, pFa, &err);
    clEnqueueWriteBuffer(queue, CUDA_CC, CL_BLOCKING, 0, faSize, Fa, 0, NULL, NULL);

#endif
#else // WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    /*auto memFa = (freq_context*)_aligned_malloc(faSize, 128);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, &err);
    void* pFa = clEnqueueMapBuffer(queue, CUDA_CC, CL_BLOCKING, CL_MAP_WRITE, 0, faSize, 0, NULL, NULL, &err);
    memcpy(pFa, Fa, faSize);
    clEnqueueUnmapMemObject(queue, CUDA_CC, pFa, 0, NULL, NULL);
    clFlush(queue);*/
    auto pFa = (freq_context*)_aligned_malloc(faSize, 128);
    //memcpy(pFa, Fa, faSize);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, pFa, &err);
    clEnqueueWriteBuffer(queue, CUDA_CC, CL_BLOCKING, 0, faSize, Fa, 0, NULL, NULL);
#endif
#endif

    // auto CUDA_CC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFb, err);
#if !defined _WIN32
    auto pFb = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, pFb, &err);
#else
    auto pFb = (freq_context*)_aligned_malloc(faSize, 128);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, pFb, &err);
#endif

    //cl_int* end = (cl_int*)malloc(sizeof(cl_int));
    //*end = -90;

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
    // auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(theEnd), &theEnd, err);
    // queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd);
    cl_mem CUDA_End = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(theEnd), &theEnd, &err);
    err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);

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

#if !defined _WIN32
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    // cl_int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    // cl_uint frSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    // void *memIn = (void *)aligned_alloc(8, frSize);
    // void *memIn = (void *)aligned_alloc(8, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    // void* pfr;

    // void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
    // void *pfr = (void *)aligned_alloc(8, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, CUDA_grid_dim_precalc * sizeof(freq_result), pfr, err);
    // auto memFr __attribute__((aligned(8))) = new freq_result[CUDA_grid_dim_precalc];
    // auto memFr = (freq_result *)aligned_alloc(128, frSize);
    // auto memFr = new (freq_result *)aligned_alloc(128, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,  frSize, memFr, err);
    cl_uint frSize = (sizeof(freq_result) * CUDA_grid_dim_precalc / 128 + 1) * 128;
    // auto pfr = new freq_result[CUDA_grid_dim_precalc];
    auto pfr = (freq_result*)aligned_alloc(128, frSize);
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
    // void *pfr;
#elif NVIDIA
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void* memIn = (void*)aligned_alloc(8, frSize);
#endif // NVIDIA
#else // WIN
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    //int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    //void* memIn = (void*)_aligned_malloc(frSize, 256);
    //auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    //void* pfr;

    //size_t frSize = sizeof(freq_result) * CUDA_grid_dim_precalc;
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    auto pfr = (freq_result*)_aligned_malloc(frSize, 128);
    //auto pfr = new freq_result[CUDA_grid_dim_precalc];
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
#elif NVIDIA
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void* memIn = (void*)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void* pfr;
#endif // NViDIA
#endif // WIN

#pragma region SetKernelArgs
    err = clSetKernelArg(kernelClCheckEnd, 0, sizeof(cl_mem), &CUDA_End);

    err = clSetKernelArg(kernelCalculatePrepare, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePrepare, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePrepare, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePrepare, 3, sizeof(freq_start), &freq_start);
    err = clSetKernelArg(kernelCalculatePrepare, 4, sizeof(freq_step), &freq_step);
    err = clSetKernelArg(kernelCalculatePrepare, 5, sizeof(max_test_periods), &max_test_periods);

    err = clSetKernelArg(kernelCalculatePreparePole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePreparePole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculatePreparePole, 2, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePreparePole, 3, sizeof(cl_mem), &cgFirst);
    err = clSetKernelArg(kernelCalculatePreparePole, 4, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePreparePole, 5, sizeof(cl_mem), &CUDA_CC2);
    //kernelCalculatePreparePole.setArg(5, sizeof(double), &lcoef);
    // NOTE: 7th arg 'm' is set a little further as 'm' is an iterator counter

    err = clSetKernelArg(kernelCalculateIter1Begin, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Begin, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculateIter1Begin, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculateIter1Begin, 3, sizeof(int), &n_iter_min);
    err = clSetKernelArg(kernelCalculateIter1Begin, 4, sizeof(int), &n_iter_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 5, sizeof(double), &iter_diff_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 6, sizeof(double), &((*Fa).Alamda_start));

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateFinishPole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateFinishPole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateFinishPole, 2, sizeof(cl_mem), &CUDA_FR);
#pragma endregion

    // Allocate result space
    //res = (freq_result*)malloc(CUDA_grid_dim * sizeof(freq_result));
    // freq_result* fres; // = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);

    /* Sets local_work_size to BLOCK_DIM = 128 */
    size_t local = BLOCK_DIM;
    size_t sLocal = 1;

    for (n = 1; n <= max_test_periods; n += (int)CUDA_grid_dim_precalc)
    {

#if defined INTEL
        pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        queue.flush();
#elif defined AMD
        // pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        // pfr = clEnqueueMapBuffer(queue, CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, 0, NULL, NULL, &err);
        // queue.flush();
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

#if defined INTEL
        queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frOptimizedSize, pfr);
#elif AMD
        clEnqueueWriteBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#elif NVIDIA
        queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        queue.flush();
#endif
        err = clSetKernelArg(kernelCalculatePrepare, 6, sizeof(n), &n);
        err = EnqueueNDRangeKernel(queue, kernelCalculatePrepare, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
        if (getError(err)) return err;
        //clFinish(queue);

        for (m = 1; m <= N_POLES; m++)
        {
            theEnd = 0; //zero global End signal
            err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);
            err = clSetKernelArg(kernelCalculatePreparePole, 6, sizeof(m), &m);
            err = EnqueueNDRangeKernel(queue, kernelCalculatePreparePole, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;
            //clFinish(queue);

            //void* pFb = clEnqueueMapBuffer(queue, CUDA_CC2, CL_BLOCKING, CL_MAP_READ, 0, faSize, 0, NULL, NULL, &err);
            //clFlush(queue);
            clEnqueueReadBuffer(queue, CUDA_CC2, CL_BLOCKING, 0, faSize, pFb, 0, NULL, NULL);
            int error = 0;
            for (int j = 0; j < MAX_N_OBS + 1; j++) {
                if ((*(freq_context*)pFb).Brightness[j] != (*Fa).Brightness[j]) {
                    error++;
                }
            }

            clEnqueueReadBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
            //pcc = clEnqueueMapBuffer(queue, CUDA_MCC2, CL_BLOCKING, CL_MAP_READ, 0, pccSize, 0, NULL, NULL, &err);
            //clFlush(queue);
            int errCnt = 0;
            for (int j = 0; j < CUDA_grid_dim_precalc; j++)
            {
                for (int i = 1; i <= n_coef; i++)
                {
                    auto CUDA_LCC = ((mfreq_context*)pcc)[j];
                    if (CUDA_LCC.cg[i] != cg_first[i])
                    {
                        errCnt++;
                    }
                    //if(blockIdx.x == 0)
                    //	printf("cg[%3d]: %10.7f\n", i, CUDA_cg_first[i]);
                }
            }
            //clEnqueueUnmapMemObject(queue, CUDA_MCC2, pcc, 0, NULL, NULL);
            clEnqueueUnmapMemObject(queue, CUDA_CC2, pFb, 0, NULL, NULL);
            clFlush(queue);
#ifdef _DEBUG
            // printf(".");
            cout << ".";
            cout.flush();
#endif
            int count = 0;
            while (!theEnd)
            {
                count++;
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Begin, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);
                for (iC = 1; iC < l_curves; iC++)
                {
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 2, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);
                }

                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1End, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin1End, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                for (iC = 1; iC < l_curves; iC++)
                {
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 2, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);
                }

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2End, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin2End, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue); // ***

                err = clEnqueueReadBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);

                // printf("[%d][%d][%d] %d\n", n, m, count, theEnd);
                theEnd = theEnd == CUDA_grid_dim_precalc;
            }

            err = EnqueueNDRangeKernel(queue, kernelCalculateFinishPole, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;
            //clFinish(queue);
        }

        printf("\n");

        //queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, res);
#if !defined _WIN32
#if defined (INTEL)
        fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
        queue.finish();
#elif AMD
        // queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, sizeof(frSize), pfr);
        // pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frSize, NULL, NULL, err);
        // pfr = clEnqueueMapBuffer(queue, CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frSize, 0, NULL, NULL, &err);
        //queue.flush(); // ***
        // queue.enqueueReadBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
        clEnqueueReadBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#elif NVIDIA
        pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        queue.flush();
#endif
#else
#if defined (INTEL)
        fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
        queue.finish();
#elif AMD
        clEnqueueReadBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#elif NVIDIA
        pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        queue.flush();
#endif
#endif
        //err=cudaThreadSynchronize(); memcpy is synchro itself

        //read results here
        //err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim_precalc, cudaMemcpyDeviceToHost);
#if defined (INTEL)
        auto res = (freq_result*)fres;
#else
        //auto res = (freq_result*)pfr;
        auto res = new freq_result[CUDA_grid_dim_precalc];
        memcpy(res, pfr, frSize);
#endif

        for (m = 1; m <= CUDA_grid_dim_precalc; m++)
        {
            if (res[m - 1].isReported == 1)
            {
                sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
#if defined _DEBUG
                printf("[%3d] res[%3d].dark_best: %10.16f, sum_dark_facet: %10.16f\n", m, m - 1, res[m - 1].dark_best, sum_dark_facet);
#endif
            }
        }

#if !defined _WIN32
#if defined (INTEL)
        queue.enqueueUnmapMemObject(CUDA_FR, fres);
        queue.flush();
#elif AMD
        // queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        // queue.flush();
        // clEnqueueUnmapMemObject(queue, CUDA_FR, pfr, 0, NULL, NULL);
        // clFlush(queue);
        delete[] res;
#elif NVIDIA
#elif NVIDIA
        queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        queue.flush();
#endif
#else
#if defined (INTEL)
        queue.enqueueUnmapMemObject(CUDA_FR, fres);
        queue.flush();
#elif AMD
        //queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        //queue.flush();
#elif NVIDIA
        queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        queue.flush();
#endif
#endif
    } /* period loop */

    clReleaseMemObject(CUDA_MCC2);
    clReleaseMemObject(CUDA_CC);
    clReleaseMemObject(CUDA_CC2);
    clReleaseMemObject(CUDA_End);
    clReleaseMemObject(CUDA_FR);
    clReleaseMemObject(cgFirst);

#if !defined _WIN32
#if defined INTEL
    free(pcc);
#elif defined AMD
    free(pcc);
    free(pFb);
    free(pfr);
#elif defined NVIDIA
    free(memIn);
    free(pcc);
    delete[] pcc;
#endif
#else // WIN
    //_aligned_free(pfr);  // res does not need to be freed as it's just a pointer to *pfr.
#if defined (INTEL)
    _aligned_free(pcc);
#elif defined AMD
    _aligned_free(pcc);
    _aligned_free(pFb);
    _aligned_free(pfr);
    //delete[] pfr;
    //_aligned_free(memPcc);
    //delete[] pcc;
#elif defined NVIDIA
    delete[] pcc;
#endif
#endif // WIN

    ave_dark_facet = sum_dark_facet / max_test_periods;

    if (ave_dark_facet < 1.0)
        *new_conw = 1; /* new correct conwexity weight */
    if (ave_dark_facet >= 1.0)
        *conw_r = *conw_r * 2;

#if defined _DEBUG
    printf("ave_dark_facet: %10.17f\n", ave_dark_facet);
    printf("conw_r:         %10.17f\n", *conw_r);
#endif
    return 0;
}

int ClStart(int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
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

    // cl_int* err = nullptr;
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

    cl_int err = 0;

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
    // auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
    // queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);
    cl_mem cgFirst = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_double) * (MAX_N_PAR + 1), cg_first, &err);
#endif

#if !defined _WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    // cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    // auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);

    cl_uint pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto pcc = new mfreq_context[CUDA_grid_dim];
#elif NVIDIA
    cl_uint pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#else  // WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    //cl_uint pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    //auto memPcc = (mfreq_context*)_aligned_malloc(pccSize, 128);
    //cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pccSize, memPcc, &err);
    //void* pcc = clEnqueueMapBuffer(queue, CUDA_MCC2, CL_BLOCKING, CL_MAP_WRITE, 0, pccSize, 0, NULL, NULL, &err);

    size_t pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto pcc = new mfreq_context[CUDA_grid_dim];
#elif NVIDIA
    int pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#endif


    //#if defined (INTEL)
    //	cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    //#if !defined _WIN32
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
        //std::fill_n(pcc[m].Area, MAX_N_FAC + 1, 0.0);
        //std::fill_n(pcc[m].Dg, (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
        //std::fill_n(pcc[m].alpha, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        //std::fill_n(pcc[m].covar, (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        //std::fill_n(pcc[m].beta, MAX_N_PAR + 1, 0.0);
        //std::fill_n(pcc[m].da, MAX_N_PAR + 1, 0.0);
        //std::fill_n(pcc[m].atry, MAX_N_PAR + 1, 0.0);
        //std::fill_n(pcc[m].dave, MAX_N_PAR + 1, 0.0);
        //std::fill_n(pcc[m].dytemp, (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
        //std::fill_n(pcc[m].ytemp, POINTS_MAX + 1, 0.0);
        //std::fill_n(pcc[m].sh_big, BLOCK_DIM, 0.0);
        //std::fill_n(pcc[m].sh_icol, BLOCK_DIM, 0);
        //std::fill_n(pcc[m].sh_irow, BLOCK_DIM, 0);
        ////pcc[m].conw_r = 0.0;
        //pcc[m].icol = 0;
        //pcc[m].pivinv = 0;

        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Area), MAX_N_FAC + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Dg), (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].alpha), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].covar), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].beta), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].da), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].atry), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].dave), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].dytemp), (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].ytemp), POINTS_MAX + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_big), BLOCK_DIM, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_icol), BLOCK_DIM, 0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_irow), BLOCK_DIM, 0);
        //pcc[m].conw_r = 0.0;
        ((mfreq_context*)pcc)[m].icol = 0;
        ((mfreq_context*)pcc)[m].pivinv = 0;
    }

#if !defined _WIN32
#if defined (INTEL)
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#else
    // queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
    cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, &err);
    clEnqueueWriteBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
#endif
#else // WIN32
#if defined (INTEL)
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#else
    cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, &err);
    clEnqueueWriteBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);

    //clEnqueueUnmapMemObject(queue, CUDA_MCC2, pcc, 0, NULL, NULL);
    //clFlush(queue);
#endif
#endif

#if !defined _WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    // cl_uint faSize = sizeof(freq_context);
    // auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, Fa, err);
    // queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);
    auto memFa = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, &err);
    void* pFa = clEnqueueMapBuffer(queue, CUDA_CC, CL_BLOCKING, CL_MAP_WRITE, 0, faSize, 0, NULL, NULL, &err);
    memcpy(pFa, Fa, faSize);
    clEnqueueUnmapMemObject(queue, CUDA_CC, pFa, 0, NULL, NULL);
    clFlush(queue);
#endif
#else // WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    // cl_uint faSize = sizeof(freq_context);
    // auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, Fa, err);
    // queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);
    auto memFa = (freq_context*)_aligned_malloc(faSize, 128);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, &err);
    void* pFa = clEnqueueMapBuffer(queue, CUDA_CC, CL_BLOCKING, CL_MAP_WRITE, 0, faSize, 0, NULL, NULL, &err);
    memcpy(pFa, Fa, faSize);
    clEnqueueUnmapMemObject(queue, CUDA_CC, pFa, 0, NULL, NULL);
    clFlush(queue);
#endif
#endif

#if defined (INTEL)
    auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int), &theEnd, err);
#else
    // auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
    // queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
    cl_mem CUDA_End = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(theEnd), &theEnd, &err);
    err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);
#endif

#if !defined _WIN32
    // freq_context* Fb;
    // auto CUDA_CC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(freq_context), Fb, err);
    auto memFb = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFb, &err);
#else
    auto memFb = (freq_context*)_aligned_malloc(faSize, 128);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFb, &err);
#endif

#if !defined _WIN32
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    // cl_uint frSize = CUDA_grid_dim * sizeof(freq_result);
    // void *memIn = (void *)aligned_alloc(128, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    // void *pfr;
    cl_uint frSize = sizeof(freq_result) * CUDA_grid_dim;
    auto pfr = new freq_result[CUDA_grid_dim];
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
#elif NVIDIA
    cl_uint = CUDA_grid_dim * sizeof(freq_result);
    void* memIn = (void*)aligned_alloc(8, frSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void* pfr;
#endif // NVIDIA
#else  // WIN
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    //int frSize = CUDA_grid_dim * sizeof(freq_result);
    //void* memIn = (void*)_aligned_malloc(frSize, 256);
    //auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    //void* pfr;
    size_t frSize = sizeof(freq_result) * CUDA_grid_dim;
    auto pfr = new freq_result[CUDA_grid_dim];
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
#elif NVIDIA
    int frSize = CUDA_grid_dim * sizeof(freq_result);
    void* memIn = (void*)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void* pfr;
#endif // NViDIA
#endif // WIN

    //#if defined (INTEL)
    //	cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    //#if !defined _WIN32
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
    //#if !defined _WIN32
    //	void* memIn = (void*)aligned_alloc(8, frSize);
    //#else
    //	void* memIn = (void*)_aligned_malloc(frSize, 256);
    //#endif
    //	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    //	void* pfr;
    //#endif

        //pfr = queue.enqueueMapBuffer(CUDA_FR, CL_NON_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        //queue.flush();

#pragma region SetKernelArguments
    err = clSetKernelArg(kernelClCheckEnd, 0, sizeof(cl_mem), &CUDA_End);

    err = clSetKernelArg(kernelCalculatePrepare, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePrepare, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePrepare, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePrepare, 3, sizeof(freq_start), &freq_start);
    err = clSetKernelArg(kernelCalculatePrepare, 4, sizeof(freq_step), &freq_step);
    err = clSetKernelArg(kernelCalculatePrepare, 5, sizeof(n_max), &n_max);

    err = clSetKernelArg(kernelCalculatePreparePole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePreparePole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculatePreparePole, 2, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePreparePole, 3, sizeof(cl_mem), &cgFirst);
    err = clSetKernelArg(kernelCalculatePreparePole, 4, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePreparePole, 5, sizeof(cl_mem), &CUDA_CC2);
    //kernelCalculatePreparePole.setArg(5, sizeof(double), &lcoef);
    // NOTE: 7th arg 'm' is set a little further as 'm' is an iterator counter

    err = clSetKernelArg(kernelCalculateIter1Begin, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Begin, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculateIter1Begin, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculateIter1Begin, 3, sizeof(int), &n_iter_min);
    err = clSetKernelArg(kernelCalculateIter1Begin, 4, sizeof(int), &n_iter_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 5, sizeof(double), &iter_diff_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 6, sizeof(double), &((*Fa).Alamda_start));

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateFinishPole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateFinishPole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateFinishPole, 2, sizeof(cl_mem), &CUDA_FR);
#pragma endregion
    //	}
        // catch (cl::Error& e)
        // {
        // 	cerr << "Error " << e.err() << " - " << e.what() << std::endl;
        // }

        //int firstreport = 0;//beta debug
    auto oldFractionDone = 0.0001;
    int count = 0;
    size_t local = BLOCK_DIM;
    size_t sLocal = 1;

    // freq_result* fres;

    for (n = n_start_from; n <= n_max; n += (int)CUDA_grid_dim)
    {
        auto fractionDone = (double)n / (double)n_max;

#ifndef INTEL
        // pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        // queue.flush();
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
        // queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        // queue.flush();
        clEnqueueWriteBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#endif
        err = clSetKernelArg(kernelCalculatePrepare, 6, sizeof(n), &n);
        err = EnqueueNDRangeKernel(queue, kernelCalculatePrepare, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
        if (getError(err)) return err;
        //clFinish(queue);

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
            err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);
            err = clSetKernelArg(kernelCalculatePreparePole, 6, sizeof(m), &m);
            err = EnqueueNDRangeKernel(queue, kernelCalculatePreparePole, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;
            //clFinish(queue);

            count = 0;

            while (!theEnd)
            {
                count++;
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Begin, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                //mrqcof
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                for (iC = 1; iC < l_curves; iC++)
                {
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 2, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);
                }

                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                // //mrqcof
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1End, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin1End, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                for (iC = 1; iC < l_curves; iC++)
                {
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 2, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 2, sizeof(in_rel[iC]), &(in_rel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 3, sizeof(l_points[iC]), &(l_points[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                    //clFinish(queue);
                }

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 2, sizeof(in_rel[l_curves]), &(in_rel[l_curves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 3, sizeof(l_points[l_curves]), &(l_points[l_curves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2End, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);
                //mrqcof

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin2End, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue);

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;
                //clFinish(queue); // ***

                err = clEnqueueReadBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);

                theEnd = theEnd == CUDA_grid_dim;
            }

            printf("."); fflush(stdout);
            err = EnqueueNDRangeKernel(queue, kernelCalculateFinishPole, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;
            //clFinish(queue);
        }

#if defined (INTEL)
        fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
        queue.finish();
#else
        // pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        // queue.flush();
        clEnqueueReadBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#endif
        //err=cudaThreadSynchronize(); memcpy is synchro itself

        //read results here
        //err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim_precalc, cudaMemcpyDeviceToHost);

        oldFractionDone = fractionDone;
        LinesWritten = 0;
#if defined (INTEL)
        auto res = (freq_result*)fres;
#else
        // auto res = (freq_result*)pfr;
        auto res = new freq_result[CUDA_grid_dim];
        memcpy(res, pfr, frSize);
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
        // queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        // queue.flush();
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

    clReleaseMemObject(CUDA_MCC2);
    clReleaseMemObject(CUDA_CC);
    clReleaseMemObject(CUDA_CC2);
    clReleaseMemObject(CUDA_End);
    clReleaseMemObject(CUDA_FR);
    clReleaseMemObject(cgFirst);

#if !defined _WIN32
#if defined INTEL
    free(pcc);
#elif defined AMD
    // free(memIn);
    // free(pcc);
    delete[] pcc;
    delete[] pfr;
    free(pFa);
#elif defined NVIDIA
    free(memIn);
    free(pcc);
    delete[] pcc;
#endif
#else // WIN
    //_aligned_free(pfr); // res does not need to be freed as it's just a pointer to *pfr.
#if defined(INTEL)
    _aligned_free(pcc);
#elif defined AMD
    _aligned_free(memFa);
    _aligned_free(memFb);
    delete[] pfr;
    //_aligned_free(memPcc);
    delete[] pcc;
    _aligned_free(Fa);
#elif defined NVIDIA
    delete[] pcc;
#endif
#endif // WIN


    return 0;
}

#endif
