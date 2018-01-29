#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include "globals.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "../UnitTest_opencl_amd_win/declarations.h"
using std::cout;
using std::endl;
using std::cerr;
using std::string;

cl_double xP;

// Helper function to print vector elements
void printVector(const string arrayName,
    const double *arrayData,
    const unsigned int length)
{
    int numElementsToPrint = (256 < length) ? 256 : length;
    cout << endl << arrayName << ":" << endl;
    for (int i = 0; i < numElementsToPrint; ++i)
        cout << arrayData[i] << " ";
    cout << endl;
}

void prepareCurvCl()
{
    cl::Program::Sources sources(1, std::make_pair(kernelCurv.c_str(), kernelCurv.length()));
    program = cl::Program(context, sources);
    program.build(devices);
    kernel = cl::Kernel(program, "curv");
}

void prepareDaveCl()
{
    cl::Program::Sources sources(1, std::make_pair(kernelDave.c_str(), kernelDave.length()));
    program = cl::Program(context, sources);
    program.build(devices);
    kernel = cl::Kernel(program, "dave");
}

void Init()
{
    try {
        // Allocate and initialize memory on the host
        //initHost();

        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator iter;
        for (iter = platforms.begin(); iter != platforms.end(); ++iter)
        {
            auto name = (*iter).getInfo<CL_PLATFORM_NAME>();
            auto vendor = (*iter).getInfo<CL_PLATFORM_VENDOR>();
            cout << "Platform vendor: " << vendor << endl;
            if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                "Advanced Micro Devices, Inc."))
            {
                break;
            }
            if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                "NVIDIA Corporation"))
            {
                break;
            }
            /*if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(),
            "Intel(R) Corporation"))
            {
            break;
            }*/
        }

        // Create an OpenCL context
        cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0 };
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

        // Detect OpenCL devices
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        auto deviceName = devices[0].getInfo<CL_DEVICE_NAME>();
        auto deviceMaxWorkItemDims = devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
        cout << "Device name: " << deviceName << endl;
        cout << "Max work item dimensions: " << deviceMaxWorkItemDims << endl;

        // Load CL file, build CL program object, create CL kernel object
        std::ifstream f("../../period_search/curv.cl");
        std::stringstream st;
        st << f.rdbuf();
        kernelCurv = st.str();

        std::ifstream f1("../../period_search/dave.cl");
        std::stringstream st1;
        st1 << f.rdbuf();
        kernelDave = st1.str();
        /*const char* source = curv.c_str();
        const size_t len = curv.length();*/

        // Create an OpenCL command queue
        queue = cl::CommandQueue(context, devices[0]);

        prepareCurvCl();
    }
    catch (cl::Error err)
    {
        // Catch OpenCL errors and print log if it is a build error
        cerr << "ERROR: " << err.what() << "(" << err.err() << ")" <<
            endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            string str =
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            cout << "Program Info: " << str << endl;
        }
        //cleanupHost();
    }
    catch (string msg)
    {
        cerr << "Exception caught in main(): " << msg << endl;
        //cleanupHost();
    }
}

double *prepareFc(int maxI, int maxJ)
{
    int len = maxI * maxJ;
    auto *_Fc = vector_double(len);
    int k = 0;
    for (int j = 0; j < maxJ; j++) {
        for (int i = 0; i < maxI; i++)
        {
            _Fc[k++] = Fc[i][j];
        }
    }
    return _Fc;
}

double *prepareFs(int maxI, int maxJ)
{
    int len = maxI * maxJ;
    auto *_Fs = vector_double(len);
    int k = 0;
    for (int j = 0; j < maxJ; j++) {
        for (int i = 0; i < maxI; i++)
        {
            _Fs[k++] = Fs[i][j];
        }
    }
    return _Fs;
}

double *preparePleg(int maxI, int maxJ, int maxK)
{
    int len = maxI * maxJ;
    auto *_Pleg = vector_double(len);
    int k = 0;
    for (int k = 0; k < maxK; k++) {
        for (int j = 0; j < maxJ; j++) {
            for (int i = 0; i < maxI; i++)
            {
                _Pleg[k++] = Pleg[i][j][k];
            }
        }
    }
    return _Pleg;
}

double *prepareDg(int maxI, int maxJ)
{
    int len = maxI * maxJ;
    auto *_Dg = vector_double(len);
    int k = 0;
    for (int j = 0; j < maxJ; j++) {
        for (int i = 0; i < maxI; i++)
        {
            _Dg[k++] = Dg[i][j];
        }
    }
    return _Dg;
}

double *prepareDsph(int maxI, int maxJ)
{
    int len = maxI * maxJ;
    auto *_Dsph = vector_double(len);
    int k = 0;
    for (int j = 0; j < maxJ; j++) {
        for (int i = 0; i < maxI; i++)
        {
            _Dsph[k++] = Dsph[i][j];
        }
    }
    return _Dsph;
}

void curvCl(double cg[]) // , double Fc[][MAX_LM + 1], double Fs[][MAX_LM + 1], double Dsph[][MAX_N_PAR + 1], double Dg[][MAX_N_PAR + 1])
{
    try 
    {
        bufCg = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * MAX_N_PAR, cg);
        bufArea = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * Numfac, Area);
        int lenDarea = MAX_N_FAC + 1;
        /*double *_Darea = vector_double(lenDarea);
        for (int i = 0; i < lenDarea; i++) _Darea[i] = Darea[i];*/
        bufDarea = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * lenDarea, Darea);

        int lenFc = (MAX_N_FAC + 1) * (MAX_LM + 1);
        bufFc = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * lenFc, prepareFc(MAX_N_FAC + 1, MAX_LM + 1));
        int lenFs = (MAX_N_FAC + 1) * (MAX_LM + 1);
        bufFs = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * lenFs, prepareFs(MAX_N_FAC + 1, MAX_LM + 1));
        int lenPleg = (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1);
        bufPleg = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * lenPleg, preparePleg(MAX_N_FAC + 1, MAX_LM + 1, MAX_LM + 1));
        int lenDg = (MAX_N_FAC + 1) * (MAX_N_PAR + 1);
        bufDg = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * lenDg, prepareDg(MAX_N_FAC + 1, MAX_N_PAR + 1));
        int lenDsph = (MAX_N_FAC + 1) * (MAX_N_PAR + 1);
        bufDsph = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * lenDsph, prepareDsph(MAX_N_FAC + 1, MAX_N_PAR + 1));
        
        // Set the arguments that will be used for kernel execution
        kernel.setArg(0, bufCg);
        kernel.setArg(1, bufArea);
        kernel.setArg(2, bufDarea);
        kernel.setArg(3, bufFc);
        kernel.setArg(4, MAX_N_FAC + 1); 
        kernel.setArg(5, MAX_LM + 1);
        kernel.setArg(6, bufFs);
        kernel.setArg(7, MAX_N_FAC + 1);
        kernel.setArg(8, MAX_LM + 1);
        kernel.setArg(9, bufPleg);
        kernel.setArg(10, MAX_N_FAC + 1);
        kernel.setArg(11, MAX_LM + 1);
        kernel.setArg(12, MAX_LM + 1);
        kernel.setArg(13, Mmax);
        kernel.setArg(14, Lmax);
        kernel.setArg(15, bufDg);
        kernel.setArg(16, MAX_N_FAC + 1);
        kernel.setArg(17, MAX_N_PAR + 1);
        kernel.setArg(18, bufDsph);
        kernel.setArg(19, MAX_N_FAC + 1);
        kernel.setArg(20, MAX_N_PAR + 1);
        /*kernel.setArg(0, bufFs);
        kernel.setArg(0, bufDsph);
        kernel.setArg(0, bufDg);
        kernel.setArg(0, bufPleg);
        */

        auto maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        int rangeLocal = Numfac <= maxWorkGroupSize ? Numfac : maxWorkGroupSize;

        // Enqueue the kernel to the queue
        // with appropriate global and local work sizes
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(Numfac), cl::NDRange(rangeLocal));

        // Enqueue blocking call to read back buffer Y
        queue.enqueueReadBuffer(bufArea, CL_TRUE, 0, Numfac * sizeof(cl_float), tmpArea);
        //printVector("Area", Area, Numfac);
    }
    catch (cl::Error err)
    {
        // Catch OpenCL errors and print log if it is a build error
        cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        cout << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            string str =
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            cout << "Program Info: " << str << endl;
        }
        //cleanupHost();
    }
    catch (string msg)
    {
        cerr << "Exception caught in main(): " << msg << endl;
        //cleanupHost();
    }
}

void daveCl(double *dave, double *dyda)
{
    cl_double xx;
    try
    {
        bufDave = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * MAX_N_PAR, dave);
    }
    catch (cl::Error err)
    {
        // Catch OpenCL errors and print log if it is a build error
        cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        cout << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            string str =
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            cout << "Program Info: " << str << endl;
        }
        //cleanupHost();
    }
    catch (string msg)
    {
        cerr << "Exception caught in main(): " << msg << endl;
        //cleanupHost();
    }
}
