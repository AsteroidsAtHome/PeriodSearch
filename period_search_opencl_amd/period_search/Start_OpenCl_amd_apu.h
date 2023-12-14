//#pragma once
#ifndef AMDAPU
#define AMDAPU

/*
    This strategy class contains AMD's APU (AMD Accelerated Processing Unit) platform specific dedicated fucntions.
    Covered platforms:
        "Advanced Micro Devices, Inc."
        "Mesa"
*/

//#if !defined INTEL

//#if !defined _WIN32
//#define CL_TARGET_OPENCL_VERSION 110
//#define CL_HPP_MINIMUM_OPENCL_VERSION 110
//#define CL_HPP_TARGET_OPENCL_VERSION 110
//#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
//#define CL_HPP_CL_1_1_DEFAULT_BUILD
//// #define CL_API_SUFFIX__VERSION_1_0 CL_API_SUFFIX_COMMON
//#define CL_BLOCKING 	CL_TRUE
//#else // WIN32
//#define CL_TARGET_OPENCL_VERSION 120
//// #define CL_HPP_ENABLE_EXCEPTIONS
//#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
//#define CL_HPP_CL_1_2_DEFAULT_BUILD
//#define CL_HPP_ENABLE_EXCEPTIONS
//typedef unsigned int uint;
//#endif

// #include <CL/opencl.hpp>
#include <CL/cl.h>
//#include "opencl_helper.h"

//#include <cmath>
//#include <stdlib.h>
//#include <cstdio>
//#include <cstdlib>
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <array>
//#include <algorithm>
//#include <ctime>
//#include "boinc_api.h"
//#include "mfile.h"
//
//#include "globals.h"
//#include "constants.h"
//#include "declarations.hpp"
//#include "Start_OpenCl.h"
//#include "kernels.cpp"


#ifdef _WIN32
#include "boinc_win.h"
//#include <Shlwapi.h>
#else
#endif

//#include "Globals_OpenCl.h"
#include "ClStrategy.h"

class clAmdApuStrategy : public clStrategy
{
public:
    clAmdApuStrategy() {};
    ~clAmdApuStrategy() = default;

    //virtual
    freq_result* CreateFreqResult(size_t frSize) const;

    mfreq_context* CreateFreqContext(size_t pccSize) const;

    cl_mem CreateBufferCL_FR(cl_context context, size_t frSize, void* pfr) const;

    // TODO: Test if reflection wil do the trick here with 'pfr'
    void EnqueueMapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const;

    // TODO: Test if reflection wil do the trick here with 'pfr'
    void EnqueueMapReadCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const;

    cl_int clAmdApuStrategy::EnqueueMapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const;

    cl_int clAmdApuStrategy::EnqueueUnmapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const;

    cl_int clAmdApuStrategy::EnqueueUnmapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const;
};

#endif
