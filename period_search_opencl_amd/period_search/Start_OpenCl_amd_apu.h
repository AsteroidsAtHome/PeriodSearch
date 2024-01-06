#ifndef AMDAPU
#define AMDAPU

/*
    This strategy class contains AMD's APU (AMD Accelerated Processing Unit) platform specific dedicated fucntions.
    Covered platforms:
        "Advanced Micro Devices, Inc."
        "Mesa"
*/

#include <CL/cl.h>

#ifdef _WIN32
#include "boinc_win.h"
#else
#endif

#include "ClStrategy.h"

class clAmdApuStrategy : public clStrategy
{
public:
    clAmdApuStrategy() {};
    ~clAmdApuStrategy() = default;

    freq_result* CreateFreqResult(size_t size) const;

    freq_context* CreateFreqContext(size_t size) const;

    mfreq_context* CreateMFreqContext(size_t size) const;

    cl_mem CreateBufferCL(cl_context context, size_t size, void* ptr) const;

    // TODO: Test if reflection wil do the trick here with 'ptr'
    void EnqueueMapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const;

    // TODO: Test if reflection wil do the trick here with 'ptr'
    void EnqueueMapReadCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const;

    cl_int EnqueueMapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const;

    cl_int EnqueueUnmapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const;

    cl_int EnqueueUnmapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const;

//    template<typename T>
//    T* CreateStruct(size_t size) const
//    {
//#if !defined __GNUC__ && defined _WIN32
//        T* ptr = (T*)_aligned_malloc(size, 128);
//#elif defined __GNUC__
//        T* ptr = (T*)aligned_alloc(128, size);
//#endif
//
//        return ptr;
//    }

};

#endif
