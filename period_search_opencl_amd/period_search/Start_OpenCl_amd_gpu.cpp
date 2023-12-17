#include "Start_OpenCl_amd_gpu.h"

// TODO: Use different allign function depending on compiler (GNUC/MSVC). Add proper deallocator functions.
//
// TODO: Use single Template function.

freq_result* clAmdGpuStrategy::CreateFreqResult(size_t size) const
{
#if !defined __GNUC__ && defined _WIN32
    freq_result* ptr = (freq_result*)_aligned_malloc(size, 128);
#elif defined __GNUC__
    freq_result* ptr = (mfreq_context*)aligned_alloc(128, size);
#endif

    return ptr;
}

freq_context* clAmdGpuStrategy::CreateFreqContext(size_t size) const
{
#if !defined __GNUC__ && defined _WIN32
    freq_context* ptr = (freq_context*)_aligned_malloc(size, 128);
#elif defined __GNUC__
    freq_context* ptr = (freq_context*)aligned_alloc(128, size);
#endif

    return ptr;
}

mfreq_context* clAmdGpuStrategy::CreateMFreqContext(size_t size) const
{
#if !defined __GNUC__ && defined _WIN32
    mfreq_context* ptr = (mfreq_context*)_aligned_malloc(size, 128);
#elif defined __GNUC__
    mfreq_context* ptr = (mfreq_context*)aligned_alloc(128, size);
#endif

    return ptr;
}

cl_mem clAmdGpuStrategy::CreateBufferCL(cl_context context, size_t size, void* ptr) const
{
    cl_int error = 0;
    cl_mem clMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, ptr, &error);

    return clMem;
}

// TODO: Test if reflection wil do the trick here with 'pfr'
void clAmdGpuStrategy::EnqueueMapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int error = 0;
    // GPU
    // Do nothing;
}

// TODO: Test if reflection wil do the trick here with 'pfr'
void clAmdGpuStrategy::EnqueueMapReadCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int error = 0;
    error = clEnqueueReadBuffer(queue, clMem, CL_BLOCKING, 0, size, ptr, 0, NULL, NULL);
}

cl_int clAmdGpuStrategy::EnqueueMapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int error = 0;
    error = clEnqueueWriteBuffer(queue, clMem, CL_BLOCKING, 0, size, ptr, 0, NULL, NULL);

    return error;
}

cl_int clAmdGpuStrategy::EnqueueUnmapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int result = 0;
    //GPU
    // Do nothing

    return result;
}

cl_int clAmdGpuStrategy::EnqueueUnmapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int result = 0;
    result = clEnqueueWriteBuffer(queue, clMem, CL_BLOCKING, 0, size, ptr, 0, NULL, NULL);

    return result;
}
