#include "Start_OpenCl_amd_apu.h"

// TODO: Use different allign function depending on compiler (GNUC/MSVC). Add proper deallocator functions.
// TODO: Change alignment boundary according to AMD APU's specifications. (4096?)
// TODO: Use Template function.

freq_result* clAmdApuStrategy::CreateFreqResult(size_t size) const
{
#if !defined __GNUC__ && defined _WIN32
    freq_result* ptr = (freq_result*)_aligned_malloc(size, 4096);
#elif defined __GNUC__
    freq_result* ptr = (mfreq_context*)aligned_alloc(4096, size);
#endif

    return ptr;
}

freq_context* clAmdApuStrategy::CreateFreqContext(size_t size) const
{
#if !defined __GNUC__ && defined _WIN32
    freq_context* ptr = (freq_context*)_aligned_malloc(size, 4096);
#elif defined __GNUC__
    freq_context* ptr = (freq_context*)aligned_alloc(4096, size);
#endif

    return ptr;
}

mfreq_context* clAmdApuStrategy::CreateMFreqContext(size_t size) const
{
#if !defined __GNUC__ && defined _WIN32
    mfreq_context* ptr = (mfreq_context*)_aligned_malloc(size, 4096);
#elif defined __GNUC__
    mfreq_context* ptr = (mfreq_context*)aligned_alloc(4096, size);
#endif

    return ptr;
}

cl_mem clAmdApuStrategy::CreateBufferCL(cl_context context, size_t size, void* ptr) const
{
    cl_int error = 0;
    cl_mem clMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, ptr, &error);

    return clMem;
}

// TODO: Test if reflection wil do the trick here with 'ptr'
void clAmdApuStrategy::EnqueueMapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int error = 0;
    ptr = clEnqueueMapBuffer(queue, clMem, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
}

// TODO: Test if reflection wil do the trick here with 'ptr'
void clAmdApuStrategy::EnqueueMapReadCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int error = 0;
    ptr = clEnqueueMapBuffer(queue, clMem, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    clFlush(queue);
}

cl_int clAmdApuStrategy::EnqueueMapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int error = 0;
    ptr = clEnqueueMapBuffer(queue, clMem, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    clFlush(queue);

    return error;
}

cl_int clAmdApuStrategy::EnqueueUnmapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int result = clEnqueueUnmapMemObject(queue, clMem, ptr, 0, NULL, NULL);
    clFlush(queue);

    return result;
}

cl_int clAmdApuStrategy::EnqueueUnmapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const
{
    cl_int result = clEnqueueUnmapMemObject(queue, clMem, ptr, 0, NULL, NULL);
    clFlush(queue);

    return result;
}
