#include "Start_OpenCl_amd_gpu.h"

freq_result* clAmdGpuStrategy::CreateFreqResult(size_t frSize) const
{
    // TODO: Use different allign function depending on compiler (GNUC/MSVC)
    freq_result* pfr = (freq_result*)_aligned_malloc(frSize, 128);

    return pfr;
}

cl_mem clAmdGpuStrategy::CreateBufferCL_FR(cl_context context, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    cl_mem CL_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &error);

    return CL_FR;
}

// TODO: Test if reflection wil do the trick here with 'pfr'
void clAmdGpuStrategy::EnqueueMapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    // GPU
    // Do nothing;
}

// TODO: Test if reflection wil do the trick here with 'pfr'
void clAmdGpuStrategy::EnqueueMapReadCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    error = clEnqueueReadBuffer(queue, CL_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
}

cl_int clAmdGpuStrategy::EnqueueMapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    error = clEnqueueWriteBuffer(queue, CL_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);

    return error;
}

cl_int clAmdGpuStrategy::EnqueueUnmapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int result = 0;
    //GPU
    // Do nothing

    return result;
}

cl_int clAmdGpuStrategy::EnqueueUnmapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int result = 0;
    result = clEnqueueWriteBuffer(queue, CL_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);

    return result;
}
