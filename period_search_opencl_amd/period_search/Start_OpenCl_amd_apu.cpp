#include "Start_OpenCl_amd_apu.h"

//clAmdApuStrategy::clAmdApuStrategy(){}

freq_result* clAmdApuStrategy::CreateFreqResult(size_t frSize) const
{
    freq_result* pfr = (freq_result*)_aligned_malloc(frSize, 128);

    return pfr;
}

cl_mem clAmdApuStrategy::CreateBufferCL_FR(cl_context context, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    //APU
    //cl_mem CLFR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, pfr, &error);

    //GPU
    cl_mem CL_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &error);

    return CL_FR;
}

// TODO: Test if reflection wil do the trick here with 'pfr'
void clAmdApuStrategy::EnqueueMapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    pfr = clEnqueueMapBuffer(queue, CL_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, 0, NULL, NULL, &error);
    //return pfr;
}

// TODO: Test if reflection wil do the trick here with 'pfr'
void clAmdApuStrategy::EnqueueMapReadCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    pfr = clEnqueueMapBuffer(queue, CL_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, 0, NULL, NULL, &error);
    clFlush(queue);
}

cl_int clAmdApuStrategy::EnqueueMapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int error = 0;
    //APU
    //pfr = clEnqueueMapBuffer(queue, CL_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, 0, NULL, NULL, &error);
    //clFlush(queue);

    //GPU
    error = clEnqueueWriteBuffer(queue, CL_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);

    return error;
}

cl_int clAmdApuStrategy::EnqueueUnmapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int result = clEnqueueUnmapMemObject(queue, CL_FR, pfr, 0, NULL, NULL);
    clFlush(queue);

    return result;
}

cl_int clAmdApuStrategy::EnqueueUnmapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const
{
    cl_int result = clEnqueueUnmapMemObject(queue, CL_FR, pfr, 0, NULL, NULL);
    clFlush(queue);

    return result;
}
