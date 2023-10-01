/* helper.h - OpenCL helper macros
 * Copyright 2010 (c) Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 * Released under GNU LGPL.
 */

#ifndef __OPENCL_HELPER_MACROS__
#define __OPENCL_HELPER_MACROS__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#if defined _WIN32
//#include <cl_boinc.h>
#include <CL/cl.h>
#else
#include <CL/opencl.hpp>
#endif

#define CL_WRAPPER(FUNC) \
    { \
        cl_int err = FUNC; \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "Error %d executing %s on %s:%d (%s)\n", \
                err, #FUNC, __FILE__, __LINE__, cl_error_to_str(err)); \
            abort(); \
        }; \
    }

/* The following macro assumes the assignment will store the error code to err */
int err;  // error code returned from api calls
#define CL_ASSIGN(ASSIGNMENT) \
    { \
        ASSIGNMENT; \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "Error %d executing %s on %s:%d (%s)\n", \
                err, #ASSIGNMENT, __FILE__, __LINE__, cl_error_to_str(err)); \
            abort(); \
        }; \
    }


const char *cl_error_to_str(cl_int e)
{
    switch (e) {
        case CL_SUCCESS: return "success";
        case CL_DEVICE_NOT_FOUND: return "device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "device not available";
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
        case CL_COMPILER_NOT_AVAILABLE: return "device compiler not available";
#endif
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "mem object allocation failure";
        case CL_OUT_OF_RESOURCES: return "out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "profiling info not available";
        case CL_MEM_COPY_OVERLAP: return "mem copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "build program failure";
        case CL_MAP_FAILURE: return "map failure";

        case CL_INVALID_VALUE: return "invalid value";
        case CL_INVALID_DEVICE_TYPE: return "invalid device type";
        case CL_INVALID_PLATFORM: return "invalid platform";
        case CL_INVALID_DEVICE: return "invalid device";
        case CL_INVALID_CONTEXT: return "invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "invalid command queue";
        case CL_INVALID_HOST_PTR: return "invalid host ptr";
        case CL_INVALID_MEM_OBJECT: return "invalid mem object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "invalid image size";
        case CL_INVALID_SAMPLER: return "invalid sampler";
        case CL_INVALID_BINARY: return "invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "invalid build options";
        case CL_INVALID_PROGRAM: return "invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "invalid kernel definition";
        case CL_INVALID_KERNEL: return "invalid kernel";
        case CL_INVALID_ARG_INDEX: return "invalid arg index";
        case CL_INVALID_ARG_VALUE: return "invalid arg value";
        case CL_INVALID_ARG_SIZE: return "invalid arg size";
        case CL_INVALID_KERNEL_ARGS: return "invalid kernel args";
        case CL_INVALID_WORK_DIMENSION: return "invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "invalid event wait list";
        case CL_INVALID_EVENT: return "invalid event";
        case CL_INVALID_OPERATION: return "invalid operation";
        case CL_INVALID_GL_OBJECT: return "invalid gl object";
        case CL_INVALID_BUFFER_SIZE: return "invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "invalid mip level";
#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
        case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR: return "invalid gl sharegroup reference number";
#endif
#ifdef CL_VERSION_1_1
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "misaligned sub-buffer offset";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "exec status error for events in wait list";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "invalid global work size";
#endif
        default: return "invalid/unknown error code";
    }
}

#endif

bool getError(cl_int err)
{
    if(err != CL_SUCCESS)
    {
        std::cerr << "Error enqueueing kernel: " << cl_error_to_str(err) << " (" << err << ")" << std::endl;
        return true;
    }

    return false;
}

cl_int EnqueueNDRangeKernel(cl_command_queue command_queue,
                            cl_kernel kernel,
                            cl_uint work_dim,
                            const size_t* global_work_offset,
                            const size_t* global_work_size,
                            const size_t* local_work_size,
                            cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list,
                            cl_event* event)
{

#if defined DEBUG_LEVEL_5
    size_t nameSize;
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &nameSize);
    auto kernelName = new char[nameSize];
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, nameSize, kernelName, NULL);
    std::cerr << "Starting kernel [" << kernelName << "]... ";
#endif

    cl_int err = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    if (err != CL_SUCCESS)
    {
#if !defined DEBUG_LEVEL_5
        size_t nameSize;
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &nameSize);
        auto kernelName = new char[nameSize];
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, nameSize, kernelName, NULL);
#endif
        std::cerr << std::endl << "Error enqueueing kernel ["<< kernelName << "]: " << cl_error_to_str(err) << " (" << err << ")" << std::endl;
        return true;
    }

#if defined DEBUG_LEVEL_5
    std::cerr << "done." << std::endl;
#endif

    return false;
}
