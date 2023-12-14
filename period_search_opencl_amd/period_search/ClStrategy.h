
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

#include <string>
#include <memory>
#include <iostream>
#include <string>

#include <CL/cl.h>
//#include "opencl_helper.h"
#include "boinc_api.h"
#include "Globals_OpenCl.h"
//#include "ClStrategy.h"

#ifndef STRATEGY
#define STRATEGY
/**
 * The Strategy interface declares operations common to all supported versions
 * of some algorithm.
 *
 * The Context uses this interface to call the algorithm defined by Concrete
 * Strategies.
 */
class clStrategy
{
public:
    //clStrategy() {};

    virtual freq_result* CreateFreqResult(size_t frSize) const = 0;
    virtual mfreq_context* CreateFreqContext(size_t pccSize) const = 0;
    virtual cl_mem CreateBufferCL_FR(cl_context context, size_t frSize, void* pfr) const = 0;
    virtual void EnqueueMapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const = 0;
    virtual void EnqueueMapReadCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const = 0;
    virtual cl_int EnqueueMapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const = 0;
    virtual cl_int EnqueueUnmapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const = 0;
    virtual cl_int EnqueueUnmapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr) const = 0;

    virtual ~clStrategy() = default;
};
#endif


#ifndef STRATEGY_CONTEXT
#define STRATEGY_CONTEXT
/**
 * The Context defines the interface of interest to clients.
 */

class clStrategyContext
{
    /**
     * @var Strategy The Context maintains a reference to one of the Strategy
     * objects. The Context does not know the concrete class of a strategy. It
     * should work with all strategies via the Strategy interface.
     */
private:
    std::unique_ptr<clStrategy> clStrategy_;

    void ReportError(std::string function)
    {
        std::cerr << "Error in Context while calling " << function << "(): clStrategy isn't set" << std::endl;
    }

    /**
     * Usually, the Context accepts a strategy through the constructor, but also
     * provides a setter to change it at runtime.
     */
public:
    explicit clStrategyContext(std::unique_ptr<clStrategy>&& clStrategy = {}) : clStrategy_(std::move(clStrategy))
    {
    }
    /**
     * Usually, the Context allows replacing a Strategy object at runtime.
     */
    void SetClStrategy(std::unique_ptr<clStrategy>&& clStrategy)
    {
        clStrategy_ = std::move(clStrategy);
    }
    /**
     * The Context delegates some work to the Strategy object instead of
     * implementing +multiple versions of the algorithm on its own.
     */
    freq_result* CallCreateFreqResult(size_t frSize)
    {
        if (clStrategy_)
        {
            freq_result* pfr = clStrategy_->CreateFreqResult(frSize);

            return pfr;
        }
        else
        {
            ReportError(__FUNCTION__);
            return nullptr;
        }

    }

    mfreq_context* CallCreateFreqContext(size_t pccSize)
    {
        if (clStrategy_)
        {
            mfreq_context* pcc = clStrategy_->CreateFreqContext(pccSize);

            return pcc;
        }
        else
        {
            ReportError(__FUNCTION__);
            return nullptr;
        }

    }

    cl_mem CallCreateBufferCl_FR(cl_context context, size_t frSize, void* pfr)
    {
        if (clStrategy_)
        {
            cl_mem resutl = clStrategy_->CreateBufferCL_FR(context, frSize, pfr);

            return resutl;
        }
        else
        {
            ReportError(__FUNCTION__);
            return nullptr;
        }
    }

    void CallEnqueueMapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr)
    {
        if (clStrategy_)
        {
            clStrategy_->EnqueueMapCL_FR(queue, CL_FR, frSize, pfr);
        }
        else
        {
            ReportError(__FUNCTION__);
        }
    }

    void CallEnqueueMapReadCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr)
    {
        if (clStrategy_)
        {
            clStrategy_->EnqueueMapReadCL_FR(queue, CL_FR, frSize, pfr);
        }
        else
        {
            ReportError(__FUNCTION__);
        }
    }

    cl_int CallEnqueueMapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr)
    {
        if (clStrategy_)
        {
            cl_int result = 0;
            result = clStrategy_->EnqueueMapWriteCL_FR(queue, CL_FR, frSize, pfr);

            return result;
        }
        else
        {
            ReportError(__FUNCTION__);
            return 1;
        }
    }

    cl_int CallEnqueueUnmapCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr)
    {
        if (clStrategy_)
        {
            cl_int result = 0;
            result = clStrategy_->EnqueueUnmapCL_FR(queue, CL_FR, frSize, pfr);

            return result;
        }
        else
        {
            ReportError(__FUNCTION__);
            return 1;
        }

    }

    cl_int CallEnqueueUnmapWriteCL_FR(cl_command_queue queue, cl_mem CL_FR, size_t frSize, void* pfr)
    {
        if (clStrategy_)
        {
            cl_int result = clStrategy_->EnqueueUnmapWriteCL_FR(queue, CL_FR, frSize, pfr);

            return result;
        }
        else
        {
            ReportError(__FUNCTION__);
            return 1;
        }
    }
};
#endif
