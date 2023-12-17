#include <string>
#include <memory>
#include <iostream>
#include <string>

#include <CL/cl.h>
#include "boinc_api.h"
#include "Globals_OpenCl.h"

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
protected:

public:
    //template <typename T>
    //T* CreateStruct(size_t size);

    virtual freq_result* CreateFreqResult(size_t size) const = 0;
    virtual freq_context* CreateFreqContext(size_t size) const = 0;
    virtual mfreq_context* CreateMFreqContext(size_t size) const = 0;
    virtual cl_mem CreateBufferCL(cl_context context, size_t size, void* ptr) const = 0;
    virtual void EnqueueMapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const = 0;
    virtual void EnqueueMapReadCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const = 0;
    virtual cl_int EnqueueMapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const = 0;
    virtual cl_int EnqueueUnmapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const = 0;
    virtual cl_int EnqueueUnmapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr) const = 0;

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

    //template <typename T>
    //T* CallCreateStruct(size_t size)
    //{
    //    if (clStrategy_)
    //    {
    //        T* ptr = clStrategy_->CreateStruct<T>(size);

    //        return ptr;
    //    }
    //    else
    //    {
    //        ReportError(__FUNCTION__);
    //        return nullptr;
    //    }
    //};

    freq_result* CallCreateFreqResult(size_t size)
    {
        if (clStrategy_)
        {
            freq_result* ptr = clStrategy_->CreateFreqResult(size);

            return ptr;
        }
        else
        {
            ReportError(__FUNCTION__);
            return nullptr;
        }
    }

    freq_context* CallCreateFreqContext(size_t size)
    {
        if (clStrategy_)
        {
            freq_context* ptr = clStrategy_->CreateFreqContext(size);

            return ptr;
        }
        else
        {
            ReportError(__FUNCTION__);
            return nullptr;
        }
    }

    mfreq_context* CallCreateMFreqContext(size_t size)
    {
        if (clStrategy_)
        {
            mfreq_context* ptr = clStrategy_->CreateMFreqContext(size);

            return ptr;
        }
        else
        {
            ReportError(__FUNCTION__);
            return nullptr;
        }
    }

    cl_mem CallCreateBufferCl(cl_context context, size_t size, void* ptr)
    {
        if (clStrategy_)
        {
            cl_mem clMem = clStrategy_->CreateBufferCL(context, size, ptr);

            return clMem;
        }
        else
        {
            ReportError(__FUNCTION__);
            return nullptr;
        }
    }

    void CallEnqueueMapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr)
    {
        if (clStrategy_)
        {
            clStrategy_->EnqueueMapCL(queue, clMem, size, ptr);
        }
        else
        {
            ReportError(__FUNCTION__);
        }
    }

    void CallEnqueueMapReadCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr)
    {
        if (clStrategy_)
        {
            clStrategy_->EnqueueMapReadCL(queue, clMem, size, ptr);
        }
        else
        {
            ReportError(__FUNCTION__);
        }
    }

    cl_int CallEnqueueMapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr)
    {
        if (clStrategy_)
        {
            cl_int result = 0;
            result = clStrategy_->EnqueueMapWriteCL(queue, clMem, size, ptr);

            return result;
        }
        else
        {
            ReportError(__FUNCTION__);
            return -999;
        }
    }

    cl_int CallEnqueueUnmapCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr)
    {
        if (clStrategy_)
        {
            cl_int result = 0;
            result = clStrategy_->EnqueueUnmapCL(queue, clMem, size, ptr);

            return result;
        }
        else
        {
            ReportError(__FUNCTION__);
            return -999;
        }

    }

    cl_int CallEnqueueUnmapWriteCL(cl_command_queue queue, cl_mem clMem, size_t size, void* ptr)
    {
        if (clStrategy_)
        {
            cl_int result = clStrategy_->EnqueueUnmapWriteCL(queue, clMem, size, ptr);

            return result;
        }
        else
        {
            ReportError(__FUNCTION__);
            return -999;
        }
    }
};
#endif
