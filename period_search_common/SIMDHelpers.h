#pragma once

#include <memory>
#include <cstdlib>
#include <stdlib.h>
#include <stdexcept>
#include <new> // for placement new

/**
 * @brief A custom aligned allocator to ensure proper alignment during dynamic allocation.
 *
 * This template struct provides an allocator that ensures memory alignment for objects of type `T`.
 * It uses `_aligned_malloc` and `_aligned_free` for allocation and deallocation, respectively.
 *
 * @tparam T The type of objects to be allocated.
 */
template <typename T>
struct AlignedAllocator
{
    using value_type = T;
    std::size_t alignment;

    //AlignedAllocator() = default;
    explicit AlignedAllocator(std::size_t align = alignof(T)) noexcept : alignment(align) {}

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U>& other) noexcept : alignment(other.alignment) {}

    T* allocate(const std::size_t n)
    {
#if defined __GNUC__ && !defined _WIN32
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#else
        void* ptr = _aligned_malloc(n * sizeof(T), alignment);
#endif
        if (!ptr)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept
    {
#if defined __GNUC__ && !defined _WIN32
        free(p);
#else
        _aligned_free(p);
#endif
    }
};

template <typename T>
struct AlignedDeleter
{
    void operator()(T* ptr) const
    {
        ptr->~T();			// Explicitly call the destructor
#if defined __GNUC__ && !defined _WIN32
        free(ptr);
#else
        _aligned_free(ptr);   // Free the aligned memory
#endif
    }
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return true; }

template <typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return false; }

/**
 * @brief Creates a shared pointer to an object with aligned memory allocation.
 *
 * This function template allocates aligned memory and constructs an object of type `T` using placement new.
 * It returns a `std::shared_ptr` to the newly created object, ensuring proper alignment and memory management.
 *
 * @tparam T The type of the object to be created.
 * @param alignment The alignment requirement for the memory allocation.
 * @return Returns a `std::shared_ptr<T>` to the newly created and aligned object.
 *
 * @note The function uses `AlignedAllocator` for memory allocation and `AlignedDeleter` for memory deallocation.
 */
template <typename T>
std::shared_ptr<T> CreateAlignedShared(std::size_t alignment)
{
    // Allocate aligned memory and construct the object using placement new
    auto ptr = std::shared_ptr<T>(new (AlignedAllocator<T>(alignment).allocate(1)) T(), AlignedDeleter<T>());
    return ptr;
}

// ***************************



