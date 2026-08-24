#include <string>
#include <cstring>
#include <iostream>
#include <array>
#include <cstdint>

#include "Enums.h"
#include "declarations.h"
#include "globals.h"

#if !defined _VC140_XP
#include "CalcStrategyAvx512.hpp"
#endif

#include "CalcStrategyFma.hpp"
#include "CalcStrategyAvx.hpp"
#include "CalcStrategySse3.hpp"
#include "CalcStrategySse2.hpp"
#include "CalcStrategyNone.hpp"
#include "SIMDHelpers.h"

#if !defined __GNUC__ && defined _WIN32 // !ARM
#include <intrin.h>
#elif defined __GNUC__
#include <x86intrin.h>
#endif

#if !defined __GNUC__ && defined _WIN32
#define cpuid(info, x) __cpuidex(info, x, 0)
#elif defined(__GNUC__)
#include <cpuid.h>
#define cpuid(info, x) __cpuid_count(x, 0, (info)[0], (info)[1], (info)[2], (info)[3])
#endif

unsigned long long xgetbv(unsigned long ctr)
{
#if !defined __GNUC__ && defined _WIN32
    return _xgetbv(ctr);
#elif defined(__GNUC__)
    uint32_t a = 0;
    uint32_t d;
    __asm("xgetbv"
    : "=a"(a), "=d"(d)
        : "c"(ctr)
        : );
    return a | (((uint64_t)(d)) << 32);
#endif
}

#if !defined __GNUC__ && defined _WIN32
/**
 * @brief Retrieves the CPU information string for Windows systems.
 *
 * This function uses the `__cpuid` instruction to query the CPU information, including
 * manufacturer, model, and clockspeed, and returns it as a concatenated string.
 *
 * @return Returns a string containing the CPU information.
 *
 * @note This function is designed for Windows systems and uses the `__cpuid` intrinsic.
 *
 * @see https://learn.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=vs-2019
 */
std::string GetCpuInfo()
{
    std::array<int, 4> integerBuffer = {};
    constexpr size_t sizeofIntegerBuffer = sizeof(int) * integerBuffer.size();
    std::array<char, 64> charBuffer = {};

    constexpr std::array<int, 3> functionIds = {
        // Manufacturer
        //  EX: "Intel(R) Core(TM"
        static_cast<int>(0x8000'0002),
        // Model
        //  EX: ") i7-8700K CPU @"
        static_cast<int>(0x8000'0003),
        // Clockspeed
        //  EX: " 3.70GHz"
       static_cast<int>(0x8000'0004) };

    std::string cpu;

    for (int id : functionIds)
    {
        // Get the data for the current ID.
        __cpuid(integerBuffer.data(), id);
        std::memcpy(charBuffer.data(), integerBuffer.data(), sizeofIntegerBuffer);
        cpu += std::string(charBuffer.data());
    }

    return cpu;
}

#elif defined __GNUC__
/**
 * @brief Retrieves the CPU information string for Unix-like operating systems.
 *
 * This function uses the `__cpuid` instruction to query the CPU information, including
 * manufacturer, model, and clockspeed, and returns it as a concatenated string.
 *
 * @return Returns a string containing the CPU information.
 *
 * @note This function is designed for Unix-like operating systems and uses the `__cpuid` intrinsic.
 */
std::string GetCpuInfo()
{
    char CPUBrandString[0x40];
    unsigned int CPUInfo[4] = { 0, 0, 0, 0 };

    __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    unsigned int nExIds = CPUInfo[0];

    memset(CPUBrandString, 0, sizeof(CPUBrandString));

    for (unsigned int i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);

        if (i == 0x80000002)
            std::memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            std::memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            std::memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }

    std::string cpu(CPUBrandString);

    return cpu;
}
#endif

/**
 * @brief Retrieves CPU information based on the specified info type.
 *
 * This function uses the `cpuid` instruction to query CPU information and stores
 * the results in the provided references for registers `a`, `b`, `c`, and `d`.
 *
 * @param info_type An unsigned integer specifying the type of information to query.
 * @param a A reference to an unsigned integer to store the value of the `EAX` register.
 * @param b A reference to an unsigned integer to store the value of the `EBX` register.
 * @param c A reference to an unsigned integer to store the value of the `ECX` register.
 * @param d A reference to an unsigned integer to store the value of the `EDX` register.
 *
 * @note The function uses the `cpuid` instruction to obtain CPU information.
 */
static void GetCpuid(unsigned int info_type, unsigned int& a, unsigned int& b, unsigned int& c, unsigned int& d)
{
    int CPUInfo[4] = { 0, 0, 0, 0 };

    cpuid(CPUInfo, info_type);

    a = CPUInfo[0];
    b = CPUInfo[1];
    c = CPUInfo[2];
    d = CPUInfo[3];

    return;
}

#ifndef _XCR_XFEATURE_ENABLED_MASK
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

static bool IsAVXSupportedByOS(unsigned int avx_mask)
{
    unsigned int a, b, c, d;
    GetCpuid(1, a, b, c, d);

    bool osUsesXSAVE_XRSTORE = c & (1 << 27);

    return osUsesXSAVE_XRSTORE && (xgetbv(_XCR_XFEATURE_ENABLED_MASK) & avx_mask);
}

static bool IsAVXSupportedByOS() {
    return IsAVXSupportedByOS(0x6);
}

static bool IsAVX512SupportedByOS() {
    return IsAVXSupportedByOS(0xe6);
}

/**
 * @brief Checks if the CPU belongs to the Bulldozer family.
 *
 * This function determines if the CPU is part of the Bulldozer family by querying the
 * CPU vendor and family information using the `cpuid` instruction.
 *
 * @return Returns true if the CPU is from the Bulldozer family, false otherwise.
 *
 * @note The Bulldozer CPU family does technically support AVX/FMA, but its performance is worse compared to SSE3.
 */
static bool IsBulldozer()
{
    unsigned int a, b, c, d;

    GetCpuid(0, a, b, c, d);
    char vendor[13];
    std::memcpy(vendor + 0, &b, 4);
    std::memcpy(vendor + 4, &d, 4);
    std::memcpy(vendor + 8, &c, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "AuthenticAMD") != 0) {
        return 0;
    }

    GetCpuid(1, a, b, c, d);

    uint32_t family = (a >> 8) & 0xf;
    uint32_t extended_family = (a >> 20) & 0xff;

    if (family == 0xf) {
        family += extended_family;
    }

    if (family == 0x15) {
        return 1;
    }
    else {
        return 0;
    }
}

/**
 * @brief Checks if the CPU belongs to the Bulldozer family.
 *
 * This function determines if the CPU is part of the Bulldozer family by querying the
 * CPU vendor and family information using the `cpuid` instruction.
 *
 * @return Returns true if the CPU is from the Bulldozer family, false otherwise.
 *
 * @note The Bulldozer CPU family does technically support AVX/FMA, but its performance is worse compared to SSE3.
 */
void GetSupportedSIMDs()
{
    unsigned int std_eax = 0, std_ebx = 0, std_ecx = 0, std_edx = 0;
    unsigned int struc_eax = 0, struc_ebx = 0, struc_ecx = 0, struc_edx = 0;
    unsigned int std_supported = 0, struc_ext_supported = 0;

    GetCpuid(0x00000000, struc_eax, struc_ebx, struc_ecx, struc_edx);
    if (struc_eax >= 0x00000007)
    {
        struc_ext_supported = 1;
        GetCpuid(0x00000007, struc_eax, struc_ebx, struc_ecx, struc_edx);
    }

    GetCpuid(0x00000000, std_eax, std_ebx, std_ecx, std_edx);
    if (std_eax >= 0x00000001)
    {
        std_supported = 1;
        GetCpuid(0x00000001, std_eax, std_ebx, std_ecx, std_edx);
    }

    CPUopt.hasSSE2 = std_supported && (std_edx & (1 << 26));
    CPUopt.hasSSE3 = std_supported && ((std_ecx & (1 << 0)) || (std_ecx & (1 << 9)));

#if !defined _VC140_XP
    CPUopt.hasAVX = std_supported && IsAVXSupportedByOS() && (std_ecx & (1 << 28));

    if (CPUopt.hasAVX)
    {
        CPUopt.hasFMA = std_supported && (std_ecx & (1 << 12));
    }
    if (struc_ext_supported && IsAVX512SupportedByOS())
    {
        CPUopt.hasAVX512 = struc_ebx & (1 << 16);
        CPUopt.hasAVX512dq = struc_ebx & (1 << 17);
    }
    CPUopt.isBulldozer = IsBulldozer();
#endif
}

/**
 * @brief Checks if a manually overridden optimization is supported and returns the closest supported fallback if not.
 *
 * This function checks the provided SIMDEnum value against the supported SIMD instructions
 * on the CPU and returns the closest supported optimization as a fallback if the provided value is not supported.
 *
 * @param simd The SIMDEnum value representing the manually overridden optimization to check.
 * @return Returns the closest supported SIMDEnum value if the provided value is not supported.
 */
SIMDEnum CheckSupportedSIMDs(SIMDEnum simd)
{
    SIMDEnum tempSimd = simd;
    // NOTE: As there is no pattern matching implemented yet in C++ we'll go with the ugly nested IF statements - GVidinski 29.01.2024
    if (simd == SIMDEnum::OptAVX512)
    {
        simd = CPUopt.hasAVX512 && CPUopt.hasAVX512dq
            ? SIMDEnum::OptAVX512
            : SIMDEnum::OptFMA;
    }

    if (simd == SIMDEnum::OptFMA)
    {
        simd = CPUopt.hasFMA
            ? SIMDEnum::OptFMA
            : SIMDEnum::OptAVX;
    }

    if (simd == SIMDEnum::OptAVX)
    {
        simd = CPUopt.hasAVX
            ? SIMDEnum::OptAVX
            : SIMDEnum::OptSSE3;
    }

    if (simd == SIMDEnum::OptSSE3)
    {
        simd = CPUopt.hasSSE3
            ? SIMDEnum::OptSSE3
            : SIMDEnum::OptSSE2;
    }

    if (simd == SIMDEnum::OptSSE2)
    {
        simd = CPUopt.hasSSE2
            ? SIMDEnum::OptSSE2
            : SIMDEnum::OptNONE;
    }

    if (simd == SIMDEnum::OptASIMD)
    {
        simd = SIMDEnum::OptNONE;
    }

    if (tempSimd != simd)
    {
        std::cerr << "Choosen optimization " << getSIMDEnumName(tempSimd) << " is not supported. Switching to " << getSIMDEnumName(simd) << "." << std::endl;
    }

    return simd;
}

/**
 * @brief Determines the best supported SIMD optimization for the CPU.
 *
 * This function checks the supported SIMD instructions on the CPU and returns the best available SIMDEnum value.
 * It prioritizes AVX512, FMA, AVX, SSE3, and SSE2 in that order, and prints a message indicating which optimization is being used.
 *
 * @return Returns the best supported SIMDEnum value based on the CPU capabilities.
 */
SIMDEnum GetBestSupportedSIMD()
{
    if (CPUopt.hasAVX512 && CPUopt.hasAVX512dq)
    {
        std::cerr << "Using AVX512 SIMD optimizations." << std::endl;
        return SIMDEnum::OptAVX512;
    }
    else if (CPUopt.hasFMA && !CPUopt.isBulldozer)
    {
        std::cerr << "Using FMA SIMD optimizations." << std::endl;
        return SIMDEnum::OptFMA;
    }
    else if (CPUopt.hasAVX && !CPUopt.isBulldozer)
    {
        std::cerr << "Using AVX SIMD optimizations." << std::endl;
        return SIMDEnum::OptAVX;
    }
    else if (CPUopt.hasSSE3)
    {
        std::cerr << "Using SSE3 SIMD optimizations." << std::endl;
        return SIMDEnum::OptSSE3;
    }
    else if (CPUopt.hasSSE2)
    {
        std::cerr << "Using SSE2 SIMD optimizations." << std::endl;
        return SIMDEnum::OptSSE2;
    }
    else
    {
        std::cerr << "Not using SIMD optimizations." << std::endl;
        return SIMDEnum::OptNONE;
    }
}

/**
 * @brief Sets the calculation strategy based on the specified SIMD optimization.
 *
 * This function selects and sets the appropriate calculation strategy based on the provided SIMDEnum value.
 * It utilizes different strategies for AVX512, FMA, AVX, SSE3, SSE2, and falls back to a default strategy if none are specified.
 *
 * @param useOptimization The SIMDEnum value representing the desired SIMD optimization strategy.
 */
void SetOptimizationStrategy(const SIMDEnum useOptimization)
{
    switch (useOptimization)
    {
#if !defined _VC140_XP
    case SIMDEnum::OptAVX512:
        calcCtx.SetStrategy(CreateAlignedShared<CalcStrategyAvx512>(64));
        break;
    case SIMDEnum::OptFMA:
        calcCtx.SetStrategy(CreateAlignedShared<CalcStrategyFma>(64));
        break;
    case SIMDEnum::OptAVX:
        calcCtx.SetStrategy(CreateAlignedShared<CalcStrategyAvx>(64));
        break;
#endif
    case SIMDEnum::OptSSE3:
        calcCtx.SetStrategy(CreateAlignedShared<CalcStrategySse3>(64));
        break;
    case SIMDEnum::OptSSE2:
        calcCtx.SetStrategy(CreateAlignedShared<CalcStrategySse2>(64));
        break;
    //case SIMDEnum::OptASIMD:
    //	calcCtx.set_strategy(std::make_unique<CalcStrategyAsimd>(64));  // TODO: Needs to be finished
    case SIMDEnum::OptNONE:
    case SIMDEnum::Undefined:
    default:
        calcCtx.SetStrategy(CreateAlignedShared<CalcStrategyNone>(64));
        break;
    }
}
