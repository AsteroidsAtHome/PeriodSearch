#include <string>
#include <cstring>
#include <iostream>
#include <array>
#include "Enums.h"
#include "declarations.h"
#include "globals.h"
#include "CalcStrategyAvx512.hpp"
#include "CalcStrategyFma.hpp"
#include "CalcStrategyAvx.hpp"
#include "CalcStrategySse3.hpp"
#include "CalcStrategySse2.hpp"
#include "CalcStrategyNone.hpp"

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
		  :);
	return a | (((uint64_t)(d)) << 32);
#endif
}

#if !defined __GNUC__ && defined _WIN32
std::string GetCpuInfo()
{
	std::array<int, 4> integerBuffer = {};
	constexpr size_t sizeofIntegerBuffer = sizeof(int) * integerBuffer.size();
	std::array<char, 64> charBuffer = {};

	// The information you wanna query __cpuid for.
	// https://learn.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=vs-2019

	constexpr std::array<int, 3> functionIds = {
		// Manufacturer
		//  EX: "Intel(R) Core(TM"
		0x8000'0002,
		// Model
		//  EX: ") i7-8700K CPU @"
		0x8000'0003,
		// Clockspeed
		//  EX: " 3.70GHz"
		0x8000'0004};

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
std::string GetCpuInfo()
{
	char CPUBrandString[0x40];
	// std::string CPUBrandString;
	unsigned int CPUInfo[4] = {0, 0, 0, 0};

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

static void GetCpuid(unsigned int info_type, unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d)
{
	int CPUInfo[4] = {0, 0, 0, 0};

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
static bool IsAVXSupportedByOS()
{
        unsigned int a, b, c, d;
        GetCpuid(1, a, b, c, d);

        bool osUsesXSAVE_XRSTORE = c & (1 << 27);

        return osUsesXSAVE_XRSTORE && (xgetbv(_XCR_XFEATURE_ENABLED_MASK) & 0x6);
}

static bool IsAVX512SupportedByOS()
{
        unsigned int a, b, c, d;
        GetCpuid(1, a, b, c, d);

        bool osUsesXSAVE_XRSTORE = c & (1 << 27);

        return osUsesXSAVE_XRSTORE && (xgetbv(_XCR_XFEATURE_ENABLED_MASK) & 0xe6);
}

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
	CPUopt.hasAVX = std_supported && (std_ecx & (1 << 28));
	if (CPUopt.hasAVX)
	{
		CPUopt.hasFMA = std_supported && (std_ecx & (1 << 12));
	}
	if (struc_ext_supported && IsAVX512SupportedByOS())
	{
		CPUopt.hasAVX512 = struc_ebx & (1 << 16);
		CPUopt.hasAVX512dq = struc_ebx & (1 << 17);
	}
}

/// <summary>
/// Check if manualy overriden optimization is supported. If not return the closest supported as a falback.
/// </summary>
/// <param name="simdEnum"></param>
/// <returns>SIMDEnum</returns>
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
	// else
	//{
	//	simd = SIMDEnum::OptNONE;
	// }

	if (tempSimd != simd)
	{
		std::cerr << "Choosen optimization " << getSIMDEnumName(tempSimd) << " is not supported. Switching to " << getSIMDEnumName(simd) << "." << std::endl;
	}

	return simd;
}

SIMDEnum GetBestSupportedSIMD()
{
	if (CPUopt.hasAVX512 && CPUopt.hasAVX512dq)
	{
		std::cerr << "Using AVX512 SIMD optimizations." << std::endl;
		return SIMDEnum::OptAVX512;
	}
	else if (CPUopt.hasFMA)
	{
		std::cerr << "Using FMA SIMD optimizations." << std::endl;
		return SIMDEnum::OptFMA;
	}
	else if (CPUopt.hasAVX)
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

void SetOptimizationStrategy(SIMDEnum useOptimization)
{
	switch (useOptimization)
	{
	case SIMDEnum::OptAVX512:
		calcCtx.set_strategy(std::make_unique<CalcStrategyAvx512>());
		break;
	case SIMDEnum::OptFMA:
		calcCtx.set_strategy(std::make_unique<CalcStrategyFma>());
		break;
	case SIMDEnum::OptAVX:
		calcCtx.set_strategy(std::make_unique<CalcStrategyAvx>());
		break;
	case SIMDEnum::OptSSE3:
		calcCtx.set_strategy(std::make_unique<CalcStrategySse3>());
		break;
	case SIMDEnum::OptSSE2:
		calcCtx.set_strategy(std::make_unique<CalcStrategySse2>());
		break;
	case SIMDEnum::OptNONE:
	default:
		calcCtx.set_strategy(std::make_unique<CalcStrategyNone>());
		break;
	}
}
