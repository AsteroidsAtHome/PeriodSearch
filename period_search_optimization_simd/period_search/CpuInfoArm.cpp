#include <string>
#include <cstring>
#include <iostream>
#include <array>
#include "Enums.h"
#include "declarations.h"
#include "globals.h"
#include "CalcStrategyNone.hpp"
#include "CalcStrategyAsimd.hpp"

#if defined(__linux__) && (defined(__arm__) || defined(_M_ARM) || defined(__aarch64__) || defined(_M_ARM64))
  #include <sys/auxv.h>
  #include <asm/hwcap.h>
#endif

std::string GetCpuInfo()
{
	return "";
}

void GetSupportedSIMDs()
{
	#if defined(__linux__)
	  #if (defined(__aarch64__) || defined(_M_ARM64))
	    uint64_t hwcap = getauxval(AT_HWCAP);
        CPUopt.hasASIMD = hwcap & HWCAP_ASIMD;
	  #elif (defined(__arm__) || defined(_M_ARM))
	    uint64_t hwcap = getauxval(AT_HWCAP);
        CPUopt.hasASIMD = hwcap & HWCAP_NEON;
	  #endif
	#elif defined(__APPLE__)
	  CPUopt.hasASIMD = true; // M1
	#else
	  CPUopt.hasASIMD = false;
	#endif
}

SIMDEnum CheckSupportedSIMDs(SIMDEnum simd)
{
	SIMDEnum tempSimd = simd;
	if (simd == SIMDEnum::OptASIMD)
	{
		simd = CPUopt.hasASIMD
				   ? SIMDEnum::OptASIMD
				   : SIMDEnum::OptNONE;
	}

	if (tempSimd != simd)
	{
		std::cerr << "Choosen optimization " << getSIMDEnumName(tempSimd) << " is not supported. Switching to " << getSIMDEnumName(simd) << "." << std::endl;
	}

	return simd;
}

SIMDEnum GetBestSupportedSIMD()
{
	if (CPUopt.hasASIMD)
	{
		std::cerr << "Using ASIMD (NEON) SIMD optimizations." << std::endl;
		return SIMDEnum::OptASIMD;
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
	case SIMDEnum::OptASIMD:
		calcCtx.set_strategy(std::make_unique<CalcStrategyAsimd>());
		break;
	case SIMDEnum::OptNONE:
	default:
		calcCtx.set_strategy(std::make_unique<CalcStrategyNone>());
		break;
	}
}
