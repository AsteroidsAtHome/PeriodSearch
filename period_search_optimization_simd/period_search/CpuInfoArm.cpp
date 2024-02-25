#include <string>
#include <cstring>
#include <iostream>
#include <array>
#include "Enums.h"
#include "declarations.h"
#include "globals.h"
#include "CalcStrategyNone.hpp"
#include "CalcStrategyAsimd.hpp"
#include "CalcStrategySve.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
  #include <sys/auxv.h>

  #define AT_HWCAP 16
  #define HWCAP_ASIMD (1 << 1)
  #define HWCAP_ASIMDDP (1 << 20)
  #define HWCAP_SVE (1 << 22)
#endif

std::string GetCpuInfo()
{
	return "";
}

void GetSupportedSIMDs()
{
	#if defined(__aarch64__) || defined(_M_ARM64)
	  uint64_t hwcap = getauxval(AT_HWCAP);
      CPUopt.hasASIMD = hwcap & HWCAP_ASIMD && hwcap & HWCAP_ASIMDDP;
	  CPUopt.hasSVE = hwcap & HWCAP_SVE;
	#elif
	  CPUopt.hasASIMD = false;
	  CPUopt.hasSVE = false;
	#endif
}

SIMDEnum CheckSupportedSIMDs(SIMDEnum simd)
{
	SIMDEnum tempSimd = simd;
	if (simd == SIMDEnum::OptSVE)
	{
		simd = CPUopt.hasSVE
				   ? SIMDEnum::OptSVE
				   : SIMDEnum::OptASIMD;
	}

	if (simd == SIMDEnum::OptASIMD)
	{
		simd = CPUopt.hasASIMD
				   ? SIMDEnum::OptASIMD
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
	if (CPUopt.hasSVE)
	{
		std::cerr << "Using SVE SIMD optimizations." << std::endl;
		return SIMDEnum::OptSVE;
	}
	else if (CPUopt.hasASIMD)
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
	case SIMDEnum::OptSVE:
		calcCtx.set_strategy(std::make_unique<CalcStrategySve>());
		break;
	case SIMDEnum::OptASIMD:
		calcCtx.set_strategy(std::make_unique<CalcStrategyAsimd>());
		break;
	case SIMDEnum::OptNONE:
	default:
		calcCtx.set_strategy(std::make_unique<CalcStrategyNone>());
		break;
	}
}
