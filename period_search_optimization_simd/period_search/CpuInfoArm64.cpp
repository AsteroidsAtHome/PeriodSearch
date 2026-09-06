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
#include "SIMDHelpers.h"

#if defined(__linux__) && (defined(__arm__) || defined(_M_ARM) || defined(__aarch64__) || defined(_M_ARM64))
  #include <sys/auxv.h>
  #include <asm/hwcap.h>
  #ifndef HWCAP_SVE			// missing on pre-4.15 kernel headers
    #define HWCAP_SVE (1 << 22)
  #endif
#endif

std::string GetCpuInfo()
{
	return "";
}

#if !defined(_WIN32) && !defined(__APPLE__)
void DetectArmv8CpuFeatures(long hwcaps)
{
    if(hwcaps & HWCAP_AES)
	{
        std::cerr << "HWCAP_AES: AES instructions are available." << std::endl;
    }
	else
	{
		std::cerr << "HWCAP_AES: AES instructions are NOT available." << std::endl;
	}

    if(hwcaps & HWCAP_CRC32)
	{
		std::cerr << "HWCAP_CRC32: CRC32 instructions are available." << std::endl;
    }
	else
	{
		std::cerr << "HWCAP_CRC32: CRC32 instructions are NOT available." << std::endl;
	}

    if(hwcaps & HWCAP_PMULL)
	{
		std::cerr << "HWCAP_PMULL: PMULL/PMULL2 instructions that operate on 64-bit data are available." << std::endl;
    }
	else
	{
		std::cerr << "HWCAP_PMULL: PMULL/PMULL2 instructions that operate on 64-bit data are NOT available." << std::endl;
	}

    if(hwcaps & HWCAP_SHA1)
	{
		std::cerr << "HWCAP_SHA1: SHA1 instructions are available." << std::endl;
    }
	else
	{
		std::cerr << "HWCAP_SHA1: SHA1 instructions are NOT available." << std::endl;
	}

    if(hwcaps & HWCAP_SHA2)
	{
        std::cerr << "HWCAP_SHA2: SHA2 instructions are available." << std::endl;
    }
	else
	{
		std::cerr << "HWCAP_SHA2: SHA2 instructions are NOT available." << std::endl;
	}
}
#endif

void GetSupportedSIMDs()
{
	CPUopt.hasASIMD = false;
	CPUopt.hasSVE = false;
	#if defined(_WIN32) // ARM win, snapdragon...
      CPUopt.hasASIMD = true;
	#elif defined(__linux__)
	  #if (defined(__aarch64__) || defined(_M_ARM64))
	    uint64_t hwcap = getauxval(AT_HWCAP);
		// NOTE: For debug purposes:
		// DetectArmv8CpuFeatures(hwcap);

        CPUopt.hasASIMD = hwcap & HWCAP_ASIMD;
        CPUopt.hasSVE = (hwcap & HWCAP_SVE);
	  #elif (defined(__arm__) || defined(_M_ARM))
	    uint64_t hwcap = getauxval(AT_HWCAP);
        CPUopt.hasASIMD = hwcap & HWCAP_NEON;
	  #endif
	#elif defined(__APPLE__)
	  CPUopt.hasASIMD = true; // M1
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

	if (tempSimd != simd)
	{
		std::cerr << "Choosen optimization " << getSIMDEnumName(tempSimd) << " is not supported. Switching to " << getSIMDEnumName(simd) << "." << std::endl;
	}

	return simd;
}

SIMDEnum GetBestSupportedSIMD()
{
	//if (CPUopt.hasSVE) // slower than ASIMD, don't choose automatically
	//{
	//	std::cerr << "Using SVE SIMD optimizations." << std::endl;
	//	return SIMDEnum::OptSVE;
	//}
	//else 
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
		case SIMDEnum::OptSVE:
			calcCtx.SetStrategy(CreateAlignedShared<CalcStrategySve>(64));
			break;
		case SIMDEnum::OptASIMD:
			calcCtx.SetStrategy(CreateAlignedShared<CalcStrategyAsimd>(64));
			break;
	
		case SIMDEnum::OptNONE:
		default:
			calcCtx.SetStrategy(CreateAlignedShared<CalcStrategyNone>(64));
			break;
	}
}
