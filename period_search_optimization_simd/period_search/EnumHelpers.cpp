#include <string>
#include "Enums.h"

const std::string getSIMDEnumName(SIMDEnum simdEnum)
{
	std::string result = "";

	switch (simdEnum)
	{
		case SIMDEnum::OptSSE2:
			result = "SSE2";
			break;
		case SIMDEnum::OptSSE3:
			result = "SSE3";
			break;
		case SIMDEnum::OptAVX:
			result = "AVX";
			break;
		case SIMDEnum::OptFMA:
			result = "FMA";
			break;
		case SIMDEnum::OptAVX512:
			result = "AVX512";
			break;
		case SIMDEnum::OptASIMD:
			result = "ASIMD";
			break;
		case SIMDEnum::OptSVE:
			result = "SVE";
			break;
		case SIMDEnum::OptNONE:
			result = "NONE";
			break;
		case SIMDEnum::Undefined:
		default:
			result = "Undefined";
			break;
	}
	return result;
};