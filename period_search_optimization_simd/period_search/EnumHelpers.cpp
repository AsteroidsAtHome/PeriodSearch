#include <string>
#include "Enums.h"

/**
 * @brief Retrieves the string representation of a given SIMDEnum value.
 *
 * This function takes a SIMDEnum value and returns its corresponding string representation.
 * It provides a human-readable name for various SIMD optimization levels.
 *
 * @param simdEnum The SIMDEnum value to be converted to a string.
 * @return Returns a string representing the name of the SIMDEnum value.
 *
 * @note The function covers several SIMD optimization levels including SSE2, SSE3, AVX, FMA, AVX512, ASIMD, SVE, NONE, and an Undefined state.
 */
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