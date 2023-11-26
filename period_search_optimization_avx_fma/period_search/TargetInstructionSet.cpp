#include <string>

std::string GetTargetInstructionSet()
{

#if defined AVX512
	std::string target = "AVX512";
#elif defined FMA
	std::string target = "FMA";
#else
	std::string target = "AVX";
#endif

	return target;
}