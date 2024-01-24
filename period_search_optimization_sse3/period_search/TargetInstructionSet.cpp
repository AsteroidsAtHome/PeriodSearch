#include <string>

std::string GetTargetInstructionSet()
{

#if defined NO_SSE3
	std::string target = "SSE2";
#else
	std::string target = "SSE3";
#endif

	return target;
}