#if !defined __GNUC__ && defined _WIN32
#include <intrin.h>
#elif defined __GNUC__ && !defined ARM64
#include <cpuid.h>
#endif

#include <string>
#include <cstring>
#include <array>

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
		0x8000'0004
	};

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
	//std::string CPUBrandString;
	unsigned int CPUInfo[4] = { 0,0,0,0 };

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