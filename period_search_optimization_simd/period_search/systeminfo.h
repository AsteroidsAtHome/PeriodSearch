#pragma once

#include <string>

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64) || defined __APPLE__
void getSystemInfo();
#endif
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
void getCpuInfoByArch(std::ifstream &cpuinfo);
#endif
double getTotalSystemMemory();

