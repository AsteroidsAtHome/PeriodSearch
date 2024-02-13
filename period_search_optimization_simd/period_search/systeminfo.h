#pragma once

#include <string>

#if defined(ARM) || defined(ARM32) || defined(ARM64)
void getSystemInfo();
void getCpuInfoByArch(std::ifstream &cpuinfo);
#endif
#ifdef _WIN32
std::string getTotalSystemMemory();
#else
float getTotalSystemMemory();
#endif
