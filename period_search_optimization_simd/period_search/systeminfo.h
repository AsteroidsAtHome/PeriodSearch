#pragma once

#include <string>

#if defined(ARM) || defined(ARM32) || defined(ARM64)
void getSystemInfo();
void getCpuInfoByArch(std::ifstream &cpuinfo);
#endif
std::string getTotalSystemMemory();