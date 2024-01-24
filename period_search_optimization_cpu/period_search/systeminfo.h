#pragma once
//#include <iostream>
//#include <sstream>
//#include <string>

#if defined(ARM) || defined(ARM32) || defined(ARM64) || defined __APPLE__
void getSystemInfo();
#endif

#if defined(ARM) || defined(ARM32) || defined(ARM64)
void getCpuInfoByArch(std::ifstream &cpuinfo);
#endif

float getTotalSystemMemory();