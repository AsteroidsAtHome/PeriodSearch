#pragma once
//#include <iostream>
//#include <sstream>
//#include <string>

#if defined(ARM) || defined(ARM32) || defined(ARM64)
void getSystemInfo();
void getCpuInfoByArch(std::ifstream &cpuinfo);
#endif
float getTotalSystemMemory();