#pragma once
#ifdef __GNUC__
bool GetVersionInfo(int& major, int& minor, int& build, int& revision);
#else
#include "Windows.h"
bool GetVersionInfo(LPCTSTR filename, int& major, int& minor, int& build, int& revision);
#endif

