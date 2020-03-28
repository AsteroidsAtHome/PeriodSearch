#pragma once
#include <windows.h>

bool GetVersionInfo(
    LPCTSTR filename,
    int& major,
    int& minor,
    int& build,
    int& revision);
