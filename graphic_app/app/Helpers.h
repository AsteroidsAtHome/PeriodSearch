#pragma once
#include <string>
#include "PeriodStruct.h"
using namespace std;
using namespace ps;

string GetFileNameFromSoftLink(const string& softLink);

bool GetWorkUnitOutputData(const string& softLinkOut, const string& wuOutTempFilename);

bool GetPeriodOutputData(const string& softLinkOut, double& lastFileSize, PeriodStruct& ps);