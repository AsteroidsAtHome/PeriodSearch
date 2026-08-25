#pragma once

#if defined __GNUC__
	const int _major = 102;
	const int _minor = 22;
	const int _build = 1;
	const int _revision = 1;
#else // _WIN32
#include "Windows.h"
#endif


#if defined __GNUC__
bool GetVersionInfo(
	int& major,
	int& minor,
	int& build,
	int& revision)
	{
		major = _major;
		minor = _minor;
		build = _build;
		revision = _revision;

		return true;
	};

#else

bool GetVersionInfo(
	LPCTSTR filename,
	int& major,
	int& minor,
	int& build,
	int& revision);
#endif