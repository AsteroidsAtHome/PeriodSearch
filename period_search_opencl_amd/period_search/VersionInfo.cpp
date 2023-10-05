#include "stdafx.h"
#if defined __GNUC__
const int _major = 102;
const int _minor = 18;
const int _build = 0;
const int _revision = 0;
#else
#include <windows.h>
#include <tchar.h>
#endif

#if defined __GNUC__
bool GetVersionInfo(int &major, int &minor, int &build, int &revision)
{
    major = _major;
    minor = _minor;
    build = _build;
    revision = _revision;
    
    return true;
}

#else

bool GetVersionInfo(LPCTSTR filename, int &major, int &minor, int &build, int &revision)
{
    DWORD verBufferSize;
    char verBuffer[8192];

    //  Get the size of the version info block in the file
    verBufferSize = GetFileVersionInfoSize(filename, NULL);
    if (verBufferSize > 0 && verBufferSize <= sizeof(verBuffer))
    {
        //  get the version block from the file
        if (TRUE == GetFileVersionInfo(filename, NULL, verBufferSize, verBuffer))
        {
            UINT length;
            VS_FIXEDFILEINFO *verInfo = NULL;

            //  Query the version information for neutral language
            if (TRUE == VerQueryValue(
                            verBuffer,
                            _T("\\"),
                            reinterpret_cast<LPVOID *>(&verInfo),
                            &length))
            {
                //  Pull the version values.
                major = HIWORD(verInfo->dwProductVersionMS);
                minor = LOWORD(verInfo->dwProductVersionMS);
                build = HIWORD(verInfo->dwProductVersionLS);
                revision = LOWORD(verInfo->dwProductVersionLS);
                return true;
            }
        }
    }

    return false;
}

#endif
