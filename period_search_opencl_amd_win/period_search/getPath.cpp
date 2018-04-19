/*
 * https://stackoverflow.com/questions/143174/how-do-i-get-the-directory-that-a-program-is-running-from
 */

#include <stdio.h>  /* defines FILENAME_MAX */
#include <string>

#ifdef _WIN32
    #include <direct.h>
    #define GetCurrentDir _getcwd
#elif __linux__
    #include <unistd.h>
    #define GetCurrentDir getcwd
#endif
// Other OSes ?

std::string getCurrentPath() {
    char cCurrentPath[FILENAME_MAX];

    if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
    {
        return "";
    }

    return cCurrentPath;
}

  //std::string getCurrentPath()
  //  {
  //      char result[MAX_PATH];
  //      return std::string(result, GetModuleFileName(NULL, result, MAX_PATH));
  //  }

  //void getCurrentPath() {
  //    std::string path = getexepath();

  //    printf("The current working directory is %s", path);
  //}
