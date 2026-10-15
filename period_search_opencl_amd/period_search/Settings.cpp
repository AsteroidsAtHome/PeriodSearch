#include "ini.h"
#include "kernels.cpp"

bool ShouldReadSource(char *deviceName)
{
    mINI::INIFile file("settings.ini");
    mINI::INIStructure ini;

    if (!file.read(ini))
    {
        file.generate(ini, true);
    }

    std::string &savedKernelsHashString = ini["kernels"]["hash"];
    std::string savedKernelsHash = savedKernelsHashString.empty() ? "0" : savedKernelsHashString;

    bool shouldReadsource = false;
    if (savedKernelsHash != std::string(kernel_hash))
    {
        shouldReadsource = true;
        ini["kernels"]["hash"] = std::string(kernel_hash);
        file.write(ini, true);

        return shouldReadsource;
    }

    std::string &savedDeviceName = ini["device"]["name"];
    if (strcmp(savedDeviceName.c_str(), deviceName) != 0)
    {
        ini["device"]["name"] = deviceName;
        shouldReadsource = true;
        file.write(ini, true);
    }

    return shouldReadsource;
}
