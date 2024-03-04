#include <string>
#include <iostream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <thread>
#include <bitset>
#include <iomanip>
#include "systeminfo.h"
#include "constants.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined __APPLE__
#include <sys/sysctl.h>
#endif

using namespace std;
using std::string;
constexpr auto MAXC = 2048;

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)

void printCerr(string line)
{
	if(!line.empty())
	  cerr << line << endl;
	return;
}

void getCpuInfoByArch(std::ifstream &cpuinfo)
{
	if (cpuinfo.fail())
	{
		return;
	}

	string bogoMips, cpuArch, cpuImpl, hardware, model, revision, variant, part, cpuRevision, vendorId, modelName;
	string data;
	while (std::getline(cpuinfo, data))
	{
#ifdef _DEBUG
		// cout << data << "\n";
#endif

        vendorId = data.find("vendor_id") != string::npos ? data : vendorId;
        modelName = data.find("model name") != string::npos ? data : modelName;
		model = data.find("Model") != string::npos ? data : model;
		hardware = data.find("Hardware") != string::npos ? data : hardware;
		revision = data.find("Revision") != string::npos ? data : revision;
		cpuImpl = data.find("CPU implementer") != string::npos ? data : cpuImpl;
		cpuArch = data.find("CPU architecture") != string::npos ? data : cpuArch;
		bogoMips = data.find("BogoMIPS") != string::npos ? data : bogoMips;
		variant = data.find("CPU variant") != string::npos ? data : variant;
		part = data.find("CPU part") != string::npos ? data : part;
		cpuRevision = data.find("CPU revision") != string::npos ? data : cpuRevision;
	}
	cpuinfo.close();

    printCerr(vendorId);
    printCerr(modelName);
	printCerr(model);
	printCerr(hardware);
	printCerr(revision);
	printCerr(cpuImpl);
	printCerr(cpuArch);
	printCerr(bogoMips);
	printCerr(variant);
	printCerr(part);
	printCerr(cpuRevision);
}
#endif

std::ifstream getIfstream(const char *fileName)
{
	std::ifstream ifstream(fileName);
	#ifdef _DEBUG
		if (ifstream.fail())
		{
			std::cout << "Cannot open file " << fileName << endl;
		}
	#endif

	return ifstream;
}

void getCpuFrequency(std::ifstream &cpufreqIfstream, string definition)
{
	if (cpufreqIfstream.fail())
	{
		cpufreqIfstream.close();
		return;
	}
	else
	{
		string data;
		while (std::getline(cpufreqIfstream, data))
		{
			int freqKiloHerz = std::stoi(data);
			auto freqMegaHerz = freqKiloHerz / 1000;
			cerr << "CPU " << definition << " frequency: " << freqMegaHerz << " MHz" << endl;
		}

		cpufreqIfstream.close();
	}
}

#if defined _WIN32 || (defined __GNUC__ && defined __APPLE__ && !(defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)))
void getCpuInfoByArch(ifstream &cpuinfo)
{
}
#endif // _WIN32 || macOS

#if (defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)) && !defined __APPLE__
void getSystemInfo()
{
	auto cpuinfo = getIfstream("/proc/cpuinfo");
	getCpuInfoByArch(cpuinfo);

	const auto processor_count = std::thread::hardware_concurrency();
	fprintf(stderr, "Number of processors: %d\n", processor_count);

	auto cpuCurFreqIfstream = getIfstream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
	getCpuFrequency(cpuCurFreqIfstream, "current");

	auto cpuMinFreqIfstream = getIfstream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq");
	getCpuFrequency(cpuMinFreqIfstream, "minimum");

	auto cpuMaxFreqIfstream = getIfstream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
	getCpuFrequency(cpuMaxFreqIfstream, "maximum");

	auto totalMemory = getTotalSystemMemory();
	cerr.precision(2);
	cerr << "Available memory: " << totalMemory << " GB" << endl;
}
#elif defined __APPLE__
void getSystemInfo()
{
	char cpuBrand[64];
	size_t len = sizeof(cpuBrand);
	int mib[2];
    int64_t physical_memory;
    size_t length;
    length = sizeof(int64_t);

	cerr << "CPU: ";
	if(sysctlbyname("machdep.cpu.brand_string", &cpuBrand[0], &len, 0, 0)==0)
	{
		cerr << cpuBrand << endl;
	} else
	{
		cerr << "Unknown";
	}

    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
	if(physical_memory != NULL || physical_memory > 0)
	{
		float memSizeG =  physical_memory / 1024.0 / 1024.0 / 1024.0;
		cerr << "RAM: " << memSizeG << "GB" << endl;
	}
}
#endif

#ifdef _WIN32
double getTotalSystemMemory()
{
	MEMORYSTATUSEX status = { sizeof status };
	//status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	//float memory = (float)status.ullTotalPhys;

	//double memory = status.ullTotalPhys;
	auto memoryGb = (unsigned long long)status.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);

	return memoryGb;
}

#else
double getTotalSystemMemory()
{
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	double memory = (double)(pages * page_size);
	double memoryGb = memory / 1024 / 1024 / 1024;

	return memoryGb;
}
#endif