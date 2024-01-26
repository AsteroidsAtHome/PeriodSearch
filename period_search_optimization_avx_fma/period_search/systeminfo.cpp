#include <string>
#include <iostream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <thread>
#include "systeminfo.h"
#include <bitset>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;
using std::string;
constexpr auto MAXC = 2048;

#if defined(ARM) || defined(ARM32) || defined(ARM64)

//TODO: Obsolete
void getRevisionCodes(std::string revisionCodes)
{
	cout << revisionCodes << endl;
	// Revision        : b03115
	auto delimiter = ":";
	auto index = revisionCodes.find(delimiter);
	auto len = revisionCodes.length();
	cout << len << endl;
	cout << index << endl;
	auto revisionStr = revisionCodes.substr(index + 1, len - index - 1);

	uint32_t revision = (uint32_t)std::stoi(revisionStr, NULL, 16);
	// https://www.raspberrypi.org/documentation/hardware/raspberrypi/revision-codes/README.md
	printf("\nHW revision: 0x%08x\n", revision);
	//std::bitset<30> bits(revision);
	cout << bitset<8 * sizeof(revision)>(revision) << endl;

	if (revision & (1 << 23)) { // new style revision
		printf("  Type: %d\n", (revision >> 4) & 0xff);
		printf("  Rev: %d\n", revision & 0xf);
		printf("  Proc: %d\n", (revision >> 12) & 0xf);
		printf("  Manufacturer: %d\n", (revision >> 16) & 0xf);
		printf("  Ram: %d\n", (revision >> 20) & 0x7);
	}
}

void getCpuInfoByArch(std::ifstream& cpuinfo)
{
	if (cpuinfo.fail()) {
		return;
	}

	string bogoMips, cpuArch, cpuImpl, hardware, model, revision, variant, part, cpuRevision;
	string data;
	while (std::getline(cpuinfo, data))
	{
#ifdef _DEBUG
		//cout << data << "\n";
#endif

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

	cerr << model << endl;
	cerr << hardware << endl;
	cerr << revision << endl;
	cerr << cpuImpl << endl;
	cerr << cpuArch << endl;
	cerr << bogoMips << endl;
	cerr << variant << endl;
	cerr << part << endl;
	cerr << cpuRevision << endl;

	//getRevisionCodes(revision);
}
#endif // ARM / ARM32 / ARM64

std::ifstream getIfstream(const char* fileName) {
	std::ifstream ifstream(fileName);
	if (ifstream.fail())
	{
		std::cout << "Cannot open file " << fileName << endl;
	}

	return ifstream;
}

void getCpuFrequency(std::ifstream& cpufreqIfstream, string definition) {
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

#if defined _WIN32 || (defined __GNUC__ && defined __APPLE__)
void getCpuInfoByArch(ifstream& cpuinfo) {

}
#endif // _WIN32 || macOS

#if defined(ARM) || defined(ARM32) || defined(ARM64)
void getSystemInfo() {
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
#endif

#ifdef _WIN32
float getTotalSystemMemory()
{
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	auto memory = status.ullTotalPhys;
	auto memoryGb = memory / 1024 / 1024 / 1024;

	return memoryGb;
}

#else
float getTotalSystemMemory()
{
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	auto memory = (float)(pages * page_size);
	auto memoryGb = memory / 1024 / 1024 / 1024;

	return memoryGb;
}
#endif