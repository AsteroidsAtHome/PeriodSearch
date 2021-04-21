#include <cstdlib>
#include <filesystem>
#include <string>
#include "boinc_api.h"
#include "PeriodStruct.h"

using namespace std;
namespace fs = std::filesystem;
using namespace ps;

string GetFileNameFromSoftLink(const string& softLink)
{
	std::string resolvedName;
	const auto retval = boinc_resolve_filename_s(softLink.c_str(), resolvedName);
	if (retval)
	{
		cerr << "Can't resolve filename of soft link: " << softLink << endl;
		return "";
	}

	auto filename = fs::path(resolvedName).filename().string();

	return filename;
}

bool GetWorkUnitOutputData(const string &softLinkOut, const string &wuOutTempFilename)
{
	char m_buf[256];

	string periodSearchOut;
	const auto retval = boinc_resolve_filename_s(softLinkOut.c_str(), periodSearchOut);
	if (retval) {
		cerr << "Can't resolve filename of soft link: " << periodSearchOut << endl;
	}

	//f = boinc_fopen(resolved_name.c_str(), "r");
	auto* const infile = boinc_fopen(periodSearchOut.c_str(), "r");
	if (!infile) {
		fprintf(stderr, "%s Couldn't find result file, resolved name %s.\n", boinc_msg_prefix(m_buf, sizeof(m_buf)), periodSearchOut.c_str());
		return true;
	}

	auto* const outfile = boinc_fopen(wuOutTempFilename.c_str(), "w");
	if (!infile) {
		fprintf(stderr, "%s Couldn't write to temp file, resolved name %s.\n", boinc_msg_prefix(m_buf, sizeof(m_buf)), wuOutTempFilename.c_str());
		return true;
	}

	const auto buffsize = 8192;
	char line[2000];
	char buffer[buffsize], c;
	double filesize;
	file_size(periodSearchOut.c_str(), filesize);
	if(filesize <= 0)
	{
		return true;
	}
	
	setvbuf(infile, buffer, _IOFBF, buffsize);

	// while testing
	//auto now = steady_clock::now();
	//

	while (fgets(line, 2000, infile) != nullptr)
	{
		fputs(line, outfile);
	}

	// while testing
	//auto last = steady_clock::now();
	//auto duration = duration_cast<milliseconds>(last - now).count();
	//cout << "Duration: " << duration << " ms" << endl;
	// ------

	fclose(infile);
	fclose(outfile);
	return false;
}

bool GetPeriodOutputData(const string& softLinkOut, double &lastFileSize, PeriodStruct& ps)
{
	char m_buf[256];

	string periodSearchOut;
	const auto retval = boinc_resolve_filename_s(softLinkOut.c_str(), periodSearchOut);
	if (retval) {
		cerr << "Can't resolve filename of soft link: " << periodSearchOut << endl;
	}

	//f = boinc_fopen(resolved_name.c_str(), "r");
	auto* const infile = boinc_fopen(periodSearchOut.c_str(), "r");
	if (!infile) {
		fprintf(stderr, "%s Couldn't find result file, resolved name %s.\n", boinc_msg_prefix(m_buf, sizeof(m_buf)), periodSearchOut.c_str());
		return true;
	}

	double fileSize;
	file_size(periodSearchOut.c_str(), fileSize);
	if (fileSize <= 0 || fileSize == lastFileSize)
	{
		return true;
	}

	lastFileSize = fileSize;
	auto psSize = ps.rms.size();
	size_t i = 0;
	std::filebuf buf(infile);
	std::istream fin(&buf);
	//std::ifstream fin;
	//fin.open(wu_out_temp_filename, std::ifstream::in);
	if (fin.good()) {
		while (!fin.eof())
		{
			string buffer;
			getline(fin, buffer, '\n');
				if (buffer.empty() || i < psSize) 
				{
					continue;
				}
			ResizeVectors(i + 1, ps);
			std::stringstream strStream(buffer);
			strStream >> ps.period[i] >> ps.rms[i] >> ps.chi[i] >> ps.dark[i] >> ps.alpha[i] >> ps.beta[i];
			//cout << ps.period[i] << ps.rms[i] << ps.chi[i] << ps.dark[i] << ps.alpha[i] << ps.beta[i] << endl;
			i++;
		}
	}

	fclose(infile);
	
	return false;
}