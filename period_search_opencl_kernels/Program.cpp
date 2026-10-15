#include <fstream>
#include <iostream>
#include <sstream>

#include "declarations.h"

#if defined _MSC_VER
#include <windows.h>
#endif

#define MINI_CASE_SENSITIVE
#include "ini.h"

static std::vector<std::string> splitString(const std::string& str, char delimiter)
{
	std::vector<std::string> tokens;
	std::stringstream ss(str);
	std::string token;

	while (std::getline(ss, token, delimiter)) {
		tokens.push_back(token);
	}

	return tokens;
}

std::vector<std::string> readFileList(const std::string& filename)
{
	std::vector<std::string> lines;
	std::ifstream file(filename);
	std::string line;

	if (file.is_open())
	{
		while (std::getline(file, line))
		{
			// Check if line is empty or contains only whitespace
			bool isWhitespaceOnly = true;
			for (char ch : line)
			{
				if (!std::isspace(ch))
				{
					isWhitespaceOnly = false;
					break;
				}
			}

			if (line.empty() || isWhitespaceOnly || line[0] == '#') {
				continue;
			}

			lines.push_back(line);
		}

		file.close();
	}
	else
	{
		std::cerr << "Unable to open file: " << filename << std::endl;
	}

	return lines;
}

int main(int argc, char** argv)
{
	mINI::INIFile file("kernels.ini");
	mINI::INIStructure ini;

	if (!file.read(ini))
	{
		exit(1);
	}

	std::string sourcepath = ini["kernel_files"]["sourcepath"];
	std::string outputpath = ini["kernel_files"]["outputpath"];
	std::string fileList = ini["kernel_files"]["file_list"];

	std::time_t now = std::time(nullptr);

	char datetime[100];
	struct tm timeinfo;
	localtime_s(&timeinfo, &now);
	std::strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", &timeinfo);

	std::string dateTimeString(datetime);
	std::hash<std::string> hasher;
	size_t hashValue = hasher(dateTimeString);

	std::string kernelHashFile = outputpath + "kernels_hash.cpp";
	std::ofstream hFile(kernelHashFile, std::ios::out | std::ios::binary);
	hFile << std::endl << "inline const char* kernel_hash = \"" << hashValue << "\";" << std::endl;
	hFile.close();

	std::string kernelsOutputFile = outputpath + "kernelSource.cl";
	std::stringstream st;

	auto files = readFileList(fileList);

	for (int i = 0; i < files.size(); i++)
	{
		auto clFile = sourcepath + files[i];
#if !defined _WIN32
		std::ifstream clfile(clFile, std::ios::in | std::ios::binary);
#else
		std::ifstream clfile(clFile);
#endif
		st << clfile.rdbuf();
		clfile.close();
	}

	auto kernel_code = st.str(); //.c_str();
	st.flush();

	std::ofstream out(kernelsOutputFile, std::ios::out | std::ios::binary);
	out << kernel_code << std::endl;

	out << "// Created on: " << datetime << std::endl;

	out.close();

	std::cout << "Kernel source file created: " << kernelsOutputFile << std::endl;
}
