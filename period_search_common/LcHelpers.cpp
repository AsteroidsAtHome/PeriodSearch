//#include "pch.h"
#include <string>
#include <cstring>
#include <fstream>
#include <numeric>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <vector>

#if defined __GNU__
#include <bits/stdc++.h>
#endif

#include "constants.h"
#include "arrayHelpers.hpp"

void processLine15(struct globals& gl, const char* line, int& err)
{
	err = sscanf(line, "%d", &gl.Lcurves);
	if (err != 1) {
		err = -1;
		return;
	}

	gl.Lpoints.resize(gl.Lcurves + 2, 0);
}

void processLine16(struct globals& gl, const char* line, int& err, int& offset, int& i, int& i_temp, int lineNumber)
{
	err = sscanf(line, "%d %d", &gl.Lpoints[0], &i_temp);
	if (err != 2) {
		err = -1;
		return;
	}

	offset = lineNumber;
	i++;
}

void processSubsequentLines(struct globals& gl, const char* line, int& err, int& offset, int& i, int& i_temp, int lineNumber) {
	err = sscanf(line, "%d %d", &gl.Lpoints[i], &i_temp);
	if (err != 2) {
		err = -1;
		return;
	}
	offset = lineNumber;
	i++;
}

/**
 * @brief Performs convexity regularization to create a final 'lightcurve'.
 *
 * This function generates a final 'lightcurve' that consists of the three components
 * of the residual non-convex vectors. These vectors should all be zero.
 *
 * @param gl A struct containing global variables.
 */
void MakeConvexityRegularization(struct globals& gl)
{
	gl.Lcurves = gl.Lcurves + 1;
	gl.Lpoints[gl.Lcurves] = 3;
	gl.Inrel[gl.Lcurves] = 0;

	gl.maxDataPoints = std::accumulate(gl.Lpoints.begin(), gl.Lpoints.end(), 0);

	//for (auto q = 0; q <= gl.Lcurves; q++)
	//    fprintf(stderr, "Lpoints[%d] %d\n", q, gl.Lpoints[q]);
}

/**
 * @brief Performs the first loop over lightcurves to find all data points.
 *
 * This function performs the initial loop over lightcurves to locate all data points,
 * replacing MAX_LC_POINTS, MAX_N_OBS, etc. It processes the lightcurve data based on
 * the provided global structure and the filename.
 *
 * @param gl A struct containing global variables.
 * @param filename A constant character pointer representing the filename.
 * @return An integer indicating the success or failure of the operation.
 */
int PrepareLcData(struct globals& gl, const char* filename)
{
	int i_temp;
	int err = 0;

	std::ifstream file(filename);
	std::string lineStr;
	gl.Lcurves = 0;

	if (!file.is_open())
	{
		return 2;
	}

	int lineNumber = 0;
	int offset = 0;
	int i = 0;
#if defined(_MSC_VER) && _MSC_VER < 1900
#else
	std::unordered_map<int, std::function<void(const char*, int&)>> actions;
	actions[15] = [&](const char* line, int& err) { processLine15(gl, line, err); };
	actions[16] = [&](const char* line, int& err) { processLine16(gl, line, err, offset, i, i_temp, lineNumber); };
#endif

	while (std::getline(file, lineStr))
	{
		char line[2000];
		std::strcpy(line, lineStr.c_str());
		lineNumber++;

#if defined(_MSC_VER) && _MSC_VER < 1900
		if (lineNumber == 15)
		{
			processLine15(gl, line, err);
			if (err <= 0) {
				file.close();
				return err;
			}
		}
		else if (lineNumber == 16)
		{
			processLine16(gl, line, err, offset, i, i_temp, lineNumber);
			if (err <= 0) {
				file.close();
				return err;
			}
		}
#else
		if (actions.find(lineNumber) != actions.end()) {
			actions[lineNumber](line, err);
			if (err <= 0) {
				file.close();
				return err;
			}
		}
#endif

		if (lineNumber <= 16)
		{
			continue;
		}

		if (lineNumber == offset + 1 + gl.Lpoints[i - 1])
		{
			processSubsequentLines(gl, line, err, offset, i, i_temp, lineNumber);
			if (err <= 0) {
				file.close();
				return err;
			}
			if (i == gl.Lcurves)
			{
				break;
			}
		}
	}

	file.close();

	gl.Inrel.resize(gl.Lcurves + 2, 0);
	gl.maxLcPoints = *(std::max_element(gl.Lpoints.begin(), gl.Lpoints.end()));

	gl.ytemp.resize(gl.maxLcPoints + 2, 0.0);        // Not used in CUDA

	gl.dytemp_sizeY = MAX_N_PAR + 1 + 4;
	gl.dytemp_sizeX = gl.maxLcPoints + 2;
	init_matrix(gl.dytemp, gl.dytemp_sizeX + 1, gl.dytemp_sizeY + 1, 0.0);  // Not used in CUDA

	gl.maxDataPoints = std::accumulate(gl.Lpoints.begin(), gl.Lpoints.end(), 0);   // OK

	gl.Weight.resize(gl.maxDataPoints + 1 + 4, 0.0);
	gl.ave = 0.0;

	return 1;
}