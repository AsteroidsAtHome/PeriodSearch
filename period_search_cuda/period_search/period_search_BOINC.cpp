// This file is part of BOINC.
// http://boinc.berkeley.edu
// Copyright (C) 2008 University of California
//
// BOINC is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// BOINC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with BOINC.  If not, see <http://www.gnu.org/licenses/>.

/**
 * @brief Scans input lightcurves to find the best period, pole, and scattering solution.
 *
 * This program takes the input lightcurves, scans over the given period range,
 * and finds the best period + pole + shape + scattering solution. Shape is
 * forgotten. The period, RMS residual of the fit, and pole solution (lambda, beta)
 * are given to the output. It starts from six initial poles and selects the best period,
 * then reports the best pole solution.
 *  Version for BOINC.
 *	8.11.2006
 *
 * @syntax:
 * period_search_BOINC
 *
 * @output:
 * - period [hr]
 * - RMS deviation
 * - chi^2
 * - dark facet [%]
 * - lambda_best
 * - beta_best
 *
 * @note:
 * - 16.4.2012: New version of lightcurve files (new input LCs format).
 *              Testing the dark facet, finding the optimal value for convexity weight: 0.1, 0.2, 0.4, 0.8, ... <10.0
 *              First line of output: fourth column is the optimized conw (not dark facet), all other lines include dark facet.
 */


// ReSharper disable All
#include "stdafx.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include "cc_config.h"

#include "resource.h"
#include "declarations.h"
#include "constants.h"
#include "globals.h"
#include "cuda.h"
#include <cuda_runtime.h>


#ifdef _WIN32
#include "boinc_win.h"
#include "Windows.h"
#include <Shlwapi.h>

#else
//#include "../win_build/config.h"
#include <cstdio>
#include <cctype>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <csignal>
//#include <io.h>
#include <unistd.h>
#include <limits>
#endif

#include "str_util.h"
#include "util.h"
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"
#include "start_CUDA.h"
#include "Version.h"
#include "arrayHelpers.hpp"
#include "LcHelpers.hpp"
#include "diagnostics.h"

#ifdef APP_GRAPHICS
#include "graphics2.h"
#include "uc2.h"
UC_SHMEM* shmem;
#endif

#if !defined _WIN32
#include <stdarg.h>

int fscanf_s(FILE* file, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int result = vfscanf(file, format, args);
    va_end(args);

    if (result == EOF) {
        fprintf(stderr, "\nError: reading input\n"); fflush(stderr); std::exit(2);
    }
    else if (result == 0) {
        fprintf(stderr, "\nError: input format mismatch\n"); fflush(stderr); std::exit(2);
    }
    return result;
}
#endif

using std::string;

#if _MSC_VER >= 1900 // Visual Studio 2015 or later
constexpr auto checkpoint_file = "period_search_state";
constexpr auto input_filename = "period_search_in";
constexpr auto output_filename = "period_search_out";
#else
const auto checkpoint_file = "period_search_state";
const auto input_filename = "period_search_in";
const auto output_filename = "period_search_out";
#endif

int DoCheckpoint(MFILE& mf, const int nlines, const int newConw, const double conwr) {
	string resolvedName;

	const auto f = fopen("temp", "w");
	// const auto f = fopen("temp", "e");
	if (!f) return 1;
	fprintf(f, "%d %d %.17g", nlines, newConw, conwr);
	fclose(f);

	auto retval = mf.flush();
	if (retval) return retval;
	boinc_resolve_filename_s(checkpoint_file, resolvedName);
	retval = boinc_rename("temp", resolvedName.c_str());
	if (retval) return retval;

	return 0;
}

#ifdef APP_GRAPHICS
void update_shmem() {
	if (!shmem) return;

	// always do this; otherwise a graphics app will immediately
	// assume we're not alive
	shmem->update_time = dtime();

	// Check whether a graphics app is running,
	// and don't bother updating shmem if so.
	// This doesn't matter here,
	// but may be worth doing if updating shmem is expensive.
	//
	if (shmem->countdown > 0) {
		// the graphics app sets this to 5 every time it renders a frame
		shmem->countdown--;
	}
	else {
		return;
	}
	shmem->fraction_done = boinc_get_fraction_done();
	shmem->cpu_time = boinc_worker_thread_cpu_time();;
	boinc_get_status(&shmem->status);
}
#endif

// NOTE: global parameters
int l_max, m_max, n_iter, last_call,
n_coef, num_fac, n_ph_par,
deallocate;

double o_chi_square, chi_square, a_lambda, a_lamda_incr, a_lamda_start, phi_0, scale,
d_area[MAX_N_FAC + 1],
f_c[MAX_N_FAC + 1][MAX_LM + 1], f_s[MAX_N_FAC + 1][MAX_LM + 1],
t_c[MAX_N_FAC + 1][MAX_LM + 1], t_s[MAX_N_FAC + 1][MAX_LM + 1],
d_sphere[MAX_N_FAC + 1][MAX_N_PAR + 1], d_g[MAX_N_FAC + 1][MAX_N_PAR + 1],
normal[MAX_N_FAC + 1][3], bl_matrix[4][4],
pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
d_bl_matrix[3][4][4];

auto cuda_device = -1;
APP_INIT_DATA aid;
const char project_name[] = "asteroidsathome";

/*--------------------------------------------------------------*/

int main(int argc, char** argv)
{
	int retval, nlines, checkpointExists, nStartFrom;  //int c, nchars = 0 // double fsize, fd;
	char inputPath[512], outputPath[512], chkptPath[512], buf[256];
	MFILE out;
	FILE* state, * infile;

	int i, j, l, m, k, n, nrows, onlyrel, ndata, k2, ndir, iTemp, nIterMin,
		ial0, ial0_abs, ia_beta_pole, ia_lambda_pole, ia_prd, ia_par[4], ia_cl,
		lcNumber, newConw;

	double startPeriod, periodStepCoef, endPeriod,
		startFrequency, frequencyStep, endFrequency,
		stopCondition;

	// resolve logical name first
	boinc_resolve_filename(input_filename, inputPath, sizeof(inputPath));

	auto gl = globals();
	auto res = PrepareLcData(gl, inputPath);
	if (res <= 0)
	{
		fprintf(stderr, "\nCouldn't find input file, resolved name %s.\n", inputPath);
		fflush(stderr);
	}

	/* Minimum JD*/
	double jdMin;
	/* Maximum JD*/
	double jdMax;
	/* Time in JD*/
	//double* tim;
	std::vector<double> tim(gl.maxDataPoints + 2, 0.0);
	/* Brightness*/
	std::vector<double> brightness(gl.maxDataPoints + 4, 0.0);
	/* Solar phase angle */
	std::vector<double> al(gl.Lcurves + 1, 0.0);
	/* Weights...*/
	std::vector<double> weightLc(gl.Lcurves + 1, 0.0);
	/* Ecliptic astronomical tempo-centric coordinates of the Sun in AU*/
	double e0[4];
	/* Ecliptic astronomical centric coordinates of the Earth in AU*/
	double e[4];
	/* Normalization of distance vectors*/
	std::vector<std::vector<double>> ee;
	init_matrix(ee, gl.maxDataPoints + 1, 3, 0.0);
	std::vector<std::vector<double>> ee0;
	init_matrix(ee0, gl.maxDataPoints + 1, 3, 0.0);

	double jd0, jd00, a0 = 1.05, b0 = 1.00, c0 = 0.95, a, b, cAxis, conw, conwR,
		cl, al0, al0Abs, ave, e0Len, elen, cosAlpha, dth, dph, rfit, escl,
		par[4];

	auto stringTemp = static_cast<char*>(malloc(MAX_LINE_LENGTH));

	std::vector<double> sig(gl.maxDataPoints + 4, 0.0);
	std::vector<double> cgFirst(gl.maxDataPoints + 2, 0.0);
	std::vector<double> t(MAX_N_FAC + 1, 0.0);
	std::vector<double> f(MAX_N_FAC + 1, 0.0);
	std::vector<double> at(MAX_N_FAC + 1, 0.0);
	std::vector<double> af(MAX_N_FAC + 1, 0.0);
	std::vector<int> ia(MAX_N_PAR + 1, 0);
	std::vector<std::vector<int>> ifp;
	init_matrix<int>(ifp, MAX_N_FAC + 1, 4 + 1, 0);

	double lambdaPole[N_POLES + 1] = { 0.0, 0.0, 90.0, 180.0, 270.0, 60.0, 180.0, 300.0, 60.0, 180.0, 300.0 };
	double betaPole[N_POLES + 1] = { 0.0, 0.0, 0.0, 0.0, 0.0, 60.0, 60.0, 60.0, -60.0, -60.0, -60.0 };

	ia_lambda_pole = ia_beta_pole = 1;

	BOINC_OPTIONS options;
	boinc_options_defaults(options);

#if defined _DEBUG
	// Initialize diagnostics with the memory leak check enabled
	boinc_init_diagnostics(BOINC_DIAG_DUMPCALLSTACKENABLED | BOINC_DIAG_HEAPCHECKENABLED | BOINC_DIAG_MEMORYLEAKCHECKENABLED 
		| BOINC_DIAG_REDIRECTSTDERR | BOINC_DIAG_TRACETOSTDERR);
#endif

	options.normal_thread_priority = true;
	retval = boinc_init_options(&options);

	if (retval) {
		fprintf(stderr, "%s boinc_init returned %d\n",
			boinc_msg_prefix(buf, sizeof(buf)), retval
			);
		exit(retval);
	}

	boinc_get_init_data(aid);

//#ifdef _WIN32
//	// -------------------
//	char buffer[MAX_PATH];
//	GetModuleFileName(NULL, buffer, MAX_PATH);
//	string::size_type pos = string(buffer).find_last_of("\\/");
//	auto result = string(buffer).substr(0, pos);
//	//--------------------------------------------
//#else
//	// open the input file (resolve logical name first)
//	//
//	boinc_resolve_filename(input_filename, inputPath, sizeof(inputPath));
//	infile = boinc_fopen(inputPath, "r");
//	if (!infile) {
//		fprintf(stderr,
//			"%s Couldn't find input file, resolved name %s.\n",
//			boinc_msg_prefix(buf, sizeof(buf)), inputPath
//		);
//		exit(-1);
//	}
//#endif

	// open the input file 
	infile = boinc_fopen(inputPath, "r");
	if (!infile) {
		fprintf(stderr,
			"%s Couldn't find input file, resolved name %s.\n",
			boinc_msg_prefix(buf, sizeof(buf)), inputPath
			);
		exit(-1);
	}

	// output file
	boinc_resolve_filename(output_filename, outputPath, sizeof(outputPath));
	//    out.open(output_path, "w");

		// See if there's a valid checkpoint file.
		// If so seek input file and truncate output file
		//
	boinc_resolve_filename(checkpoint_file, chkptPath, sizeof(chkptPath));
	state = boinc_fopen(chkptPath, "r");
	if (state) {
		n = fscanf(state, "%d %d %lf", &nlines, &newConw, &conwR);
		fclose(state);
	}
	if (state && n == 3) {
		checkpointExists = 1;
		nStartFrom = nlines + 1;
		retval = out.open(outputPath, "a");
	}
	else {
		checkpointExists = 0;
		nStartFrom = 1;
		retval = out.open(outputPath, "w");
	}
	if (retval) {
		fprintf(stderr, "%s APP: period_search output open failed:\n",
			boinc_msg_prefix(buf, sizeof(buf))
			);
		fprintf(stderr, "%s resolved name %s, retval %d\n",
			boinc_msg_prefix(buf, sizeof(buf)), outputPath, retval
			);
		perror("open");
		exit(1);
	}

#ifdef APP_GRAPHICS
	// create shared mem segment for graphics, and arrange to update it
	//
	//shmem = (UC_SHMEM*)boinc_graphics_make_shmem("uppercase", sizeof(UC_SHMEM));

	char shmemName[256];
	snprintf(shmemName, sizeof shmemName, "%s_%s", project_name, inputPath);

	shmem = (UC_SHMEM*)boinc_graphics_make_shmem(shmemName, sizeof(UC_SHMEM));
	if (!shmem) {
		fprintf(stderr, "%s failed to create shared mem segment\n",
			boinc_msg_prefix(buf, sizeof(buf))
			);
	}
	update_shmem();
	boinc_register_timer_callback(update_shmem);
#endif

    int err = 0;
    char* line = nullptr;

	// NOTE: Period interval (hours) fixed or free
	err = fscanf_s(infile, "%lf %lf %lf %d", &startPeriod, &periodStepCoef, &endPeriod, &ia_prd); 	line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Epoch of zero time t0
	err = fscanf_s(infile, "%lf", &jd00);                          	line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Initial fixed rotation angle fi0
	err = fscanf_s(infile, "%lf", &phi_0);                          	line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: The weight factor for conv. reg.
	err = fscanf_s(infile, "%lf", &conw);                           	line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Degree and order of the Laplace series
	err = fscanf_s(infile, "%d %d", &l_max, &m_max);                	line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Number of triangulation rows per octant
	err = fscanf_s(infile, "%d", &nrows);                               line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Initial guesses for phase funct. params.
	err = fscanf_s(infile, "%lf %d", &par[1], &ia_par[1]);              line = fgets(stringTemp, MAX_LINE_LENGTH, infile);
	err = fscanf_s(infile, "%lf %d", &par[2], &ia_par[2]);              line = fgets(stringTemp, MAX_LINE_LENGTH, infile);
	err = fscanf_s(infile, "%lf %d", &par[3], &ia_par[3]);              line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Initial Lambert coefficient (L - S = 1)
	err = fscanf_s(infile, "%lf %d", &cl, &ia_cl);                      line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Maximum number of iterations (when > 1) or  minimum difference in dev to stop (when < 1)
	err = fscanf_s(infile, "%lf", &stopCondition);                     line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Minimum number of iterations when stop_condition < 1
	err = fscanf_s(infile, "%d", &nIterMin);                          line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Multiplicative factor for Alamda
	err = fscanf_s(infile, "%lf", &a_lamda_incr);                       line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	// NOTE: Alamda initial value
	err = fscanf_s(infile, "%lf", &a_lamda_start);                      line = fgets(stringTemp, MAX_LINE_LENGTH, infile);

	if (boinc_is_standalone())
	{
		printf("\n%g  %g  %g  period start/step/stop (%d)\n", startPeriod, periodStepCoef, endPeriod, ia_prd);
		printf("%g epoch of zero time t0\n", jd00);
		printf("%g  initial fixed rotation angle fi0\n", phi_0);
		printf("%g  the weight factor for conv. reg.\n", conw);
		printf("%d %d  degree and order of the Laplace series\n", l_max, m_max);
		printf("%d  nr. of triangulation rows per octant\n", nrows);
		printf("%g %g %g  initial guesses for phase funct. params. (%d,%d,%d)\n", par[1], par[2], par[3], ia_par[1], ia_par[2], ia_par[3]);
		printf("%g  initial Lambert coeff. (L-S=1) (%d)\n", cl, ia_cl);
		printf("%g  stop condition\n", stopCondition);
		printf("%d  minimum number of iterations\n", nIterMin);
		printf("%g  Alamda multiplicative factor\n", a_lamda_incr);
		printf("%g  initial Alamda \n\n", a_lamda_start);
	}


	// NOTE: light curves + geometry file
	// NOTE: number of light curves and the first relative one
	int tlcurves;
	err = fscanf_s(infile, "%d", &tlcurves);

	if (boinc_is_standalone())
	{
		printf("%d  Number of light curves\n", gl.Lcurves);
	}

	ndata = 0;              /* total number of data */
	k2 = 0;                 /* index */
	al0 = al0Abs = PI;      /* the smallest solar phase angle */
	ial0 = ial0_abs = -1;   /* initialization, index of al0 */
	jdMin = 1e20;           /* initial minimum and maximum JD */
	jdMax = -1e40;
	onlyrel = 1;
	jd0 = jd00;
	a = a0; b = b0; cAxis = c0;

	//max_l_points = 0;
	/* loop over lightcurves */
	for (i = 1; i <= gl.Lcurves; i++)
	{
		ave = 0; /* average */
		err = fscanf_s(infile, "%d %d", &gl.Lpoints[i], &iTemp); /* points in this lightcurve */
		if (boinc_is_standalone())
		{
			printf("%d points in light curve[%d]\n", gl.Lpoints[i], i);
		}
		line = fgets(stringTemp, MAX_LINE_LENGTH, infile);
		gl.Inrel[i] = 1 - iTemp;
		if (gl.Inrel[i] == 0)
			onlyrel = 0;

		// NOTE: loop over one light curve
		for (j = 1; j <= gl.Lpoints[i]; j++)
		{
			ndata++;

			err = fscanf_s(infile, "%lf %lf", &tim[ndata], &brightness[ndata]); // NOTE: JD, brightness
			err = fscanf_s(infile, "%lf %lf %lf", &e0[1], &e0[2], &e0[3]);      // NOTE: ecliptic astronomical tempo-centric coordinates of the Sun in AU
			err = fscanf_s(infile, "%lf %lf %lf", &e[1], &e[2], &e[3]);         // NOTE: ecliptic astronomical centric coordinates of the Earth in AU

			// NOTE: selects the minimum and maximum JD
			if (tim[ndata] < jdMin) jdMin = tim[ndata];
			if (tim[ndata] > jdMax) jdMax = tim[ndata];

			// NOTE: normals of distance vectors
			e0Len = sqrt(dot_product(e0, e0));
			elen = sqrt(dot_product(e, e));
			ave += brightness[ndata];

			// NOTE: normalization of distance vectors
			for (k = 1; k <= 3; k++)
			{
				ee[ndata][k - 1] = e[k] / elen;
				ee0[ndata][k - 1] = e0[k] / e0Len;
			}

			if (j == 1)
			{
				cosAlpha = dot_product(e, e0) / (elen * e0Len);
				al[i] = acos(cosAlpha); /* solar phase angle */
				/* Find the smallest solar phase al0 (not important, just for info) */
				if (al[i] < al0)
				{
					al0 = al[i];
					ial0 = ndata;
				}
				if ((al[i] < al0Abs) && (gl.Inrel[i] == 0))
				{
					al0Abs = al[i];
					ial0_abs = ndata;
				}
			}
		} /* j, one lightcurve */

		ave /= gl.Lpoints[i];

		/* Mean brightness of lcurve
		   Use the mean brightness as 'sigma' to renormalize the
		   mean of each lightcurve to unity */

		for (j = 1; j <= gl.Lpoints[i]; j++)
		{
			k2++;
			sig[k2] = ave;
		}

	} /* i, all lightcurves */

	/* initiation of weights */
	for (i = 1; i <= gl.Lcurves; i++)
		weightLc[i] = -1;

	/* reads weights */
    auto scanResult = 0;
    while (true)
    {
        scanResult = fscanf(infile, "%d", &lcNumber);
        if (scanResult <= 0) break;
        scanResult = fscanf(infile, "%lf", &weightLc[lcNumber]);
        if (scanResult <= 0) break;
        if (boinc_is_standalone())
            printf("Weights: Light curve[%d], Weight[%g]\n", lcNumber, weightLc[lcNumber]);

        if (feof(infile)) break;
    }

	/* If input jd_0 <= 0 then the jd_0 is set to the day before the
	   lowest JD in the data */
	if (jd0 <= 0)
	{
		jd0 = static_cast<int>(jdMin);
		if (boinc_is_standalone())
		{
			printf("\nNew epoch of zero time  %f\n", jd0);
		}
	}

	/* loop over data - subtraction of jd_0 */
	for (i = 1; i <= ndata; i++)
	{
		tim[i] = tim[i] - jd0;
	}

	phi_0 = phi_0 * DEG2RAD;

	k = 0;
	for (i = 1; i <= gl.Lcurves; i++)
	{	    
		for (j = 1; j <= gl.Lpoints[i]; j++)
		{
			k++;
			if (weightLc[i] == -1)
				gl.Weight[k] = 1;
			else
				gl.Weight[k] = weightLc[i];
		}
	}

	for (i = 1; i <= 3; i++)
	{
		gl.Weight[k + i] = 1;
	}

	/* use calibrated data if possible */
	if (onlyrel == 0)
	{
		al0 = al0Abs;
		ial0 = ial0_abs;
	}

	/* Initial shape guess */
	rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
	escl = rfit / sqrt((a * b + b * cAxis + a * cAxis) / 3);
	if (onlyrel == 0)
		escl *= 0.8;
	a = a * escl;
	b = b * escl;
	cAxis = cAxis * escl;
	if (boinc_is_standalone())
	{
		printf("\nWild guess for initial sphere size is %g\n", rfit);
		printf("Suggested scaled a,b,c: %g %g %g\n\n", a, b, cAxis);
	}

	/* Convexity regularization: make one last 'lightcurve' that
	   consists of the three comps. of the residual nonconv. vect.
	   that should all be zero */
	//l_curves = l_curves + 1;
	//l_points[l_curves] = 3;
	//in_rel[l_curves] = 0;

	MakeConvexityRegularization(gl);

	// extract a --device option

	// NOTE: Applications that use coprocessors https://boinc.berkeley.edu/trac/wiki/AppCoprocessor
	// Some hosts have multiple GPUs. The BOINC client tells your application which instance to use.
	// Call boinc_get_init_data() to get an APP_INIT_DATA structure; the device number (0, 1, ...) is in the gpu_device_num field. Old (pre-7.0.12) clients pass the device number via a command-line argument, --device N.
	// In this case API_INIT_DATA::gpu_device_num will be -1, and your application must check its command-line args.
	if (aid.gpu_device_num >= 0)
	{
		cuda_device = aid.gpu_device_num;
	}
	else
	{
		for (auto ii = 0; ii < argc; ii++) {
			if (cuda_device < 0 && strcmp(argv[ii], "--device") == 0 && ii + 1 < argc)
				cuda_device = atoi(argv[++ii]);
		}
	}

	if (cuda_device < 0) cuda_device = 0;
	if (!checkpointExists)
	{
		fprintf(stderr, "BOINC client version %d.%d.%d\n", aid.major_version, aid.minor_version, aid.release);
		fprintf(stderr, "BOINC GPU type '%s', deviceId=%d, slot=%d\n", aid.gpu_type, cuda_device, aid.slot);

		int major, minor, build, revision;
#if defined __GNUC__
		GetVersionInfo(major, minor, build, revision);
		fprintf(stderr, "Application: period_search_BOINC_cuda12000\n");
#elif defined _WIN32
		TCHAR filepath[MAX_PATH]; // = getenv("_");
		GetModuleFileName(nullptr, filepath, MAX_PATH);
		auto filename = PathFindFileName(filepath);
		GetVersionInfo(filename, major, minor, build, revision);
		fprintf(stderr, "Application: %s\n", filename);
#endif
		fprintf(stderr, "Version: %d.%d.%d.%d\n", major, minor, build, revision);
	}

	retval = CUDAPrepare(cuda_device, betaPole, lambdaPole, par, cl, a_lamda_start, a_lamda_incr, ee, ee0, tim, phi_0, checkpointExists, ndata, gl);
	if (!retval)
	{
		fflush(stderr);
		exit(2);
	}

	/* optimization of the convexity weight **************************************************************/
	if (!checkpointExists)
	{
		conwR = conw / escl / escl;
		newConw = 0;
		boinc_fraction_done(0.0001); //signal start
#if _DEBUG
		std::time_t time = std::time(nullptr);   // get time now
		auto now = std::localtime(&time);
		printf("%02d:%02d:%02d | Fraction done: 0.0001%% (start signal)\n", now->tm_hour, now->tm_min, now->tm_sec);
		fprintf(stderr, "%02d:%02d:%02d | Fraction done: 0.0001%% (start signal)\n", now->tm_hour, now->tm_min, now->tm_sec);
		//fprintf(stderr, "WU cpu time: %f\n", aid.wu_cpu_time);
#endif

	}

	while ((newConw != 1) && ((conwR * escl * escl) < 10.0))
	{
		for (j = 1; j <= 3; j++)
		{
			ndata++;
			brightness[ndata] = 0;
			sig[ndata] = 1 / conwR;
		}

		/* the ordering of the coeffs. of the Laplace series */
		n_coef = 0; /* number of coeffs. */
		for (m = 0; m <= m_max; m++)
			for (l = m; l <= l_max; l++)
			{
				n_coef++;
				if (m != 0) n_coef++;
			}

		/*  Fix the directions of the triangle vertices of the Gaussian image sphere
			t = theta angle, f = phi angle */
		dth = PI / (2 * nrows); /* step in theta */
		k = 1;
		t[1] = 0;
		f[1] = 0;
		for (i = 1; i <= nrows; i++)
		{
			dph = PI / (2 * i); /* step in phi */
			for (j = 0; j <= 4 * i - 1; j++)
			{
				k++;
				t[k] = i * dth;
				f[k] = j * dph;
			}
		}

		/* go to south pole (in the same rot. order, not a mirror image) */
		for (i = nrows - 1; i >= 1; i--)
		{
			dph = PI / (2 * i);
			for (j = 0; j <= 4 * i - 1; j++)
			{
				k++;
				t[k] = PI - i * dth;
				f[k] = j * dph;
			}
		}

		ndir = k + 1; /* number of vertices */

		t[ndir] = PI;
		f[ndir] = 0;
		num_fac = 8 * nrows * nrows;

		if (num_fac > MAX_N_FAC)
		{
			fprintf(stderr, "\nError: Number of facets is greater than MAX_N_FAC!\n"); fflush(stderr); exit(2);
		}

		/* makes indices to triangle vertices */
		trifac(nrows, ifp);

		/* areas and normals of the triangulated Gaussian image sphere */
		areanorm(t, f, ndir, num_fac, ifp, at, af);

		/* Precompute some function values at each normal direction*/
		sphfunc(num_fac, at, af);

		ellfit(cgFirst, a, b, cAxis, num_fac, n_coef, at, af);

		startFrequency = 1 / startPeriod;
		endFrequency = 1 / endPeriod;
		frequencyStep = 0.5 / (jdMax - jdMin) / 24 * periodStepCoef;

		/* Give ia the value 0/1 if it's fixed/free */
		ia[n_coef + 1] = ia_beta_pole;
		ia[n_coef + 2] = ia_lambda_pole;
		ia[n_coef + 3] = ia_prd;
		/* phase function parameters */
		n_ph_par = 3;
		/* shape is free to be optimized */
		for (i = 1; i <= n_coef; i++)
			ia[i] = 1;
		/* The first shape param. fixed for relative br. fit */
		if (onlyrel == 1)
			ia[1] = 0;
		ia[n_coef + 3 + n_ph_par + 1] = ia_cl;
		/* Lommel-Seeliger part is fixed */
		ia[n_coef + 3 + n_ph_par + 2] = 0;

		if ((n_coef + 3 + n_ph_par + 1) > MAX_N_PAR)
		{
			fprintf(stderr, "\nError: Number of parameters is greater than MAX_N_PAR = %d\n", MAX_N_PAR); fflush(stderr); exit(2);
		}

		int n_max = static_cast<int>((startFrequency - endFrequency) / frequencyStep) + 1;
		if(n_max > MAX_N_FPOINTS) {
			fprintf(stderr, "\nError: Frequency points %d is greater than MAX_N_FPOINTS = %d!\n", n_max, MAX_N_FPOINTS); fflush(stderr); std::exit(2);
		}
		if(n_max < 0 || endFrequency > startFrequency || frequencyStep <= 0) {
			fprintf(stderr, "\nError: Invalid frequency points %d!\n", n_max); fflush(stderr); std::exit(2);
		}

		CUDAPrecalc(cuda_device, startFrequency, endFrequency, frequencyStep, stopCondition, nIterMin, &conwR, ndata, ia, ia_par, &newConw, cgFirst, sig, num_fac, 
			brightness, gl);

		ndata = ndata - 3;

		if (boinc_time_to_checkpoint() || boinc_is_standalone()) {
			retval = DoCheckpoint(out, 0, newConw, conwR); //zero lines
			if (retval)
			{
				fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof buf), retval);
				CUDAGlobalsFree();   // TODO: Double check why this was placed after exit() method?
				exit(retval);
			}
			boinc_checkpoint_completed();
		}
	}
	/*end optimizing conw *********************************************************************************/

	for (j = 1; j <= 3; j++)
	{
		ndata++;
		brightness[ndata] = 0;
		sig[ndata] = 1 / conwR;
	}

	/* the ordering of the coeffs. of the Laplace series */
	n_coef = 0; /* number of coeffs. */
	for (m = 0; m <= m_max; m++)
		for (l = m; l <= l_max; l++)
		{
			n_coef++;
			if (m != 0) n_coef++;
		}

	/*  Fix the directions of the triangle vertices of the Gaussian image sphere
		t = theta angle, f = phi angle */
	dth = PI / (2 * nrows); /* step in theta */
	k = 1;
	t[1] = 0;
	f[1] = 0;
	for (i = 1; i <= nrows; i++)
	{
		dph = PI / (2 * i); /* step in phi */
		for (j = 0; j <= 4 * i - 1; j++)
		{
			k++;
			t[k] = i * dth;
			f[k] = j * dph;
		}
	}

	/* go to south pole (in the same rot. order, not a mirror image) */
	for (i = nrows - 1; i >= 1; i--)
	{
		dph = PI / (2 * i);
		for (j = 0; j <= 4 * i - 1; j++)
		{
			k++;
			t[k] = PI - i * dth;
			f[k] = j * dph;
		}
	}

	ndir = k + 1; /* number of vertices */

	t[ndir] = PI;
	f[ndir] = 0;
	num_fac = 8 * nrows * nrows;

	if (num_fac > MAX_N_FAC)
	{
		fprintf(stderr, "\nError: Number of facets is greater than MAX_N_FAC!\n"); fflush(stderr); exit(2);
	}

	/* makes indices to triangle vertices */
	trifac(nrows, ifp);

	/* areas and normals of the triangulated Gaussian image sphere */
	areanorm(t, f, ndir, num_fac, ifp, at, af);

	/* Precompute some function values at each normal direction*/
	sphfunc(num_fac, at, af);

	ellfit(cgFirst, a, b, cAxis, num_fac, n_coef, at, af);

	startFrequency = 1 / startPeriod;
	endFrequency = 1 / endPeriod;
	frequencyStep = 0.5 / (jdMax - jdMin) / 24 * periodStepCoef;

	/* Give ia the value 0/1 if it's fixed/free */
	ia[n_coef + 1] = ia_beta_pole;
	ia[n_coef + 2] = ia_lambda_pole;
	ia[n_coef + 3] = ia_prd;
	/* phase function parameters */
	n_ph_par = 3;
	/* shape is free to be optimized */
	for (i = 1; i <= n_coef; i++)
		ia[i] = 1;
	/* The first shape param. fixed for relative br. fit */
	if (onlyrel == 1)
		ia[1] = 0;
	ia[n_coef + 3 + n_ph_par + 1] = ia_cl;
	/* Lommel-Seeliger part is fixed */
	ia[n_coef + 3 + n_ph_par + 2] = 0;

	if ((n_coef + 3 + n_ph_par + 1) > MAX_N_PAR)
	{
		fprintf(stderr, "\nError: Number of parameters is greater than MAX_N_PAR = %d\n", MAX_N_PAR); fflush(stderr); exit(2);
	}

	CUDAStart(cuda_device, nStartFrom, startFrequency, endFrequency, frequencyStep, stopCondition, nIterMin, conwR, 
		ndata, ia, ia_par, cgFirst, out, escl, sig, num_fac, brightness, gl);

	out.close();

	CUDAGlobalsFree();
	free(stringTemp);

	//auto clockRate = cudaDeviceGetAttribute()
	boinc_fraction_done(1);
#ifdef APP_GRAPHICS
	update_shmem();
#endif

//#ifdef _DEBUG
//	boinc_get_init_data(aid);
//	//fprintf(stderr, "WU cpu time: %f\n", aid.wu_cpu_time);
//#endif

	boinc_finish(0);
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR Args, int WinMode) {
	LPSTR command_line;
	char* argv[100];
	int argc;

	command_line = GetCommandLine();
	argc = parse_command_line(command_line, argv);
	return main(argc, argv);
}
#endif