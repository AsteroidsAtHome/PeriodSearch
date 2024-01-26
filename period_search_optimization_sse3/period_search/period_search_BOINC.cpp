/* This program take the input lightcurves, scans over the
   given period range and finds the best period+pole+shape+scattering
   solution. Shape is forgotten. The period, rms residual
   of the fit, and pole solution (lamdda, beta) are given to the output.
   Is starts from six initial poles and selects the best period.
   Reports also pole solution.

   syntax:
   period_search_BOINC

   output: period [hr], rms deviation, chi^2, dark facet [%] lambda_best beta_best

   8.11.2006

   new version of lightcurve files (new input lcs format)
   testing the dark facet, finding the optimal value for convexity weight: 0.1, 0.2, 0.4, 0.8, ... <10.0
   first line of output: fourth column is the optimized conw (not dark facet), all other lines include dark facet

   16.4.2012

   version for BOINC

*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
//#include <cstring>
//#include <memory.h>

#include "declarations.h"
#include "constants.h"
#include "globals.h"

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

#if defined _WIN32 
#if !defined WINXP
#include "Version.h"
#endif
#endif

#ifdef _WIN32
#include "boinc_win.h"
#include <Windows.h>
//#include <Shlwapi.h>
#else
#include "config.h" // <- freebsd
#include <cstdio>
#include <cctype>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <limits>
#include <iostream>
#endif

#ifdef __GNUC__
#include <filesystem>
#endif

#include "str_util.h"  // <- freebsd
#include "util.h"      // <- freebsd
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"
#include "systeminfo.h"

#ifdef APP_GRAPHICS
#include "graphics2.h"
#include "uc2.h"
UC_SHMEM* shmem;
#endif

using std::string;

constexpr auto checkpoint_file = "period_search_state";
constexpr auto input_filename = "period_search_in";
constexpr auto output_filename = "period_search_out";

int DoCheckpoint(MFILE& mf, int nlines, int newconw, double conwr, double sumdarkfacet, int testperiods)
{
	string resolvedName;

	const auto file = fopen("temp", "w");
	if (!file) return 1;
	fprintf(file, "%d %d %.17g %.17g %d", nlines, newconw, conwr, sumdarkfacet, testperiods);
	fclose(file);

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

/* global parameters */
int Lmax, Mmax, Niter, Lastcall,
Ncoef, Numfac, Lcurves, Nphpar,
Lpoints[MAX_LC + 1], Inrel[MAX_LC + 1],
Deallocate, n_iter;

double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,
Sclnw[MAX_LC + 1],
Yout[MAX_N_OBS + 1],
Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Blmat[4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4],
Weight[MAX_N_OBS + 1];

#ifdef __GNUC__
double Nor[3][MAX_N_FAC + 2] __attribute__((aligned(16))),
Area[MAX_N_FAC + 2] __attribute__((aligned(16))),
Darea[MAX_N_FAC + 2] __attribute__((aligned(16))),
Dg[MAX_N_FAC + 4][MAX_N_PAR + 10] __attribute__((aligned(16)));
#else
__declspec(align(16)) double Nor[3][MAX_N_FAC + 2], Area[MAX_N_FAC + 2], Darea[MAX_N_FAC + 2], Dg[MAX_N_FAC + 4][MAX_N_PAR + 10]; //Nor,Dg ARE ZERO INDEXED
#endif

int main(int argc, char **argv) {
	int /*c,*/ /*nchars = 0,*/ retval, nlines, ntestperiods, checkpoint_exists, n_start_from;
	//    double fsize, fd;
	char input_path[512], output_path[512], chkpt_path[512], buf[256];
	MFILE out;
	FILE* state, *infile;

	int i, j, l, m, k, n, nrows, ndata, k2, ndir, i_temp, onlyrel,
		n_iter_max, n_iter_min,
		*ia, ial0, ial0_abs, ia_beta_pole, ia_lambda_pole, ia_prd, ia_par[4], ia_cl, //ia is zero indexed
		lc_number,
		**ifp, new_conw, max_test_periods;

	double per_start, per_step_coef, per_end,
		freq, freq_start, freq_step, freq_end, jd_min, jd_max,
		dev_old, dev_new, iter_diff, iter_diff_max, stop_condition,
		totarea, sum, dark, dev_best, per_best, dark_best, la_tmp, be_tmp, la_best, be_best, //fraction_done,
		*t, *f, *at, *af, sum_dark_facet, ave_dark_facet;

	double jd_0, jd_00, conw, conw_r, a0 = 1.05, b0 = 1.00, c0 = 0.95, a, b, c_axis,
		prd, cl, al0, al0_abs, ave, e0len, elen, cos_alpha,
		dth, dph, rfit, escl,
		*brightness, e[4], e0[4], **ee,
		**ee0, *cg, *cg_first, **covar,
		**aalpha, *sig, chck[4],
		*tim, *al,
		beta_pole[N_POLES + 1], lambda_pole[N_POLES + 1], par[4], rchisq, *weight_lc;

	char *str_temp;

	str_temp = (char *)malloc(MAX_LINE_LENGTH);

	ee = matrix_double(MAX_N_OBS, 3);
	ee0 = matrix_double(MAX_N_OBS, 3);
	covar = aligned_matrix_double(MAX_N_PAR, MAX_N_PAR);
	aalpha = aligned_matrix_double(MAX_N_PAR, MAX_N_PAR + 2); //+2 due to mrqcof cycles
	ifp = matrix_int(MAX_N_FAC, 4);

	tim = vector_double(MAX_N_OBS);
	brightness = vector_double(MAX_N_OBS);
	sig = vector_double(MAX_N_OBS);
	cg = vector_double(MAX_N_PAR);
	cg_first = vector_double(MAX_N_PAR);
	t = vector_double(MAX_N_FAC);
	f = vector_double(MAX_N_FAC);
	at = vector_double(MAX_N_FAC);
	af = vector_double(MAX_N_FAC);

	ia = vector_int(MAX_N_PAR);

	lambda_pole[1] = 0;    beta_pole[1] = 0;
	lambda_pole[2] = 90;   beta_pole[2] = 0;
	lambda_pole[3] = 180;  beta_pole[3] = 0;
	lambda_pole[4] = 270;  beta_pole[4] = 0;
	lambda_pole[5] = 60;   beta_pole[5] = 60;
	lambda_pole[6] = 180;  beta_pole[6] = 60;
	lambda_pole[7] = 300;  beta_pole[7] = 60;
	lambda_pole[8] = 60;   beta_pole[8] = -60;
	lambda_pole[9] = 180;  beta_pole[9] = -60;
	lambda_pole[10] = 300; beta_pole[10] = -60;

	ia_lambda_pole = ia_beta_pole = 1;

	retval = boinc_init();
	if (retval)
	{
		fprintf(stderr, "%s boinc_init returned %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval);
		exit(retval);
	}

	// open the input file (resolve logical name first)
	//
	boinc_resolve_filename(input_filename, input_path, sizeof(input_path));
	infile = boinc_fopen(input_path, "r");
	if (!infile) {
		fprintf(stderr,
			"%s Couldn't find input file, resolved name %s.\n",
			boinc_msg_prefix(buf, sizeof(buf)), input_path
		);
		exit(-1);
	}

	// output file
	boinc_resolve_filename(output_filename, output_path, sizeof(output_path));
	//    out.open(output_path, "w");

		// See if there's a valid checkpoint file.
		// If so seek input file and truncate output file
		//
	boinc_resolve_filename(checkpoint_file, chkpt_path, sizeof(chkpt_path));
	state = boinc_fopen(chkpt_path, "r");
	if (state) {
		n = fscanf(state, "%d %d %lf %lf %d", &nlines, &new_conw, &conw_r, &sum_dark_facet, &ntestperiods);
		fclose(state);
	}
	if (state && n == 5) {
		checkpoint_exists = 1;
		n_start_from = nlines + 1;
		retval = out.open(output_path, "a");
	}
	else {
		checkpoint_exists = 0;
		n_start_from = 1;
		retval = out.open(output_path, "w");
	}
	if (retval) {
		fprintf(stderr, "%s APP: period_search output open failed:\n",
			boinc_msg_prefix(buf, sizeof(buf))
		);
		fprintf(stderr, "%s resolved name %s, retval %d\n",
			boinc_msg_prefix(buf, sizeof(buf)), output_path, retval
		);
		perror("open");
		exit(1);
	}

#ifdef APP_GRAPHICS
	// create shared mem segment for graphics, and arrange to update it
	//
	shmem = (UC_SHMEM*)boinc_graphics_make_shmem("uppercase", sizeof(UC_SHMEM));
	if (!shmem) {
		fprintf(stderr, "%s failed to create shared mem segment\n",
			boinc_msg_prefix(buf, sizeof(buf))
		);
	}
	update_shmem();
	boinc_register_timer_callback(update_shmem);
#endif


	/* period interval (hrs) fixed or free */
	fscanf(infile, "%lf %lf %lf %d", &per_start, &per_step_coef, &per_end, &ia_prd);          fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* epoch of zero time t0 */
	fscanf(infile, "%lf", &jd_00);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* initial fixed rotation angle fi0 */
	fscanf(infile, "%lf", &Phi_0);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* the weight factor for conv. reg. */
	fscanf(infile, "%lf", &conw);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* degree and order of the Laplace series */
	fscanf(infile, "%d %d", &Lmax, &Mmax);                        fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* nr. of triangulation rows per octant */
	fscanf(infile, "%d", &nrows);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* Initial guesses for phase funct. params. */
	fscanf(infile, "%lf %d", &par[1], &ia_par[1]);                fgets(str_temp, MAX_LINE_LENGTH, infile);
	fscanf(infile, "%lf %d", &par[2], &ia_par[2]);                fgets(str_temp, MAX_LINE_LENGTH, infile);
	fscanf(infile, "%lf %d", &par[3], &ia_par[3]);                fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* Initial Lambert coeff. (L-S=1) */
	fscanf(infile, "%lf %d", &cl, &ia_cl);                        fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* maximum number of iterations (when > 1) or
	   minimum difference in dev to stop (when < 1) */
	fscanf(infile, "%lf", &stop_condition);                       fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* minimum number of iterations when stop_condition < 1 */
	fscanf(infile, "%d", &n_iter_min);                            fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* multiplicative factor for Alamda */
	fscanf(infile, "%lf", &Alamda_incr);                          fgets(str_temp, MAX_LINE_LENGTH, infile);
	/* Alamda initial value*/
	fscanf(infile, "%lf", &Alamda_start);                         fgets(str_temp, MAX_LINE_LENGTH, infile);

	if (boinc_is_standalone())
	{
		printf("\n%g  %g  %g  period start/step/stop (%d)\n", per_start, per_step_coef, per_end, ia_prd);
		printf("%g epoch of zero time t0\n", jd_00);
		printf("%g  initial fixed rotation angle fi0\n", Phi_0);
		printf("%g  the weight factor for conv. reg.\n", conw);
		printf("%d %d  degree and order of the Laplace series\n", Lmax, Mmax);
		printf("%d  nr. of triangulation rows per octant\n", nrows);
		printf("%g %g %g  initial guesses for phase funct. params. (%d,%d,%d)\n", par[1], par[2], par[3], ia_par[1], ia_par[2], ia_par[3]);
		printf("%g  initial Lambert coeff. (L-S=1) (%d)\n", cl, ia_cl);
		printf("%g  stop condition\n", stop_condition);
		printf("%d  minimum number of iterations\n", n_iter_min);
		printf("%g  Alamda multiplicative factor\n", Alamda_incr);
		printf("%g  initial Alamda \n\n", Alamda_start);
	}


	/* lightcurves + geometry file */
	/* number of lightcurves and the first realtive one */
	fscanf(infile, "%d", &Lcurves);

	if (Lcurves > MAX_LC)
	{
		fprintf(stderr, "\nError: Number of lcs  is greater than MAX_LC = %d\n", MAX_LC); fflush(stderr); exit(2);
	}

	al = vector_double(Lcurves);
	weight_lc = vector_double(Lcurves);

	ndata = 0; /* total number of data */
	k2 = 0;   /* index */
	al0 = al0_abs = PI; /* the smallest solar phase angle */
	ial0 = ial0_abs = -1; /* initialization, index of al0 */
	jd_min = 1e20; /* initial minimum and minimum JD */
	jd_max = -1e40;
	onlyrel = 1;
	jd_0 = jd_00;
	a = a0; b = b0; c_axis = c0;

	/* loop over lightcurves */
	for (i = 1; i <= Lcurves; i++)
	{
		ave = 0; /* average */
		fscanf(infile, "%d %d", &Lpoints[i], &i_temp); /* points in this lightcurve */
		fgets(str_temp, MAX_LINE_LENGTH, infile);
		Inrel[i] = 1 - i_temp;
		if (Inrel[i] == 0)
			onlyrel = 0;

		if (Lpoints[i] > POINTS_MAX)
		{
			fprintf(stderr, "\nError: Number of lc points is greater than POINTS_MAX = %d\n", POINTS_MAX); fflush(stderr); exit(2);
		}

		/* loop over one lightcurve */
		for (j = 1; j <= Lpoints[i]; j++)
		{
			ndata++;

			if (ndata > MAX_N_OBS)
			{
				fprintf(stderr, "\nError: Number of data is greater than MAX_N_OBS = %d\n", MAX_N_OBS); fflush(stderr); exit(2);
			}

			auto min_double = std::numeric_limits<double>::min();
			fscanf(infile, "%lf %lf", &tim[ndata], &brightness[ndata]); /* JD, brightness */
			if (tim[ndata] < min_double)
			{
				tim[ndata] = min_double;
			}

			if (brightness[ndata] < min_double)
			{
				brightness[ndata] = min_double;
			}
			fscanf(infile, "%lf %lf %lf", &e0[1], &e0[2], &e0[3]); /* ecliptic astr_tempocentric coord. of the Sun in AU */
			fscanf(infile, "%lf %lf %lf", &e[1], &e[2], &e[3]); /* ecliptic astrocentric coord. of the Earth in AU */

		/* selects the minimum and maximum JD */
			if (tim[ndata] < jd_min) jd_min = tim[ndata];
			if (tim[ndata] > jd_max) jd_max = tim[ndata];

			/* normals of distance vectors */
			e0len = sqrt(e0[1] * e0[1] + e0[2] * e0[2] + e0[3] * e0[3]);
			elen = sqrt(e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);

			ave += brightness[ndata];

			/* normalization of distance vectors */
			for (k = 1; k <= 3; k++)
			{
				ee[ndata][k] = e[k] / elen;
				ee0[ndata][k] = e0[k] / e0len;
			}

			if (j == 1)
			{
				cos_alpha = dot_product(e, e0) / (elen * e0len);
				al[i] = acos(cos_alpha); /* solar phase angle */
				/* Find the smallest solar phase al0 (not important, just for info) */
				if (al[i] < al0)
				{
					al0 = al[i];
					ial0 = ndata;
				}
				if ((al[i] < al0_abs) && (Inrel[i] == 0))
				{
					al0_abs = al[i];
					ial0_abs = ndata;
				}
			}
		} /* j, one lightcurve */

		ave /= Lpoints[i];

		/* Mean brightness of lcurve
		   Use the mean brightness as 'sigma' to renormalize the
		   mean of each lightcurve to unity */

		for (j = 1; j <= Lpoints[i]; j++)
		{
			k2++;
			sig[k2] = ave;
		}

	} /* i, all lightcurves */

	/* initiation of weights */
	for (i = 1; i <= Lcurves; i++)
		weight_lc[i] = -1;

	/* reads weights */
	auto scanResult = 0;
	while(true)
	{
		scanResult = fscanf(infile, "%d", &lc_number);
		if (scanResult <= 0) break;
		scanResult = fscanf(infile, "%lf", &weight_lc[lc_number]);
		if (scanResult <= 0) break;
		if (boinc_is_standalone())
			printf("weights %d %g\n", lc_number, weight_lc[lc_number]);

		if (feof(infile)) break;
	}

	/* If input jd_0 <= 0 then the jd_0 is set to the day before the
	   lowest JD in the data */
	if (jd_0 <= 0)
	{
		jd_0 = (int)jd_min;
		if (boinc_is_standalone())
			printf("\nNew epoch of zero time  %f\n", jd_0);
	}

	/* loop over data - subtraction of jd_0 */
	for (i = 1; i <= ndata; i++)
		tim[i] = tim[i] - jd_0;

	Phi_0 = Phi_0 * DEG2RAD;

	k = 0;
	for (i = 1; i <= Lcurves; i++)
		for (j = 1; j <= Lpoints[i]; j++)
		{
			k++;
			if (weight_lc[i] == -1)
				Weight[k] = 1;
			else
				Weight[k] = weight_lc[i];
		}

	for (i = 1; i <= 3; i++)
		Weight[k + i] = 1;

	/* use calibrated data if possible */
	if (onlyrel == 0)
	{
		al0 = al0_abs;
		ial0 = ial0_abs;
	}

	/* Initial shape guess */
	rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
	escl = rfit / sqrt((a * b + b * c_axis + a * c_axis) / 3);
	if (onlyrel == 0)
		escl *= 0.8;
	a = a * escl;
	b = b * escl;
	c_axis = c_axis * escl;
	if (boinc_is_standalone())
	{
		printf("\nWild guess for initial sphere size is %g\n", rfit);
		printf("Suggested scaled a,b,c: %g %g %g\n\n", a, b, c_axis);
	}

	/* Convexity regularization: make one last 'lightcurve' that
	   consists of the three comps. of the residual nonconv. vect.
	   that should all be zero */
	Lcurves = Lcurves + 1;
	Lpoints[Lcurves] = 3;
	Inrel[Lcurves] = 0;

	/* optimization of the convexity weight **************************************************************/
	APP_INIT_DATA aid;
	boinc_get_init_data(aid);
	if (!checkpoint_exists)
	{
		conw_r = conw / escl / escl;
		new_conw = 0;

		fprintf(stderr, "BOINC client version %d.%d.%d\n", aid.major_version, aid.minor_version, aid.release);

int major, minor, build, revision;
#if defined _WIN32 && !WINXP && !defined __GNUC__
		TCHAR filepath[MAX_PATH]; // = getenv("_");
		GetModuleFileName(nullptr, filepath, MAX_PATH);
		auto filename = PathFindFileName(filepath);
		GetVersionInfo(filename, major, minor, build, revision);
		std::cerr << "Application: " << filename << std::endl;
#elif defined __GNUC__
		GetVersionInfo(major, minor, build, revision);
		#if !defined __APPLE__
			auto path = std::filesystem::current_path();
		#endif
		std::cerr << "Application: " << argv[0] << std::endl;
#endif
		std::cerr << "Version: " << major << "." << minor << "." << build << "." << revision << std::endl;
	}

	std::cerr << "CPU: " << GetCpuInfo() << std::endl;
	std::cerr << "Target instruction set: " << GetTargetInstructionSet() << std::endl;
	std::cerr << "RAM: " << getTotalSystemMemory() << "GB" << std::endl;

	while ((new_conw != 1) && ((conw_r * escl * escl) < 10.0))
	{
		for (j = 1; j <= 3; j++)
		{
			ndata++;
			brightness[ndata] = 0;
			sig[ndata] = 1 / conw_r;
		}

		/* the ordering of the coeffs. of the Laplace series */
		Ncoef = 0; /* number of coeffs. */
		for (m = 0; m <= Mmax; m++)
			for (l = m; l <= Lmax; l++)
			{
				Ncoef++;
				if (m != 0) Ncoef++;
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
		Numfac = 8 * nrows * nrows;

		if (Numfac > MAX_N_FAC)
		{
			fprintf(stderr, "\nError: Number of facets is greater than MAX_N_FAC!\n"); fflush(stderr); exit(2);
		}

		/* makes indices to triangle vertices */
		trifac(nrows, ifp);
		/* areas and normals of the triangulated Gaussian image sphere */
		areanorm(t, f, ndir, Numfac, ifp, at, af);
		/* Precompute some function values at each normal direction*/
		sphfunc(Numfac, at, af);

		ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

		freq_start = 1 / per_start;
		freq_end = 1 / per_end;
		freq_step = 0.5 / (jd_max - jd_min) / 24 * per_step_coef;

		/* Give ia the value 0/1 if it's fixed/free */
		ia[Ncoef + 1 - 1] = ia_beta_pole;
		ia[Ncoef + 2 - 1] = ia_lambda_pole;
		ia[Ncoef + 3 - 1] = ia_prd;
		/* phase function parameters */
		Nphpar = 3;
		/* shape is free to be optimized */
		for (i = 0; i < Ncoef; i++)
			ia[i] = 1;
		/* The first shape param. fixed for relative br. fit */
		if (onlyrel == 1)
			ia[0] = 0;
		ia[Ncoef + 3 + Nphpar + 1 - 1] = ia_cl;
		/* Lommel-Seeliger part is fixed */
		ia[Ncoef + 3 + Nphpar + 2 - 1] = 0;

		if ((Ncoef + 3 + Nphpar + 1) > MAX_N_PAR)
		{
			fprintf(stderr, "\nError: Number of parameters is greater than MAX_N_PAR = %d\n", MAX_N_PAR); fflush(stderr); exit(2);
		}

		max_test_periods = 10;
		ave_dark_facet = 0.0;
		n_iter = (int)((freq_start - freq_end) / freq_step) + 1;
		if (n_iter < max_test_periods)
			max_test_periods = n_iter;

		if (checkpoint_exists)
		{
			n = ntestperiods + 1;
			checkpoint_exists = 0; //reset for next loop
		}
		else
		{
			sum_dark_facet = 0.0;
			n = 1;
		}

		for (; n <= max_test_periods; n++)
		{
			boinc_fraction_done(n / 10000.0 / max_test_periods);

			freq = freq_start - (n - 1) * freq_step;

			/* initial poles */
			per_best = dark_best = la_best = be_best = 0;
			dev_best = 1e40;
			for (m = 1; m <= N_POLES; m++)
			{
				prd = 1 / freq;

				/* starts from the initial ellipsoid */
				for (i = 1; i <= Ncoef; i++)
					cg[i] = cg_first[i];

				cg[Ncoef + 1] = beta_pole[m];
				cg[Ncoef + 2] = lambda_pole[m];

				/* The formulas use beta measured from the pole */
				cg[Ncoef + 1] = 90 - cg[Ncoef + 1];
				/* conversion of lambda, beta to radians */
				cg[Ncoef + 1] = DEG2RAD * cg[Ncoef + 1];
				cg[Ncoef + 2] = DEG2RAD * cg[Ncoef + 2];

				/* Use omega instead of period */
				cg[Ncoef + 3] = 24 * 2 * PI / prd;

				for (i = 1; i <= Nphpar; i++)
				{
					cg[Ncoef + 3 + i] = par[i];
					ia[Ncoef + 3 + i - 1] = ia_par[i];
				}
				/* Lommel-Seeliger part */
				cg[Ncoef + 3 + Nphpar + 2] = 1;
				/* Use logarithmic formulation for Lambert to keep it positive */
				cg[Ncoef + 3 + Nphpar + 1] = log(cl);

				/* Levenberg-Marquardt loop */
				n_iter_max = 0;
				iter_diff_max = -1;
				rchisq = -1;
				if (stop_condition > 1)
				{
					n_iter_max = (int)stop_condition;
					iter_diff_max = 0;
					n_iter_min = 0; /* to not overwrite the n_iter_max value */
				}
				if (stop_condition < 1)
				{
					n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
					iter_diff_max = stop_condition;
				}
				Alamda = -1;
				Niter = 0;
				iter_diff = 1e40;
				dev_old = 1e30;
				dev_new = 0;
				Lastcall = 0;

				while (((Niter < n_iter_max) && (iter_diff > iter_diff_max)) || (Niter < n_iter_min))
				{
					mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha);
					Niter++;

					if ((Niter == 1) || (Chisq < Ochisq))
					{
						Ochisq = Chisq;
						curv(cg);
						for (i = 1; i <= 3; i++)
						{
							chck[i] = 0;
							for (j = 1; j <= Numfac; j++)
								chck[i] = chck[i] + Area[j - 1] * Nor[i - 1][j - 1];
						}
						rchisq = Chisq - (pow(chck[1], 2) + pow(chck[2], 2) + pow(chck[3], 2)) * pow(conw_r, 2);
					}
					dev_new = sqrt(rchisq / (ndata - 3));
					/* only if this step is better than the previous,
					   1e-10 is for numeric errors */
					if (dev_old - dev_new > 1e-10)
					{
						iter_diff = dev_old - dev_new;
						dev_old = dev_new;
					}
				}

				/* deallocates variables used in mrqmin */
				Deallocate = 1;
				mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha);
				Deallocate = 0;

				totarea = 0;
				for (i = 1; i <= Numfac; i++)
					totarea = totarea + Area[i - 1];
				sum = pow(chck[1], 2) + pow(chck[2], 2) + pow(chck[3], 2);
				dark = sqrt(sum);

				/* period solution */
				prd = 2 * PI / cg[Ncoef + 3];

				/* pole solution */
				la_tmp = RAD2DEG * cg[Ncoef + 2];
				be_tmp = 90 - RAD2DEG * cg[Ncoef + 1];

				if (dev_new < dev_best)
				{
					dev_best = dev_new;
					per_best = prd;
					dark_best = dark / totarea * 100;
					la_best = la_tmp;
					be_best = be_tmp;
				}
			} /* pole loop */

			if (la_best < 0)
				la_best += 360;

#ifdef __GNUC__
			if (std::isnan(dark_best) == 1)
				dark_best = 1.0;
#else
			if (_isnan(dark_best) == 1)
				dark_best = 1.0;
#endif

			sum_dark_facet = sum_dark_facet + dark_best;

			if (boinc_time_to_checkpoint() || boinc_is_standalone()) {
				retval = DoCheckpoint(out, 0, new_conw, conw_r, sum_dark_facet, n); //zero lines
				if (retval)
				{
					fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval);
				}

				boinc_checkpoint_completed();
			}

		} /* period loop */

		ave_dark_facet = sum_dark_facet / max_test_periods;

		if (ave_dark_facet < 1.0)
			new_conw = 1; /* new correct conwexity weight */
		if (ave_dark_facet >= 1.0)
			conw_r = conw_r * 2; /* still not good */
		ndata = ndata - 3;


		if (boinc_time_to_checkpoint() || boinc_is_standalone()) {
			retval = DoCheckpoint(out, 0, new_conw, conw_r, 0.0, 0); //zero lines,zero sum dark facets, zero testperiods
			if (retval)
			{
				fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval);
			}

			boinc_checkpoint_completed();
		}

	}
	/*end optimizing conw *********************************************************************************/


	for (j = 1; j <= 3; j++)
	{
		ndata++;
		brightness[ndata] = 0;
		sig[ndata] = 1 / conw_r;
	}

	/* the ordering of the coeffs. of the Laplace series */
	Ncoef = 0; /* number of coeffs. */
	for (m = 0; m <= Mmax; m++)
		for (l = m; l <= Lmax; l++)
		{
			Ncoef++;
			if (m != 0) Ncoef++;
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
	Numfac = 8 * nrows * nrows;

	if (Numfac > MAX_N_FAC)
	{
		fprintf(stderr, "\nError: Number of facets is greater than MAX_N_FAC!\n"); fflush(stderr); exit(2);
	}

	/* makes indices to triangle vertices */
	trifac(nrows, ifp);
	/* areas and normals of the triangulated Gaussian image sphere */
	areanorm(t, f, ndir, Numfac, ifp, at, af);
	/* Precompute some function values at each normal direction*/
	sphfunc(Numfac, at, af);

	ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

	freq_start = 1 / per_start;
	freq_end = 1 / per_end;
	freq_step = 0.5 / (jd_max - jd_min) / 24 * per_step_coef;

	/* Give ia the value 0/1 if it's fixed/free */
	ia[Ncoef + 1 - 1] = ia_beta_pole;
	ia[Ncoef + 2 - 1] = ia_lambda_pole;
	ia[Ncoef + 3 - 1] = ia_prd;
	/* phase function parameters */
	Nphpar = 3;
	/* shape is free to be optimized */
	for (i = 0; i < Ncoef; i++)
		ia[i] = 1;
	/* The first shape param. fixed for relative br. fit */
	if (onlyrel == 1)
		ia[0] = 0;
	ia[Ncoef + 3 + Nphpar + 1 - 1] = ia_cl;
	/* Lommel-Seeliger part is fixed */
	ia[Ncoef + 3 + Nphpar + 2 - 1] = 0;

	if ((Ncoef + 3 + Nphpar + 1) > MAX_N_PAR)
	{
		fprintf(stderr, "\nError: Number of parameters is greater than MAX_N_PAR = %d\n", MAX_N_PAR); fflush(stderr); exit(2);
	}

	for (n = n_start_from; n <= (int)((freq_start - freq_end) / freq_step) + 1; n++)
	{
		auto fraction_done = n / (((freq_start - freq_end) / freq_step) + 1);
		boinc_fraction_done(fraction_done);

#ifdef _DEBUG
		auto fraction = fraction_done * 100;
		auto time = std::time(nullptr);   // get time now
		auto now = std::localtime(&time);

		printf("%02d:%02d:%02d | Fraction done: %.3f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
		fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.3f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
#endif

		freq = freq_start - (n - 1) * freq_step;

		/* initial poles */
		per_best = dark_best = la_best = be_best = 0;
		dev_best = 1e40;
		for (m = 1; m <= N_POLES; m++)
		{
			prd = 1 / freq;
			/* starts from the initial ellipsoid */
			for (i = 1; i <= Ncoef; i++)
				cg[i] = cg_first[i];

			cg[Ncoef + 1] = beta_pole[m];
			cg[Ncoef + 2] = lambda_pole[m];

			/* The formulas use beta measured from the pole */
			cg[Ncoef + 1] = 90 - cg[Ncoef + 1];
			/* conversion of lambda, beta to radians */
			cg[Ncoef + 1] = DEG2RAD * cg[Ncoef + 1];
			cg[Ncoef + 2] = DEG2RAD * cg[Ncoef + 2];

			/* Use omega instead of period */
			cg[Ncoef + 3] = 24 * 2 * PI / prd;

			for (i = 1; i <= Nphpar; i++)
			{
				cg[Ncoef + 3 + i] = par[i];
				ia[Ncoef + 3 + i - 1] = ia_par[i];
			}
			/* Lommel-Seeliger part */
			cg[Ncoef + 3 + Nphpar + 2] = 1;
			/* Use logarithmic formulation for Lambert to keep it positive */
			cg[Ncoef + 3 + Nphpar + 1] = log(cl);

			/* Levenberg-Marquardt loop */
			n_iter_max = 0;
			iter_diff_max = -1;
			rchisq = -1;
			if (stop_condition > 1)
			{
				n_iter_max = (int)stop_condition;
				iter_diff_max = 0;
				n_iter_min = 0; /* to not overwrite the n_iter_max value */
			}
			if (stop_condition < 1)
			{
				n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
				iter_diff_max = stop_condition;
			}
			Alamda = -1;
			Niter = 0;
			iter_diff = 1e40;
			dev_old = 1e30;
			dev_new = 0;
			Lastcall = 0;

			while (((Niter < n_iter_max) && (iter_diff > iter_diff_max)) || (Niter < n_iter_min))
			{
				mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha);
				Niter++;

				if ((Niter == 1) || (Chisq < Ochisq))
				{
					Ochisq = Chisq;
					curv(cg);
					for (i = 1; i <= 3; i++)
					{
						chck[i] = 0;
						for (j = 1; j <= Numfac; j++)
							chck[i] = chck[i] + Area[j - 1] * Nor[i - 1][j - 1];
					}
					rchisq = Chisq - (pow(chck[1], 2) + pow(chck[2], 2) + pow(chck[3], 2)) * pow(conw_r, 2);
				}
				dev_new = sqrt(rchisq / (ndata - 3));
				/* only if this step is better than the previous,
				   1e-10 is for numeric errors */
				if (dev_old - dev_new > 1e-10)
				{
					iter_diff = dev_old - dev_new;
					dev_old = dev_new;
				}
			}

			/* deallocates variables used in mrqmin */
			Deallocate = 1;
			mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha);
			Deallocate = 0;

			totarea = 0;
			for (i = 1; i <= Numfac; i++)
				totarea = totarea + Area[i - 1];
			sum = pow(chck[1], 2) + pow(chck[2], 2) + pow(chck[3], 2);
			dark = sqrt(sum);

			/* period solution */
			prd = 2 * PI / cg[Ncoef + 3];

			/* pole solution */
			la_tmp = RAD2DEG * cg[Ncoef + 2];
			be_tmp = 90 - RAD2DEG * cg[Ncoef + 1];

			if (dev_new < dev_best)
			{
				dev_best = dev_new;
				per_best = prd;
				dark_best = dark / totarea * 100;
				la_best = la_tmp;
				be_best = be_tmp;
			}
		} /* pole loop */

		if (la_best < 0)
			la_best += 360;

#ifdef __GNUC__
		if (std::isnan(dark_best) == 1)
			dark_best = 1.0;
#else
		if (_isnan(dark_best) == 1)
			dark_best = 1.0;
#endif

		/* output file */
		if (n == 1)
		{
			out.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * per_best, dev_best, dev_best * dev_best * (ndata - 3), conw_r * escl * escl, round(la_best), round(be_best));
		}
		else
		{
			out.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * per_best, dev_best, dev_best * dev_best * (ndata - 3), dark_best, round(la_best), round(be_best));
		}

		if (boinc_time_to_checkpoint() || boinc_is_standalone())
		{
			retval = DoCheckpoint(out, n, new_conw, conw_r, 0.0, 0);
			if (retval)
			{
				fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval);
			}

			boinc_checkpoint_completed();
		}
	} /* period loop */

	out.close();

	deallocate_matrix_double(ee, MAX_N_OBS);
	deallocate_matrix_double(ee0, MAX_N_OBS);
	aligned_deallocate_matrix_double(covar, MAX_N_PAR);
	aligned_deallocate_matrix_double(aalpha, MAX_N_PAR);
	deallocate_matrix_int(ifp, MAX_N_FAC);

	deallocate_vector(tim);
	deallocate_vector(brightness);
	deallocate_vector(sig);
	deallocate_vector(cg);
	deallocate_vector(cg_first);
	deallocate_vector(t);
	deallocate_vector(f);
	deallocate_vector(at);
	deallocate_vector(af);
	deallocate_vector(ia);
	deallocate_vector(al);
	deallocate_vector(weight_lc);
	free(str_temp);

	boinc_fraction_done(1);
#ifdef APP_GRAPHICS
	update_shmem();
#endif
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

