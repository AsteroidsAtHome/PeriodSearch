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

// ReSharper disable CppClangTidyCertErr33C
// ReSharper disable CppClangTidyPerformanceAvoidEndl
// ReSharper disable CppClangTidyConcurrencyMtUnsafe
// ReSharper disable CppClangTidyCertErr34C
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#if defined _DEBUG
// #include <ctime>
#include <time.h>
#endif

#include "declarations.h"
#include "constants.h"
#include "globals.h"

#ifdef _WIN32
#include "boinc_win.h"
#include <Shlwapi.h>
#include "WinBase.h"
#else
#include "config.h"
#include <cstdio>
#include <cctype>
// #include <ctime>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <iostream>
#endif

#ifdef __GNUC__
#include <filesystem>
#endif

#include "str_util.h"
#include "util.h"
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"
#include "arrayHelpers.hpp"
#include "systeminfo.h"
#include "Enums.h"
#include "CalcStrategy.hpp"
#include "CalcStrategyNone.hpp"
#include "LcHelpers.hpp"
#include "SIMDHelpers.h"
#include "benchmark.hpp"

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

CalcContext calcCtx(std::allocate_shared<CalcStrategyNone>(AlignedAllocator<CalcStrategyNone>(64)));
SIMDSupport CPUopt;

constexpr auto checkpoint_file = "period_search_state";
constexpr auto input_filename = "period_search_in";
constexpr auto output_filename = "period_search_out";

int DoCheckpoint(MFILE& mf, const int nlines, const int newconw, const double conwr, const double sumdarkfacet, const int testperiods)
{
    std::string resolvedName;

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

//#if defined __GNUC__
//// Helper function to allocate aligned memory
//void* allocate_aligned_memory(std::size_t alignment, std::size_t size) {
//    void* ptr = nullptr;
//    if (posix_memalign(&ptr, alignment, size) != 0) {
//        throw std::bad_alloc();
//    }
//    return ptr;
//}
//
//// Wrapper function to create an aligned std::vector
//std::vector<double> create_aligned_vector(std::size_t size, std::size_t alignment = 64)
//{
//    double* aligned_memory = static_cast<double*>(allocate_aligned_memory(alignment, size * sizeof(double)));
//    return std::vector<double>(aligned_memory, aligned_memory + size);
//}
//#endif


/* global parameters */
int Lmax, Mmax, Niter, Lastcall,
Ncoef, Numfac, Nphpar,
Deallocate, n_iter;

double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,

Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Blmat[4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4];

std::vector<double> atry;
std::vector<double> beta;
std::vector<double> da;

// NOTE: RPi related:
//void blinkLed(int count) {
//	for (int i = 0; i < count; i++) {
//		digitalWrite(LED, HIGH);  // On
//		delay(150); // ms
//		digitalWrite(LED, LOW);	  // Off
//		delay(150);
//	}
//}

int main(int argc, char** argv)
{
    int nlines = 0, ntestperiods, checkpoint_exists, n_start_from;
    char input_path[512], output_path[512], chkpt_path[512], buf[256];
    MFILE out;

    int i, j, l, m, k, n = 0, nrows, ndir, i_temp,
        n_iter_max, n_iter_min,
        ia_prd, ia_par[4]{}, ia_cl,
        lc_number,
        new_conw, max_test_periods,
        ma = 0;

    double per_start, per_step_coef, per_end,
        freq, freq_start, freq_step, freq_end,
        dev_old, dev_new, iter_diff, iter_diff_max, stop_condition,
        totarea, sum, dark, dev_best, per_best, dark_best, la_tmp, be_tmp, la_best, be_best, fraction_done,
        sum_dark_facet = 0.0, ave_dark_facet;

    double jd_00, conw, conw_r, a0 = 1.05, b0 = 1.00, c0 = 0.95,
        prd, cl, e0len, elen, cos_alpha,
        dth, dph, rfit, escl,
        chck[4]{},
        par[4]{}, rchisq;

    auto* str_temp = static_cast<char*>(malloc(MAX_LINE_LENGTH));

    double lambda_pole[N_POLES + 1] = { 0.0, 0.0, 90.0, 180.0, 270.0, 60.0, 180.0, 300.0, 60.0, 180.0, 300.0 };
    double beta_pole[N_POLES + 1] = { 0.0, 0.0, 0.0, 0.0, 0.0, 60.0, 60.0, 60.0, -60.0, -60.0, -60.0 };

    int ia_lambda_pole = 1;
    int ia_beta_pole = 1;

    //wiringPiSetupSys();
    //pinMode(LED, OUTPUT);

    for (i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--benchmark") == 0)
        {
            free(str_temp);
            return RunBenchmark();
        }
    }

    int retval = boinc_init();
    if (retval)
    {
        fprintf(stderr, "%s boinc_init returned %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval);
        std::exit(retval);
    }

    // resolve logical name first
    boinc_resolve_filename(input_filename, input_path, sizeof(input_path));

    auto gl = globals();
    auto res = PrepareLcData(gl, input_path);
    if (res <= 0)
    {
        fprintf(stderr, "\nCouldn't find input file, resolved name %s.\n", input_path);
        fflush(stderr);
    }

    /* Time in JD*/
    std::vector<double> tim(gl.maxDataPoints + 4 + 1, 0.0);
    /* Brightness*/
    std::vector<double> brightness(gl.maxDataPoints + 4 + 1);
    /* Solar phase angle */
    std::vector<double> al(gl.Lcurves + 1, 0.0);
    /* Weights...*/
    std::vector<double> weight_lc(gl.Lcurves + 1, 0.0);
    /* Ecliptic astronomical tempo-centric coordinates of the Sun in AU*/
    double e0[4]{};
    /* Ecliptic astronomical centric coordinates of the Earth in AU*/
    double e[4]{};
    /* Normalization of distance vectors*/
    std::vector<std::vector<double>> ee;
    init_matrix(ee, gl.maxDataPoints + 4 + 1, 3 + 1, 0.0);
    std::vector<std::vector<double>> ee0;
    init_matrix(ee0, gl.maxDataPoints + 4 + 1, 3 + 1, 0.0);

    std::vector<double> sig(gl.maxDataPoints + 4 + 1, 0.0);
    std::vector<double> cg_first(MAX_N_PAR + 1, 0.0);
    std::vector<double> cg(MAX_N_PAR + 1, 0.0);

    std::vector<double> t(MAX_N_FAC + 1, 0.0);
    std::vector<double> f(MAX_N_FAC + 1, 0.0);
    std::vector<double> at(MAX_N_FAC + 1, 0.0);
    std::vector<double> af(MAX_N_FAC + 1, 0.0);
    std::vector<int> ia(MAX_N_PAR + 1, 0);
    std::vector<std::vector<int>> ifp;
    init_matrix(ifp, MAX_N_FAC + 1, 4 + 1, 0);

#if defined __GNUC__
    gl.initializeVectors(MAX_N_PAR + 1, MAX_N_PAR + 8 + 1);
#else
    init_matrix(gl.covar, MAX_N_PAR + 1, MAX_N_PAR + 1, 0.0);
    init_matrix(gl.alpha, MAX_N_PAR + 1, MAX_N_PAR + 8 + 1, 0.0);
#endif

    // open the input file
    FILE* infile = boinc_fopen(input_path, "r");
    if (!infile) {
        fprintf(stderr,
            "%s Couldn't find input file, resolved name %s.\n",
            boinc_msg_prefix(buf, sizeof(buf)), input_path
        );
        std::exit(-1);
    }

    // output file
    boinc_resolve_filename(output_filename, output_path, sizeof(output_path));
    //    out.open(output_path, "w");

        // See if there's a valid checkpoint file.
        // If so seek input file and truncate output file
        //
    boinc_resolve_filename(checkpoint_file, chkpt_path, sizeof(chkpt_path));
    FILE* state = boinc_fopen(chkpt_path, "r");
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
        std::exit(1);
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

    int err = 0;

    /* Period interval (hrs) fixed or free */
    err = fscanf_s(infile, "%lf %lf %lf %d", &per_start, &per_step_coef, &per_end, &ia_prd);	fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Epoch of zero time t0 */
    err = fscanf_s(infile, "%lf", &jd_00);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Initial fixed rotation angle fi0 */
    err = fscanf_s(infile, "%lf", &Phi_0);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* The weight factor for conv. reg. */
    err = fscanf_s(infile, "%lf", &conw);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Degree and order of the Laplace series */
    err = fscanf_s(infile, "%d %d", &Lmax, &Mmax);                        fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Number of triangulation rows per octant */
    err = fscanf_s(infile, "%d", &nrows);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Initial guesses for phase funct. params. */
    err = fscanf_s(infile, "%lf %d", &par[1], &ia_par[1]);                fgets(str_temp, MAX_LINE_LENGTH, infile);
    err = fscanf_s(infile, "%lf %d", &par[2], &ia_par[2]);                fgets(str_temp, MAX_LINE_LENGTH, infile);
    err = fscanf_s(infile, "%lf %d", &par[3], &ia_par[3]);                fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Initial Lambert coeff. (L-S=1) */
    err = fscanf_s(infile, "%lf %d", &cl, &ia_cl);                        fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Maximum number of iterations (when > 1) or
       minimum difference in dev to stop (when < 1) */
    err = fscanf_s(infile, "%lf", &stop_condition);                       fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Minimum number of iterations when stop_condition < 1 */
    err = fscanf_s(infile, "%d", &n_iter_min);                            fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Multiplicative factor for Alamda */
    err = fscanf_s(infile, "%lf", &Alamda_incr);                          fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Alamda initial value*/
    err = fscanf_s(infile, "%lf", &Alamda_start);                         fgets(str_temp, MAX_LINE_LENGTH, infile);

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
    err = fscanf_s(infile, "%d", &gl.Lcurves);

    int ndata = 0;			/* total number of data */
    int k2 = 0;				/* index */
    double al0 = PI;		/* the smallest solar phase angle */
    double al0_abs = PI;
    int ial0 = -1;			/* initialization, index of al0 */
    int ial0_abs = -1;
    double jdMin = 1e20;	/* initial minimum JD (Julian date)*/
    double jdMax = -1e40;	/* initial maximum JD (Julian date)*/
    int onlyrel = 1;
    double jd_0 = jd_00;
    double a = a0;
    double b = b0;
    double c_axis = c0;

    /* Loop over lightcurves */
    for (i = 1; i <= gl.Lcurves; i++)
    {
        double average = 0; /* average */
        err = fscanf_s(infile, "%d %d", &gl.Lpoints[i], &i_temp); /* points in this lightcurve */
        fgets(str_temp, MAX_LINE_LENGTH, infile);

        gl.Inrel[i] = 1 - i_temp;
        if (gl.Inrel[i] == 0)
            onlyrel = 0;

        /* loop over one lightcurve */
        for (j = 1; j <= gl.Lpoints[i]; j++)
        {
            ndata++;

            err = fscanf_s(infile, "%lf %lf", &tim[ndata], &brightness[ndata]); /* JD, brightness */
            err = fscanf_s(infile, "%lf %lf %lf", &e0[1], &e0[2], &e0[3]); /* ecliptic astr_tempocentric coord. of the Sun in AU */
            err = fscanf_s(infile, "%lf %lf %lf", &e[1], &e[2], &e[3]); /* ecliptic astrocentric coord. of the Earth in AU */

            /* selects the minimum and maximum JD */
            if (tim[ndata] < jdMin) jdMin = tim[ndata];
            if (tim[ndata] > jdMax) jdMax = tim[ndata];

            /* normals of distance vectors */
            e0len = sqrt(e0[1] * e0[1] + e0[2] * e0[2] + e0[3] * e0[3]);
            elen = sqrt(e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);

            average += brightness[ndata];

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
                if ((al[i] < al0_abs) && (gl.Inrel[i] == 0))
                {
                    al0_abs = al[i];
                    ial0_abs = ndata;
                }
            }
        } /* j, one lightcurve */

        // For Unit test reference only
        /*printArray(ee, ndata, 3, "ee");
        printArray(ee0, ndata, 3, "ee0");*/

        average /= gl.Lpoints[i];
        // For unit test reference only
        //printf("gl.ave: %.30f\n", gl.ave);

        /* Mean brightness of lcurve
           Use the mean brightness as 'sigma' to renormalize the
           mean of each lightcurve to unity */

        for (j = 1; j <= gl.Lpoints[i]; j++)
        {
            k2++;
            sig[k2] = average;
        }

    } /* i, all lightcurves */

    /* initiation of weights */
    for (i = 1; i <= gl.Lcurves; i++)
        weight_lc[i] = -1;

    /* reads weights */
    auto scanResult = 0;
    while (true)
    {
        scanResult = fscanf(infile, "%d", &lc_number);
        if (scanResult <= 0) break;
        scanResult = fscanf(infile, "%lf", &weight_lc[lc_number]);
        if (scanResult <= 0) break;
        if (boinc_is_standalone())
            printf("weights %d %g\n", lc_number, weight_lc[lc_number]);

        if (feof(infile)) break;
    }

    /* If input jd_0 <= 0 then the jd_0 is set to the day before the lowest JD in the data */
    if (jd_0 <= 0)
    {
        jd_0 = static_cast<int>(jdMin);
        if (boinc_is_standalone())
            printf("\nNew epoch of zero time  %f\n", jd_0);
    }

    /* loop over data - subtraction of jd_0 */
    for (i = 1; i <= ndata; i++)
        tim[i] = tim[i] - jd_0;

    // For Unit test reference only
    //printArray(tim, ndata, "tim");

    Phi_0 = Phi_0 * DEG2RAD;

    k = 0;
    for (i = 1; i <= gl.Lcurves; i++)
        for (j = 1; j <= gl.Lpoints[i]; j++)
        {
            k++;
            if (weight_lc[i] == -1)
                gl.Weight[k] = 1;
            else
                gl.Weight[k] = weight_lc[i];
        }

    for (i = 1; i <= 3; i++)
        gl.Weight[k + i] = 1;

    // For Unit tests reference only
    //printArray(Weight, 122, "Weight");

    /* use calibrated data if possible */
    if (onlyrel == 0)
    {
        al0 = al0_abs;
        ial0 = ial0_abs;
    }

    // For unit test reference only
    //printf("al0: %.30f\tial0 %d\n", al0, ial0);

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
    // gl.Lcurves = gl.Lcurves + 1;
    // gl.Lpoints[gl.Lcurves] = 3;
    // gl.Inrel[gl.Lcurves] = 0;
    MakeConvexityRegularization(gl);

    // For Unit test reference only
    //printArray(Inrel, 10, "Inrel");

    /* optimization of the convexity weight **************************************************************/
    APP_INIT_DATA aid;
    boinc_get_init_data(aid);
    if (!checkpoint_exists)
    {
        conw_r = conw / escl / escl;
        new_conw = 0;

        fprintf(stderr, "BOINC client version %d.%d.%d\n", aid.major_version, aid.minor_version, aid.release);

        int major, minor, build, revision;

#if !defined __GNUC__ && defined _WIN32
        char nameBuffer[MAX_PATH];
        GetModuleFileNameA(nullptr, nameBuffer, MAX_PATH);
        auto filename = PathFindFileName(nameBuffer);
        GetVersionInfo(filename, major, minor, build, revision);
        std::cerr << "Application: " << filename << std::endl;
#elif defined __GNUC__
        GetVersionInfo(major, minor, build, revision);
#if !defined __APPLE__ && !defined __arm__ && !defined __aarch64__
        auto path = std::filesystem::current_path();
#endif
        std::cerr << "Application: " << argv[0] << std::endl;
#endif
        std::cerr << "Version: " << major << "." << minor << "." << build << "." << revision << std::endl;
        std::cerr << "Compiled with: " << GetCompilerInfo() << std::endl;
    }

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64) || defined __APPLE__
    getSystemInfo();
#else
    std::cerr << "CPU: " << GetCpuInfo() << std::endl;
    std::cerr << "RAM: " << round(getTotalSystemMemory() * 100.0) / 100.0 << " GB" << std::endl;
#endif

    // --- Set desired CPU SIMD optimization ---
    GetSupportedSIMDs();

    SIMDEnum useOptimization = SIMDEnum::Undefined;
    for (i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--optimization") == 0 && i + 1 < argc)
        {
            auto index = atoi(argv[++i]);
            useOptimization = static_cast<SIMDEnum>(index);
            std::cerr << "Manual Optimization Override: " << getSIMDEnumName(useOptimization) << std::endl;
        }
    }

    // TEST
    //CPUopt.hasAVX512dq = false;
    //CPUopt.hasAVX512 = false;
    //CPUopt.hasFMA = false;
    //CPUopt.hasAVX = false;
    //CPUopt.hasSSE3 = false;
    //CPUopt.hasSSE2 = false;
    //CPUopt.hasASIMD = false;

    useOptimization = useOptimization == SIMDEnum::Undefined
        ? GetBestSupportedSIMD()
        : CheckSupportedSIMDs(useOptimization);

    SetOptimizationStrategy(useOptimization);

    // -------------

    while ((new_conw != 1) && ((conw_r * escl * escl) < 10.0))
    {
        for (j = 1; j <= 3; j++)
        {
            ndata++;
            brightness[ndata] = 0;
            sig[ndata] = 1 / conw_r;
        }

        // For Unit tests reference only
        //printArray(sig, 130, "sig");

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
            fprintf(stderr, "\nError: Number of facets is greater than MAX_N_FAC!\n"); fflush(stderr); std::exit(2);
        }

        /* makes indices to triangle vertices */
        trifac(nrows, ifp);

        // NOTE: For unit tests arrange only
        //printArray(f, ndir, "f");

        /* areas and normals of the triangulated Gaussian image sphere */
        areanorm(t, f, ndir, Numfac, ifp, at, af, gl);

        /* Precompute some function values at each normal direction*/
        sphfunc(Numfac, at, af);

        ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

        freq_start = 1 / per_start;
        freq_end = 1 / per_end;
        freq_step = 0.5 / (jdMax - jdMin) / 24 * per_step_coef;

        // For Unit tests ref only
        //printf("jd_max: %.6f\n", jd_max);
        //printf("jd_min: %.6f\n", jd_min);

        /* Give ia the value 0/1 if it's fixed/free */
        ia[Ncoef + 1 - 1] = ia_beta_pole;
        ia[Ncoef + 2 - 1] = ia_lambda_pole;
        ia[Ncoef + 3 - 1] = ia_prd;

        /* phase function parameters */
        Nphpar = 3;
        ma = Ncoef + 5 + Nphpar;

        atry.resize(ma + 1, 0.0);
        beta.resize(ma + 1, 0.0);
        da.resize(ma + 1, 0.0);

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
            fprintf(stderr, "\nError: Number of parameters is greater than MAX_N_PAR = %d\n", MAX_N_PAR); fflush(stderr); std::exit(2);
        }

        max_test_periods = 10;
        ave_dark_facet = 0.0;
        n_iter = static_cast<int>((freq_start - freq_end) / freq_step) + 1;

        if(n_iter > MAX_N_FPOINTS) {
            fprintf(stderr, "\nError: Frequency points %d is greater than MAX_N_FPOINTS = %d!\n", n_iter, MAX_N_FPOINTS); fflush(stderr); std::exit(2);
        }
        if(n_iter < 0 || freq_end > freq_start || freq_step <= 0) {
            fprintf(stderr, "\nError: Invalid frequency points %d!\n", n_iter); fflush(stderr); std::exit(2);
        }

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

        //NOTE: Gathering initial poles...
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
#ifdef _DEBUG
                printf(".");
                fflush(stderr);
#endif
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

                // For Unit tests reference only
                //printArray(cg, 24, "cg");
                //printArray(ia, 24, "ia");

                /* Levenberg-Marquardt loop */
                n_iter_max = 0;
                iter_diff_max = -1;
                rchisq = -1;
                if (stop_condition > 1)
                {
                    n_iter_max = static_cast<int>(stop_condition);
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

                // For Unit tests reference only
                //printArray(brightness, ndata, "brightness[ndata] -> brightness");

                while (((Niter < n_iter_max) && (iter_diff > iter_diff_max)) || (Niter < n_iter_min))
                {
                    //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha, gl);
                    //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, covar, aalpha, gl);
                    mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, gl);

                    // For Unite test reference only
                    //printArray(aalpha, 25, 25, "aaplha");
                    //printArray(covar, 25, 25, "covar");

                    Niter++;

                    if ((Niter == 1) || (Chisq < Ochisq))
                    {
                        Ochisq = Chisq;
                        calcCtx.CalculateCurv(cg, gl);

                        for (i = 1; i <= 3; i++)
                        {
                            chck[i] = 0;
                            for (j = 1; j <= Numfac; j++)
                                chck[i] = chck[i] + gl.Area[j - 1] * gl.Nor[i - 1][j - 1];
                        }
                        rchisq = Chisq - (pow(chck[1], 2) + pow(chck[2], 2) + pow(chck[3], 2)) * pow(conw_r, 2);
                    }
                    dev_new = sqrt(rchisq / (ndata - 3));
                    //printf("% 0.6f\n", dev_new);

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
                //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha, gl);
                //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, covar, aalpha, gl);
                mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, gl);
                Deallocate = 0;

                totarea = 0;
                for (i = 1; i <= Numfac; i++)
                {
                    totarea = totarea + gl.Area[i - 1];
                }

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

#ifdef _DEBUG
            printf("\n");
#endif
            if (la_best < 0)
                la_best += 360;

#ifdef __GNUC__
            if (std::isnan(dark_best) == 1)
                dark_best = 1.0;
#else
            if (_isnan(dark_best) == 1)
                dark_best = 1.0;
#endif

            sum_dark_facet += dark_best;

            if (boinc_time_to_checkpoint() || boinc_is_standalone()) {
                retval = DoCheckpoint(out, 0, new_conw, conw_r, sum_dark_facet, n); //zero lines
                if (retval) { fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); std::exit(retval); }
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
                fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); std::exit(retval);
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

    /* go to the South Pole (in the same rot. order, not a mirror image) */
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
        fprintf(stderr, "\nError: Number of facets is greater than MAX_N_FAC!\n"); fflush(stderr); std::exit(2);
    }

    /* makes indices to triangle vertices */
    trifac(nrows, ifp);
    /* areas and normals of the triangulated Gaussian image sphere */
    areanorm(t, f, ndir, Numfac, ifp, at, af, gl);
    /* Precompute some function values at each normal direction*/
    sphfunc(Numfac, at, af);

    ellfit(cg_first, a, b, c_axis, Numfac, Ncoef, at, af);

    freq_start = 1 / per_start;
    freq_end = 1 / per_end;
    freq_step = 0.5 / (jdMax - jdMin) / 24 * per_step_coef;

    /* Give ia the value 0/1 if it's fixed/free */
    ia[Ncoef + 1 - 1] = ia_beta_pole;
    ia[Ncoef + 2 - 1] = ia_lambda_pole;
    ia[Ncoef + 3 - 1] = ia_prd;

    /* phase function parameters */
    Nphpar = 3;
    ma = Ncoef + 5 + Nphpar;

    atry.resize(ma + 1, 0.0);
    beta.resize(ma + 1, 0.0);
    da.resize(ma + 1, 0.0);

    /* shape is free to be optimized */
    for (i = 0; i < Ncoef; i++)
    {
        ia[i] = 1;
    }

    /* The first shape param. fixed for relative br. fit */
    if (onlyrel == 1)
    {
        ia[0] = 0;
    }

    ia[Ncoef + 3 + Nphpar + 1 - 1] = ia_cl;

    /* Lommel-Seeliger part is fixed */
    ia[Ncoef + 3 + Nphpar + 2 - 1] = 0;

    if ((Ncoef + 3 + Nphpar + 1) > MAX_N_PAR)
    {
        fprintf(stderr, "\nError: Number of parameters is greater than MAX_N_PAR = %d\n", MAX_N_PAR); fflush(stderr); std::exit(2);
    }

    for (n = n_start_from; n <= static_cast<int>((freq_start - freq_end) / freq_step) + 1; n++)
    {
        fraction_done = n / (((freq_start - freq_end) / freq_step) + 1);
        boinc_fraction_done(fraction_done);

#ifdef _DEBUG
        auto fraction = fraction_done * 100;
        auto time = std::time(nullptr);   // get time now
        std::tm now{};
#if defined __GNUC__ && !defined _WIN32
        localtime_r(&time, &now);
#else
        _localtime64_s(&now, &time);
#endif
        printf("%02d:%02d:%02d | Fraction done: %.3f%%\n", now.tm_hour, now.tm_min, now.tm_sec, fraction);
        fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.3f%%\n", now.tm_hour, now.tm_min, now.tm_sec, fraction);
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
                n_iter_max = static_cast<int>(stop_condition);
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
                //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha, gl);
                //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, covar, aalpha, gl);
                mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, gl);
                Niter++;

                if ((Niter == 1) || (Chisq < Ochisq))
                {
                    Ochisq = Chisq;
                    calcCtx.CalculateCurv(cg, gl);

                    for (i = 1; i <= 3; i++)
                    {
                        chck[i] = 0;
                        for (j = 1; j <= Numfac; j++)
                        {
                            chck[i] = chck[i] + gl.Area[j - 1] * gl.Nor[i - 1][j - 1];
                        }
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
            //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, Ncoef + 5 + Nphpar, covar, aalpha, gl);
            //mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, covar, aalpha, gl);
            mrqmin(ee, ee0, tim, brightness, sig, cg, ia, ma, gl);
            Deallocate = 0;

            totarea = 0;
            for (i = 1; i <= Numfac; i++)
            {
                totarea = totarea + gl.Area[i - 1];
            }

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
        auto darkBest = n == 1
            ? conw_r * escl * escl
            : dark_best;

        out.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * per_best, dev_best, dev_best* dev_best* (ndata - 3), darkBest, round(la_best), round(be_best));

        //if (n == 1)
        //{
        //    out.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * per_best, dev_best, dev_best * dev_best * (ndata - 3), conw_r * escl * escl, round(la_best), round(be_best));
        //}
        //else
        //{
        //    out.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * per_best, dev_best, dev_best * dev_best * (ndata - 3), dark_best, round(la_best), round(be_best));
        //}
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
            // blinkLed(3);
#endif


        if (boinc_time_to_checkpoint() || boinc_is_standalone())
        {
            retval = DoCheckpoint(out, n, new_conw, conw_r, 0.0, 0);
            if (retval) { fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); std::exit(retval); }
            boinc_checkpoint_completed();
        }

    } /* period loop */

    out.close();

    //deallocate_matrix_double(ee, gl.maxDataPoints + 4);
    //deallocate_matrix_double(ee0, gl.maxDataPoints + 4);
    //aligned_deallocate_matrix_double(covar, MAX_N_PAR);
    //aligned_deallocate_matrix_double(aalpha, MAX_N_PAR);
    //deallocate_matrix_int(ifp, MAX_N_FAC);

    //deallocate_vector(tim);
    //deallocate_vector(brightness);
    //deallocate_vector(sig);
    //deallocate_vector(cg);
    //deallocate_vector(cg_first);
    //deallocate_vector(t);
    //deallocate_vector(f);
    //deallocate_vector(at);
    //deallocate_vector(af);
    //deallocate_vector(ia);
    //deallocate_vector(al);
    //deallocate_vector(weight_lc);

    //delete[] gl.Inrel;
    //delete[] gl.Lpoints;
    //delete[] gl.ytemp;
    //delete[] gl.Weight;
    //delete2Darray(gl.dytemp, gl.dytemp_sizeX);

    free(str_temp);

    boinc_fraction_done(1);
#ifdef APP_GRAPHICS
    update_shmem();
#endif

    //system("PAUSE");

    boinc_finish(0);
}

#ifdef _WIN32

int WINAPI WinMain(_In_ HINSTANCE hInst, _In_opt_ HINSTANCE hPrevInst, _In_ LPSTR Args, _In_ int WinMode)
{
	LPSTR command_line;
	char* argv[100];
	int argc;

	command_line = GetCommandLine();
	argc = parse_command_line(command_line, argv);
	return main(argc, argv);
}

#endif
