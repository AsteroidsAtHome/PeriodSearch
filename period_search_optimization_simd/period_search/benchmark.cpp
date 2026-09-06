#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "arrayHelpers.hpp"
#include "benchmark.hpp"

globals gl;

namespace
{
	constexpr int benchmark_triangulation_rows = 6;
	constexpr int benchmark_lmax = 6;
	constexpr int benchmark_mmax = 6;
	constexpr int benchmark_lc_points = 400;
	constexpr int benchmark_lc_count = 2;
	constexpr double benchmark_period_hr = 7.5;
	constexpr double benchmark_alamda = 1e-3;
	constexpr double benchmark_target_ms = 250.0;
	constexpr int benchmark_min_rounds = 5;
	constexpr int benchmark_max_rounds = 5000;

	struct SimdCandidate
	{
		SIMDEnum simd;
		bool supported;
	};

	struct BenchmarkRow
	{
		std::string name;
		int rounds;
		double ms_per_round;
		double chisq;
		bool failed;
	};

	struct BenchmarkWorkload
	{
		int ma = 0;
		int mfit = 0;
		int lastone = 0;
		int lastma = 0;
		double chisq = 0.0;
		bool failed = false;

		std::vector<std::vector<double>> x1;
		std::vector<std::vector<double>> x2;
		std::vector<double> x3;
		std::vector<double> y;
		std::vector<double> sig;
		std::vector<double> a;
		std::vector<double> beta_sv;
		std::vector<double> da;
		std::vector<int> ia;

		void Round()
		{
			double trial_chisq = 0.0;

			calcCtx.CalculateMrqcof(x1, x2, x3, y, sig, a, ia, ma, beta_sv, mfit, lastone, lastma, trial_chisq, gl, false);

			for (auto j = 0; j < mfit; j++)
			{
				for (auto k = 0; k < mfit; k++)
					gl.covar[j][k] = gl.alpha[j][k];

				gl.covar[j][j] *= (1.0 + benchmark_alamda);
				da[j] = beta_sv[j];
			}

			int err_code = 0;
			calcCtx.CalculateGaussErrc(gl, mfit, da, err_code);
			if (err_code != 0)
			{
				failed = true;
				return;
			}

			calcCtx.CalculateMrqcof(x1, x2, x3, y, sig, a, ia, ma, da, mfit, lastone, lastma, trial_chisq, gl, true);

			chisq = trial_chisq;
		}
	};

	double TimedRounds(BenchmarkWorkload& wl, const int rounds)
	{
		const auto start = std::chrono::steady_clock::now();
		for (auto r = 0; r < rounds; r++)
			wl.Round();

		const auto stop = std::chrono::steady_clock::now();
		return std::chrono::duration<double, std::milli>(stop - start).count() / rounds;
	}

	bool ChisqMatches(const double reference, const double value)
	{
		const double diff = std::fabs(value - reference);
		const double scale = (std::max)(std::fabs(reference), 1.0);
		return (diff / scale) < 1e-6;
	}

	void SetupGeometry(std::vector<double>& t, std::vector<double>& f,
		std::vector<double>& at, std::vector<double>& af,
		std::vector<std::vector<int>>& ifp, std::vector<double>& cg_first)
	{
		Lmax = benchmark_lmax;
		Mmax = benchmark_mmax;
		Nphpar = 3;

		Ncoef = 0;
		for (auto m = 0; m <= Mmax; m++)
			for (auto l = m; l <= Lmax; l++)
			{
				Ncoef++;
				if (m != 0) Ncoef++;
			}

		const auto nrows = benchmark_triangulation_rows;
		const double dth = PI / (2 * nrows);

		auto k = 1;
		t[1] = 0;
		f[1] = 0;
		for (auto i = 1; i <= nrows; i++)
		{
			const double dph = PI / (2 * i);
			for (auto j = 0; j <= 4 * i - 1; j++)
			{
				k++;
				t[k] = i * dth;
				f[k] = j * dph;
			}
		}

		for (auto i = nrows - 1; i >= 1; i--)
		{
			const double dph = PI / (2 * i);
			for (auto j = 0; j <= 4 * i - 1; j++)
			{
				k++;
				t[k] = PI - i * dth;
				f[k] = j * dph;
			}
		}

		const auto ndir = k + 1;
		t[ndir] = PI;
		f[ndir] = 0;
		Numfac = 8 * nrows * nrows;

		trifac(nrows, ifp);
		areanorm(t, f, ndir, Numfac, ifp, at, af, gl);
		sphfunc(Numfac, at, af);

		cg_first.assign(static_cast<size_t>(Ncoef) + 2, 0.0);
		ellfit(cg_first, 1.05, 1.00, 0.95, Numfac, Ncoef, at, af);
	}

	void SetupWorkload(BenchmarkWorkload& wl, const std::vector<double>& cg_first)
	{
		gl.Lcurves = benchmark_lc_count + 1;
		gl.Lpoints.assign(static_cast<size_t>(gl.Lcurves) + 1, 0);
		gl.Inrel.assign(static_cast<size_t>(gl.Lcurves) + 1, 0);

		for (auto i = 1; i <= benchmark_lc_count; i++)
		{
			gl.Lpoints[i] = benchmark_lc_points;
			gl.Inrel[i] = 1;
		}
		gl.Lpoints[gl.Lcurves] = 3;
		gl.Inrel[gl.Lcurves] = 0;

		auto ndata = 0;
		for (auto i = 1; i <= gl.Lcurves; i++)
			ndata += gl.Lpoints[i];

		gl.maxLcPoints = benchmark_lc_points;
		gl.maxDataPoints = ndata;

		gl.ytemp.resize(static_cast<size_t>(gl.maxLcPoints) + 2, 0.0);

		gl.dytemp_sizeY = MAX_N_PAR + 1 + 4;
		gl.dytemp_sizeX = gl.maxLcPoints + 2;
		init_matrix(gl.dytemp, gl.dytemp_sizeX + 1, gl.dytemp_sizeY + 1, 0.0);

		gl.Weight.assign(static_cast<size_t>(gl.maxDataPoints) + 1 + 4, 1.0);
		gl.ave = 0.0;

#if defined __GNUC__
		gl.initializeVectors(MAX_N_PAR + 1, MAX_N_PAR + 8 + 1);
#else
		init_matrix(gl.covar, MAX_N_PAR + 1, MAX_N_PAR + 1, 0.0);
		init_matrix(gl.alpha, MAX_N_PAR + 1, MAX_N_PAR + 8 + 1, 0.0);
#endif

		wl.ma = Ncoef + 5 + Nphpar;

		init_matrix(wl.x1, gl.maxDataPoints + 4 + 1, 3 + 1, 0.0);
		init_matrix(wl.x2, gl.maxDataPoints + 4 + 1, 3 + 1, 0.0);
		init_vector(wl.x3, gl.maxDataPoints + 4 + 1, 0.0);
		init_vector(wl.y, gl.maxDataPoints + 4 + 1, 0.0);
		init_vector(wl.sig, gl.maxDataPoints + 4 + 1, 0.0);
		init_vector(wl.a, wl.ma + 1, 0.0);
		init_vector(wl.beta_sv, wl.ma + 1, 0.0);
		init_vector(wl.da, wl.ma + 1, 0.0);
		init_vector(wl.ia, MAX_N_PAR + 1, 0);

		for (auto i = 0; i < Ncoef; i++)
			wl.ia[i] = 1;

		wl.ia[0] = 0;
		wl.ia[Ncoef] = 1;
		wl.ia[Ncoef + 1] = 1;
		wl.ia[Ncoef + 2] = 1;

		for (auto i = 1; i <= Nphpar; i++)
			wl.ia[Ncoef + 2 + i] = 1;

		wl.ia[Ncoef + 3 + Nphpar] = 1;
		wl.ia[Ncoef + 4 + Nphpar] = 0;

		wl.mfit = 0;
		wl.lastma = 0;
		for (auto j = 0; j < wl.ma; j++)
		{
			if (wl.ia[j])
			{
				wl.mfit++;
				wl.lastma = j;
			}
		}

		wl.lastone = 0;
		for (auto j = 1; j <= wl.lastma; j++)
		{
			if (!wl.ia[j]) break;
			wl.lastone = j;
		}

		for (auto i = 1; i <= Ncoef; i++)
			wl.a[i] = cg_first[i];

		wl.a[Ncoef + 1] = (90.0 - 60.0) * DEG2RAD;
		wl.a[Ncoef + 2] = 180.0 * DEG2RAD;
		wl.a[Ncoef + 3] = 24.0 * 2.0 * PI / benchmark_period_hr;
		wl.a[Ncoef + 4] = 0.5;
		wl.a[Ncoef + 5] = 0.1;
		wl.a[Ncoef + 6] = -0.5;
		wl.a[Ncoef + 7] = log(0.5);
		wl.a[Ncoef + 8] = 1.0;

		std::mt19937 gen(42);
		std::normal_distribution<double> noise(0.0, 1.0);

		auto np = 0;
		for (auto i = 1; i <= gl.Lcurves; i++)
		{
			for (auto jp = 1; jp <= gl.Lpoints[i]; jp++)
			{
				np++;

				const double tt = 0.05 * np;
				const double th = PI * (np % 719) / 719.0;
				const double ph = 2.399963229728653 * np;

				wl.x3[np] = tt;
				wl.x1[np][1] = sin(th) * cos(ph);
				wl.x1[np][2] = sin(th) * sin(ph);
				wl.x1[np][3] = cos(th);

				const double th0 = (std::min)(PI - 1e-3, th * 0.97 + 0.02);
				const double ph0 = ph + 0.7;

				wl.x2[np][1] = sin(th0) * cos(ph0);
				wl.x2[np][2] = sin(th0) * sin(ph0);
				wl.x2[np][3] = cos(th0);

				if (i < gl.Lcurves)
				{
					wl.y[np] = 0.95 + 0.12 * sin(2.0 * PI * tt / benchmark_period_hr) + 0.005 * noise(gen);
					wl.sig[np] = 0.02;
				}
				else
				{
					wl.y[np] = 0.0;
					wl.sig[np] = 10.0;
				}
			}
		}
	}
}

int RunBenchmark()
{
	std::cerr << "Running SIMD self-test benchmark..." << std::endl;

	GetSupportedSIMDs();

	std::cerr << "CPU: " << GetCpuInfo() << std::endl;

	std::string support("none");
	if (CPUopt.hasAVX512 && CPUopt.hasAVX512dq) support += " avx512dq";
	else if (CPUopt.hasAVX512) support += " avx512f";
	if (CPUopt.hasFMA) support += " fma";
	if (CPUopt.hasAVX) support += " avx";
	if (CPUopt.hasSSE3) support += " sse3";
	if (CPUopt.hasSSE2) support += " sse2";
	if (CPUopt.hasASIMD) support += " asimd";
	if (CPUopt.hasSVE) support += " sve";
	

	std::cerr << "Hardware SIMD support:" << support << std::endl;

	std::vector<double> t(MAX_N_FAC + 2, 0.0);
	std::vector<double> f(MAX_N_FAC + 2, 0.0);
	std::vector<double> at(MAX_N_FAC + 2, 0.0);
	std::vector<double> af(MAX_N_FAC + 2, 0.0);
	std::vector<std::vector<int>> ifp;
	init_matrix(ifp, MAX_N_FAC + 1, 4 + 1, 0);

	std::vector<double> cg_first;
	SetupGeometry(t, f, at, af, ifp, cg_first);

	BenchmarkWorkload wl;
	SetupWorkload(wl, cg_first);

	std::vector<SimdCandidate> candidates =
	{
		{ SIMDEnum::OptNONE, true },
		{ SIMDEnum::OptSSE2, CPUopt.hasSSE2 },
		{ SIMDEnum::OptSSE3, CPUopt.hasSSE3 },
#if !defined _VC140_XP
		{ SIMDEnum::OptAVX, CPUopt.hasAVX },
		{ SIMDEnum::OptFMA, CPUopt.hasFMA },
		{ SIMDEnum::OptAVX512, CPUopt.hasAVX512 && CPUopt.hasAVX512dq },
#endif
        { SIMDEnum::OptASIMD, CPUopt.hasASIMD },
		#if defined(__aarch64__)
        { SIMDEnum::OptSVE, CPUopt.hasSVE },
		#else
		{ SIMDEnum::OptSVE, true },
		#endif
	};

	std::fprintf(stdout, "\nBenchmark workload: %d facets, %d coefficients, %d parameters (%d fitted), %d data points\n",
		Numfac, Ncoef, wl.ma, wl.mfit, gl.maxDataPoints);
	std::fprintf(stdout, "%-10s %8s %12s %14s   %s\n", "Impl", "rounds", "ms/round", "speedup", "chi^2 check");

	std::vector<BenchmarkRow> rows;

	for (const auto& candidate : candidates)
	{
		if (!candidate.supported)
			continue;

		SetOptimizationStrategy(candidate.simd);

		wl.failed = false;
		wl.Round();

		if (wl.failed)
		{
			rows.push_back({ getSIMDEnumName(candidate.simd), 0, 0.0, 0.0, true });
			continue;
		}

		const double single_round_ms = TimedRounds(wl, 1);

		auto rounds = static_cast<int>(benchmark_target_ms / (std::max)(single_round_ms, 0.001));
		rounds = (std::max)(benchmark_min_rounds, (std::min)(rounds, benchmark_max_rounds));

		const double ms_per_round = TimedRounds(wl, rounds);

		rows.push_back({ getSIMDEnumName(candidate.simd), rounds, ms_per_round, wl.chisq, false });
	}

	const BenchmarkRow* baseline = nullptr;
	for (const auto& row : rows)
	{
		if (!row.failed && (baseline == nullptr || row.ms_per_round < baseline->ms_per_round))
			baseline = &row;
	}

	const BenchmarkRow* scalar_row = nullptr;
	for (const auto& row : rows)
	{
		if (row.name == "NONE" && !row.failed)
		{
			scalar_row = &row;
			break;
		}
	}

	const BenchmarkRow* reference = scalar_row != nullptr ? scalar_row : baseline;

	auto mismatches = 0;
	auto failures = 0;
	const BenchmarkRow* best = nullptr;

	for (const auto& row : rows)
	{
		double speedup = 0.0;
		if (!row.failed && reference != nullptr)
			speedup = reference->ms_per_round / row.ms_per_round;

		if (row.failed)
		{
			failures++;
			std::fprintf(stdout, "%-10s %8d %12s %14s   %s\n", row.name.c_str(), row.rounds, "error", "-", "FAILED");
			continue;
		}

		if (best == nullptr || row.ms_per_round < best->ms_per_round)
			best = &row;

		const char* check = "baseline";
		if (reference != nullptr && &row != reference)
		{
			check = ChisqMatches(reference->chisq, row.chisq) ? "OK" : "MISMATCH";
			if (check[0] == 'M') mismatches++;
		}

		char speedup_buf[16];
		std::snprintf(speedup_buf, sizeof(speedup_buf), "%.2fx", speedup);

		std::fprintf(stdout, "%-10s %8d %12.3f %14s   %s\n", row.name.c_str(), row.rounds, row.ms_per_round, speedup_buf, check);
	}

	std::fprintf(stdout, "\n");

	if (best != nullptr && reference != nullptr)
		std::fprintf(stdout, "Fastest implementation: %s (%.2fx vs %s)\n",
			best->name.c_str(), reference->ms_per_round / best->ms_per_round, reference->name.c_str());

	if (mismatches > 0)
		std::fprintf(stderr, "Benchmark self-test FAILED: %d implementation(s) produced inconsistent chi^2.\n", mismatches);

	if (failures > 0)
		std::fprintf(stderr, "Benchmark self-test FAILED: %d implementation(s) reported solver errors.\n", failures);

	if (mismatches == 0 && failures == 0)
		std::fprintf(stdout, "Self-test PASSED: all implementations agree within tolerance.\n");

	return (mismatches > 0 || failures > 0) ? 1 : 0;
}
