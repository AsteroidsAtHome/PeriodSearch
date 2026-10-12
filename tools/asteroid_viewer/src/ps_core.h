// period_search as a library function (see scripts/gen_ps_core.py).
#pragma once
#include <array>
#include <atomic>
#include <functional>
#include <vector>

struct PsHooks {
  // --- options
  double perStart = 0, perEnd = 0;  // [h] override the period interval of the input file (0 = keep)
  double conw = 0;                  // > 0: use this convexity weight and skip its optimisation
  bool singlePole = false;           // start only from (poleLambda, poleBeta) [deg] instead of the 10 standard poles
  double poleLambda = 0, poleBeta = 0;
  std::atomic<bool>* cancel = nullptr;
  // --- callbacks (called from the computing thread)
  std::function<void(int stage, int n, int pole, int iter, double rms)> onIter;  // stage 0 = conw optimisation, 1 = scan
  std::function<void(int n, int pole, double periodH, double lambda, double beta, double rms)> onPole;
  std::function<void(int n, int total, double trialPeriodH)> onPeriod;
  std::function<void(const char* line)> onRow;  // the line period_search writes to period_search_out
  // --- best solution of the whole run
  double bestDev = 1e40, bestPeriod = 0, bestLambda = 0, bestBeta = 0, conwUsed = 0;
  std::vector<double> area;                  // Gaussian image: facet areas
  std::vector<std::array<double, 3>> normal;  // and normals (body frame)
  std::vector<double> cg;                    // all fitted parameters (shape, pole, omega, phase, Lambert)
  int ncoef = 0, nphpar = 0;
};

// Runs period_search on input_path. Returns 0 when finished, 1 when cancelled.
// Throws std::runtime_error on errors. Not re-entrant (period_search uses globals).
int ps_run_impl(const char* input_path, PsHooks* hooks);

// Gaussian image of the model with shape coefficients cg[0..ncoef-1] on a finer triangulation
// (8*nrows^2 facets, nrows <= 11). Returns 0 on success. Same globals: not re-entrant.
int ps_gauss_image(const double* cg, int ncoef, int lmax, int mmax, int nrows,
                   std::vector<double>& area, std::vector<std::array<double, 3>>& normal);
