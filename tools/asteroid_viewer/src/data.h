// Readers for period_search input/output, OBJ shape and the cg_dump / shape_dump files.
#pragma once
#include <string>
#include <vector>
#include "vecmath.h"

struct Observation {
  double jd = 0, brightness = 0;
  V3 sun, earth;  // asteroid-centric ecliptic vectors [AU]
  int lc = 0;     // lightcurve index
};

struct Lightcurve {
  int first = 0, count = 0;
  bool calibrated = false;
  std::string label;
  double mean = 1;
  double weight = 1;  // optional weights at the end of period_search_in ("lc weight" pairs)
};

struct InputData {
  bool ok = false;
  std::string path;
  double perStart = 0, perStepCoef = 0, perEnd = 0;
  double jd00 = 0, phi0deg = 0, conw = 0;
  int lmax = 0, mmax = 0, nrows = 0;
  std::vector<Lightcurve> lcs;
  std::vector<Observation> obs;
  double jdMin = 0, jdMax = 0;
  // number of trial periods period_search will compute (= rows of a finished period_search_out)
  int totalTrials() const;
  // trial period [h] of output row n (0-based), as in the main loop of period_search
  double trialPeriod(int n) const;
  double epoch() const { return jd00 > 0 ? jd00 : (double)(long long)jdMin; }  // like period_search
};

struct OutputRow { double period, rms, chi2, dark, lambda, beta; };
struct OutputFile {
  std::string path;
  int rows = 0;            // complete lines read
  double lastPeriod = 0;   // period of the last line = most recent trial
  double firstConw = 0;    // column 4 of the first line = convexity weight used by the run
};
struct OutputData {
  std::vector<std::string> paths;
  std::vector<OutputFile> files;
  std::vector<OutputRow> rows;  // all files merged, sorted by period
  int best = -1;
};

struct Mesh {
  bool ok = false;
  std::string path;
  std::vector<V3> v;
  std::vector<int> tri;  // 3 indices per triangle
  double radius = 1;
};

// cg_dump.txt: best-fit parameters from the patched period_search
struct ModelParams {
  bool ok = false;
  std::string path;
  double colat = 0, lambda = 0, omega = 0;  // rad, rad, rad/day
  double phase[3] = {0, 0, 0};              // exp-lin phase function
  double cl = 0.1, cls = 1;                 // Lambert / Lommel-Seeliger weights
  std::vector<double> shapeCoef;            // spherical-harmonics coefficients cg[1..Ncoef] (for the smooth model)
  double periodHours() const { return omega > 0 ? 2 * M_PI / omega * 24 : 0; }
  double betaDeg() const { return 90 - colat * 180 / M_PI; }
  double lambdaDeg() const { double l = lambda * 180 / M_PI; l = std::fmod(l, 360.0); return l < 0 ? l + 360 : l; }
};

// shape_dump.txt: Gaussian image (facet area + normal) - exact photometric model
struct GaussImage {
  bool ok = false;
  std::string path;
  std::vector<double> area;
  std::vector<V3> normal;
};

bool readInput(const std::string& path, InputData& d, std::string& err);
// Appends one file and re-sorts. Only complete (newline-terminated) lines are used, so a
// file that period_search is still writing can be read safely; an empty file is accepted.
bool readOutput(const std::string& path, OutputData& d, std::string& err);

// Quality of a period_search run, by the usual convex-inversion criteria:
//  - unique period: only one interval of trial periods with chi2 below chi2_min x 1.1 in the scanned range,
//    covering at most half of it
//  - dark facet of the best period below 1 % (the convex model fits without a large non-physical facet)
struct RunQuality {
  bool ok = false;
  int minima = 0;        // separate period intervals with chi2 below chi2_min x 1.1 (1 = unique period)
  double belowFrac = 0;  // part of the scanned periods below the threshold (> 0.5: flat periodogram, not unique)
  double dark = -1;      // dark facet [%] of the best row, -1 if unknown (the first row holds the convexity weight)
  double rms = 0;        // rms of the best row
  int grade = 0;         // 2 good (unique, dark < 1 %), 1 fair, 0 poor
  const char* label() const { return grade == 2 ? "good" : grade == 1 ? "fair" : "poor"; }
};
RunQuality assessRun(const OutputData& d);
// Re-reads all files of d from disk (used for live monitoring).
bool reloadOutputs(OutputData& d, std::string& err);
// BOINC slot directories contain "<soft_link>../../projects/.../file</soft_link>" stubs
// instead of the real files; returns the path the stub points to (or path itself).
std::string resolveBoincLink(const std::string& path);
bool readObj(const std::string& path, Mesh& m, std::string& err);
bool readModel(const std::string& path, ModelParams& p, std::string& err);
bool readGauss(const std::string& path, GaussImage& g, std::string& err);

enum class FileKind { Unknown, Input, Output, Obj, Model, Gauss, Directory };
FileKind classify(const std::string& path);
