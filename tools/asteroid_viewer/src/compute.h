// Background computation of a shape model inside the viewer: runs period_search
// (ps_core) for a few trial periods, then reconstructs the polyhedron (Minkowski).
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "data.h"
#include "minkowski.h"
#include "ps_core.h"

// Smooth model: the fitted curvature function exp(sum cg*Y_lm) evaluated on a finer grid
// (8*nrows^2 facets) and turned into a polyhedron, with the non-closure spread smoothly.
// Thread-safe (serialised with the period_search runs).
bool smoothModelMesh(const std::vector<double>& shapeCoef, int lmax, int mmax, const Mesh* warmStart, Mesh& out,
                     std::string& err, std::atomic<bool>* cancel = nullptr, int nrows = 10);

struct ShapeJob {
  struct PoleResult { int n, pole; double period, lambda, beta, rms; };
  struct Snapshot {  // copy of the progress for the UI thread
    bool running = false;
    std::string phase;
    int stage = 1, n = 0, total = 0, pole = 0, nPoles = 10, iter = 0, minkIter = 0;
    double trialP = 0, rms = 0, minkMismatch = 0, elapsed = 0;
    std::vector<std::vector<double>> rmsHist;  // rms per LM iteration, one curve per starting pole (current trial)
    std::vector<PoleResult> poles;             // final rms of every (trial, pole)
    std::vector<std::string> rows;             // period_search_out lines produced
  };

  ~ShapeJob();
  // conw > 0: fixed convexity weight (skips its optimisation)
  // lmax, mmax > 0: also build the smooth model
  // singlePole: start the fit only from (lambda, beta) [deg] instead of the 10 standard poles (~10x faster)
  void start(const std::string& inputPath, double perStart, double perEnd, double conw, int lmax = 0, int mmax = 0,
             bool singlePole = false, double lambda = 0, double beta = 0);
  void cancel() { cancel_ = true; }
  bool running() const { return running_; }
  // true once after the job ended (successfully, cancelled or with an error); joins the thread
  bool collect();
  Snapshot snapshot();

  // results (valid after collect())
  bool ok = false, cancelled = false;
  std::string error;
  PsHooks hooks;            // best solution: Gaussian image + parameters
  MinkowskiResult mink;     // polyhedron (period_search resolution, dark facet as one facet)
  Mesh smoothMesh;          // smooth model (if requested)
  double seconds = 0;

  ModelParams model() const;    // as read from cg_dump.txt
  GaussImage gauss() const;     // as read from shape_dump.txt
  // writes asteroid_shape.obj, shape_dump.txt, cg_dump.txt
  bool save(const std::string& dir, std::string& err) const;

 private:
  void run(std::string inputPath, double perStart, double perEnd, double conw, int lmax, int mmax);
  std::thread th_;
  std::atomic<bool> running_{false}, cancel_{false}, finished_{false};
  std::mutex m_;
  Snapshot s_;
  double t0_ = 0;
};
