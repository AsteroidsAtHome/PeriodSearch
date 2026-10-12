#include "u8.h"
#include "compute.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <array>

namespace fs = std::filesystem;

static double nowSec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::mutex gPsMutex;  // period_search keeps its state in globals: one run at a time

ShapeJob::~ShapeJob() {
  cancel_ = true;
  if (th_.joinable()) th_.join();
}

bool smoothModelMesh(const std::vector<double>& coef, int lmax, int mmax, const Mesh* warm, Mesh& out, std::string& err,
                     std::atomic<bool>* cancel, int nrows) {
  std::vector<double> area;
  std::vector<std::array<double, 3>> nor;
  {
    std::lock_guard<std::mutex> lock(gPsMutex);
    if (coef.empty() || ps_gauss_image(coef.data(), (int)coef.size(), lmax, mmax, nrows, area, nor) != 0) {
      err = "cannot evaluate the shape function";
      return false;
    }
  }
  std::vector<V3> n;
  for (auto& v : nor) n.push_back({v[0], v[1], v[2]});
  MinkowskiOptions o;
  o.smoothClosure = true;
  o.warmStart = warm;
  o.mismatchTol = 1e-3;  // plenty for display
  MinkowskiResult r = minkowski(area, n, nullptr, cancel, o);
  if (!r.ok) { err = r.message; return false; }
  out = r.mesh;
  out.path = "(smooth model)";
  return true;
}

void ShapeJob::start(const std::string& inputPath, double perStart, double perEnd, double conw, int lmax, int mmax,
                     bool singlePole, double lambda, double beta) {
  if (running_) return;
  if (th_.joinable()) th_.join();
  ok = cancelled = false;
  error.clear();
  hooks = PsHooks();
  hooks.singlePole = singlePole;
  hooks.poleLambda = lambda;
  hooks.poleBeta = beta;
  mink = MinkowskiResult();
  cancel_ = false;
  finished_ = false;
  {
    std::lock_guard<std::mutex> l(m_);
    s_ = Snapshot();
    s_.running = true;
    s_.phase = "starting";
    s_.nPoles = singlePole ? 1 : 10;
  }
  t0_ = nowSec();
  running_ = true;
  smoothMesh = Mesh();
  th_ = std::thread(&ShapeJob::run, this, inputPath, perStart, perEnd, conw, lmax, mmax);
}

void ShapeJob::run(std::string inputPath, double perStart, double perEnd, double conw, int lmax, int mmax) {
  std::unique_lock<std::mutex> psLock(gPsMutex);
  hooks.perStart = perStart;
  hooks.perEnd = perEnd;
  hooks.conw = conw;
  hooks.cancel = &cancel_;
  hooks.onPeriod = [this](int n, int total, double p) {
    std::lock_guard<std::mutex> l(m_);
    s_.n = n; s_.total = total; s_.trialP = p; s_.stage = 1;
    s_.phase = "fitting";
    s_.rmsHist.assign(10, {});
  };
  hooks.onIter = [this](int stage, int n, int pole, int iter, double rms) {
    std::lock_guard<std::mutex> l(m_);
    if (stage == 0 && s_.phase != "optimising convexity weight") {
      s_.phase = "optimising convexity weight";
      s_.rmsHist.assign(10, {});
    }
    if (stage == 0 && n != s_.n) s_.rmsHist.assign(10, {});
    s_.stage = stage; s_.n = n; s_.pole = pole; s_.iter = iter; s_.rms = rms;
    if (pole >= 1 && pole <= 10) {
      if ((int)s_.rmsHist.size() < 10) s_.rmsHist.resize(10);
      s_.rmsHist[pole - 1].push_back(rms);
    }
  };
  hooks.onPole = [this](int n, int pole, double p, double la, double be, double rms) {
    std::lock_guard<std::mutex> l(m_);
    s_.poles.push_back({n, pole, p, la, be, rms});
  };
  hooks.onRow = [this](const char* line) {
    std::lock_guard<std::mutex> l(m_);
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    s_.rows.push_back(s);
  };
  try {
    int r = ps_run_impl(inputPath.c_str(), &hooks);
    psLock.unlock();
    if (r != 0 || cancel_) {
      cancelled = true;
    } else if (hooks.area.empty()) {
      error = "no solution";
    } else {
      {
        std::lock_guard<std::mutex> l(m_);
        s_.phase = "building the polyhedron";
      }
      std::vector<V3> n;
      for (auto& v : hooks.normal) n.push_back({v[0], v[1], v[2]});
      mink = minkowski(hooks.area, n, [this](int it, double mm) {
        std::lock_guard<std::mutex> l(m_);
        s_.minkIter = it; s_.minkMismatch = mm;
      }, &cancel_);
      if (cancel_) cancelled = true;
      else if (!mink.ok) error = "Minkowski reconstruction failed: " + mink.message;
      else ok = true;
      if (ok && lmax > 0) {
        {
          std::lock_guard<std::mutex> l(m_);
          s_.phase = "building the smooth model";
        }
        std::vector<double> coef(hooks.cg.begin(), hooks.cg.begin() + hooks.ncoef);
        std::string err;
        if (!smoothModelMesh(coef, lmax, mmax, &mink.mesh, smoothMesh, err, &cancel_)) smoothMesh = Mesh();
        if (cancel_) { cancelled = true; ok = false; }
      }
    }
  } catch (std::exception& e) {
    error = e.what();
  }
  seconds = nowSec() - t0_;
  {
    std::lock_guard<std::mutex> l(m_);
    s_.running = false;
    s_.phase = ok ? "done" : (cancelled ? "cancelled" : "error");
  }
  running_ = false;
  finished_ = true;
}

bool ShapeJob::collect() {
  if (!finished_) return false;
  finished_ = false;
  if (th_.joinable()) th_.join();
  return true;
}

ShapeJob::Snapshot ShapeJob::snapshot() {
  std::lock_guard<std::mutex> l(m_);
  Snapshot c = s_;
  c.elapsed = running_ ? nowSec() - t0_ : seconds;
  return c;
}

ModelParams ShapeJob::model() const {
  ModelParams p;
  int k = hooks.ncoef;  // hooks.cg[k-1] == cg[k] of period_search
  if ((int)hooks.cg.size() < k + 8) return p;
  p.colat = hooks.cg[k];
  p.lambda = hooks.cg[k + 1];
  p.omega = hooks.cg[k + 2];
  for (int i = 0; i < 3; i++) p.phase[i] = hooks.cg[k + 3 + i];
  p.cl = std::exp(hooks.cg[k + 6]);
  p.cls = hooks.cg[k + 7];
  p.shapeCoef.assign(hooks.cg.begin(), hooks.cg.begin() + k);
  p.path = "(computed in the viewer)";
  p.ok = true;
  return p;
}

GaussImage ShapeJob::gauss() const {
  GaussImage g;
  g.area = hooks.area;
  for (auto& v : hooks.normal) g.normal.push_back(normalize(V3{v[0], v[1], v[2]}));
  g.path = "(computed in the viewer)";
  g.ok = !g.area.empty();
  return g;
}

bool ShapeJob::save(const std::string& dir, std::string& err) const {
  std::error_code ec;
  fs::create_directories(u8p(dir), ec);
  char hdr[256];
  std::snprintf(hdr, sizeof hdr, "period_h %.8f lambda %.3f beta %.3f rms %.6f numfac %d", hooks.bestPeriod,
                hooks.bestLambda, hooks.bestBeta, hooks.bestDev, (int)hooks.area.size());
  {
    std::ofstream f(u8p(dir) / "shape_dump.txt");
    if (!f) { err = "cannot write to " + dir; return false; }
    f << "# " << hdr << "\n";
    char b[160];
    for (size_t i = 0; i < hooks.area.size(); i++) {
      std::snprintf(b, sizeof b, "%.10e %.10f %.10f %.10f\n", hooks.area[i], hooks.normal[i][0], hooks.normal[i][1], hooks.normal[i][2]);
      f << b;
    }
  }
  {
    std::ofstream f(u8p(dir) / "cg_dump.txt");
    f << hooks.cg.size() << " " << hooks.ncoef << " " << hooks.nphpar << "\n";
    char b[64];
    for (double v : hooks.cg) { std::snprintf(b, sizeof b, "%.12e\n", v); f << b; }
  }
  {
    std::ofstream f(u8p(dir) / "asteroid_shape.obj");
    f << "# convex shape model computed by asteroid_viewer, " << hdr << "\n# z = spin axis, unit volume-equivalent radius\n";
    char b[128];
    for (auto& v : mink.mesh.v) { std::snprintf(b, sizeof b, "v %.6f %.6f %.6f\n", v.x, v.y, v.z); f << b; }
    for (size_t t = 0; t + 2 < mink.mesh.tri.size(); t += 3)
      f << "f " << mink.mesh.tri[t] + 1 << " " << mink.mesh.tri[t + 1] + 1 << " " << mink.mesh.tri[t + 2] + 1 << "\n";
  }
  if (smoothMesh.ok) {
    std::ofstream f(u8p(dir) / "asteroid_shape_smooth.obj");
    f << "# smooth model (shape function on a finer grid) computed by asteroid_viewer, " << hdr << "\n";
    char b[128];
    for (auto& v : smoothMesh.v) { std::snprintf(b, sizeof b, "v %.6f %.6f %.6f\n", v.x, v.y, v.z); f << b; }
    for (size_t t = 0; t + 2 < smoothMesh.tri.size(); t += 3)
      f << "f " << smoothMesh.tri[t] + 1 << " " << smoothMesh.tri[t + 1] + 1 << " " << smoothMesh.tri[t + 2] + 1 << "\n";
  }
  return true;
}
