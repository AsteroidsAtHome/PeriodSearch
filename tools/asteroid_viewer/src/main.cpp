// Asteroid Viewer - OpenGL viewer for Asteroids@home period_search results.
//
//   asteroid_viewer [files or directories ...]
//
// Recognised files: period_search_in, period_search_out (several are merged),
// *.obj shape model, cg_dump.txt (best-fit parameters) and shape_dump.txt
// (Gaussian image) as written by Save shape files. Files can also be dropped
// onto the window. Without arguments the current directory is scanned.

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <fstream>
#include <string>
#include <vector>

#include "u8.h"
#include "boinc.h"
#include "cache.h"
#include "orbitfit.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <exception>
#include "data.h"
#include "gl_loader.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "compute.h"
#include "model.h"
#include "renderer.h"
#include "sbdb.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------- log + crash reporting
static FILE* gLog = nullptr;
static const auto gT0 = std::chrono::steady_clock::now();
static void logLine(const std::string& s) {
  if (!gLog) return;
  // own clock: glfwGetTime() must not be used here (its error callback logs -> recursion after glfwTerminate)
  double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - gT0).count();
  std::fprintf(gLog, "%8.2f  %s\n", t, s.c_str());
  std::fflush(gLog);
}

// ---- frame profiler: per section time, logged every 10 s when a frame was slow (or with ASTEROID_VIEWER_PROFILE=1)
struct Profiler {
  struct Acc { double sum = 0, max = 0; };
  std::map<std::string, Acc> acc;
  int frames = 0;
  double frameSum = 0, frameMax = 0, lastReport = 0;
  bool always = false;
  void add(const char* name, double ms) { auto& x = acc[name]; x.sum += ms; x.max = std::max(x.max, ms); }
  void frame(double ms, double now) {
    frames++; frameSum += ms; frameMax = std::max(frameMax, ms);
    if (now - lastReport < 10) return;
    if (always || frameMax > 50) {
      std::vector<std::pair<double, std::string>> v;
      for (auto& kv : acc) v.push_back({kv.second.max, kv.first});
      std::sort(v.rbegin(), v.rend());
      char b[160];
      std::snprintf(b, sizeof b, "perf: %d frames, avg %.1f ms, max %.1f ms; slowest parts (max/avg ms):", frames, frameSum / frames, frameMax);
      std::string line = b;
      for (size_t k = 0; k < v.size() && k < 6; k++) {
        std::snprintf(b, sizeof b, " %s %.1f/%.2f", v[k].second.c_str(), v[k].first, acc[v[k].second].sum / frames);
        line += b;
      }
      logLine(line);
    }
    acc.clear(); frames = 0; frameSum = frameMax = 0; lastReport = now;
  }
};
static Profiler gProf;
struct ProfScope {
  const char* name;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit ProfScope(const char* n) : name(n) {}
  ~ProfScope() { gProf.add(name, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
};
#define PROF_CAT2(a, b) a##b
#define PROF_CAT(a, b) PROF_CAT2(a, b)
#define PROF(name) ProfScope PROF_CAT(prof_, __LINE__)(name)

static void fatal(const std::string& what) {
  logLine("FATAL: " + what);
  std::fprintf(stderr, "FATAL: %s\n", what.c_str());
#ifdef _WIN32
  std::string logp;
  try { logp = logFilePath(); } catch (...) { logp = "(unknown)"; }
  std::string text = "Asteroid Viewer stopped:\n\n" + what + "\n\nLog: asteroid_viewer.log next to the program, or " + logp +
                     "\n\nPlease send the log file. Start with --no-boinc to skip the BOINC scan.";
  MessageBoxW(nullptr, u8widen(text).c_str(), L"Asteroid Viewer", MB_OK | MB_ICONERROR);
#endif
}

static void onTerminate() {
  std::string what = "unhandled C++ exception";
  if (auto e = std::current_exception()) {
    try { std::rethrow_exception(e); }
    catch (std::exception& x) { what += ": " + std::string(x.what()); }
    catch (...) {}
  }
  fatal(what);
  std::_Exit(3);
}

#ifdef _WIN32
static LONG WINAPI onSehCrash(EXCEPTION_POINTERS* ep) {
  char b[160];
  std::snprintf(b, sizeof b, "crash: exception 0x%08lX at address %p", (unsigned long)ep->ExceptionRecord->ExceptionCode,
                ep->ExceptionRecord->ExceptionAddress);
  fatal(b);
  return EXCEPTION_EXECUTE_HANDLER;
}
#endif

static std::string gStep = "start";  // last startup step, written into the log
static bool gNormalExit = false;
static void onExit() {  // exit() called from somewhere else (e.g. period_search's ludcmp on a singular matrix)
  if (!gNormalExit) fatal("the program called exit() unexpectedly (during: " + gStep + ")");
}

// --------------------------------------------------------------------------
struct App {
  InputData in;
  OutputData out;
  Mesh mesh;
  ModelParams model;
  GaussImage gauss;
  Facets facets;
  Fit fit;
  bool facetsFromGauss = false;
  std::vector<std::string> log;
  std::vector<std::string> pendingDrops;

  // view state
  int viewMode = 0;   // 0 = free orbit, 1 = as seen from Earth, 2 = solar system (the asteroid's orbit)
  double orbitDist = 0;  // solar system view: camera distance from the Sun [AU], 0 = fit the orbit
  // playing the observations in time order (asteroid position, orientation and brightness at each one)
  bool playObs = false;
  float obsRate = 20;          // observations per second (no orbit known)
  float daysPerSec = 30;       // time-lapse speed when the orbit is known
  double playJd = 0;           // time-lapse clock
  // orbit fitted to the observation positions (background, after loading period_search_in)
  OrbitFit orbitFit;
  std::future<OrbitFit> orbitFuture;
  std::string orbitFitPath;
  void startOrbitFit() {
    orbitFit = OrbitFit();
    orbitFitPath = in.path;
    if (!in.ok) return;
    InputData copy = in;
    orbitFuture = std::async(std::launch::async, [copy] { return fitOrbit(copy); });
  }
  void pollOrbitFit() {
    if (orbitFuture.valid() && orbitFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      orbitFit = orbitFuture.get();
      char b[200];
      if (orbitFit.ok)
        std::snprintf(b, sizeof b, "orbit from the observations: a %.3f AU, e %.3f, i %.2f deg (rms %.1e AU, %d positions)", orbitFit.a,
                      orbitFit.e, orbitFit.i, orbitFit.rms, orbitFit.points);
      else
        std::snprintf(b, sizeof b, "no orbit from the observations (rms %.2g AU)", orbitFit.rms);
      msg(b);
    }
  }
  double obsAccum = 0;
  std::vector<int> obsOrder;   // observation indices sorted by time
  std::vector<int> obsRank;    // position of each observation in obsOrder
  void ensureObsOrder() {
    if (obsOrder.size() == in.obs.size()) return;
    obsOrder.resize(in.obs.size());
    for (size_t i = 0; i < obsOrder.size(); i++) obsOrder[i] = (int)i;
    std::stable_sort(obsOrder.begin(), obsOrder.end(), [this](int x, int y) { return in.obs[x].jd < in.obs[y].jd; });
    obsRank.assign(obsOrder.size(), 0);
    for (size_t r = 0; r < obsOrder.size(); r++) obsRank[obsOrder[r]] = (int)r;
  }
  int shading = 0;    // 0 = studio, 1 = photometric
  bool edges = false, showAxes = true, sunLight = true, playing = true;
  float speed = 0.15f;    // rotations per second
  float exposure = 1.2f;
  double yaw = 0.6, pitch = 0.35, dist = 6.0;
  double zoomEarth = 1.0;
  int obsIdx = 0;
  double timeOffset = 0;  // days relative to the selected observation
  double spinAngle = 0;   // used when no rotation model is loaded
  int selectedRow = -1;
  char pathBuf[1024] = "";

  // live monitoring of running period_search calculations
  struct Watch {
    bool exists = false;
    fs::file_time_type mtime{};
    std::uintmax_t size = 0;
  };
  struct Progress {
    std::deque<std::pair<double, int>> hist;  // (app time [s], rows) whenever the row count changed
    double lastWrite = -1;                    // seconds since epoch (system clock) of last file write
  };
  bool live = true, followAxes = true, fullRange = false;
  double lastPoll = 0, lastOutScan = -1e9;
  std::vector<Watch> outWatch;
  Watch inWatch;
  std::vector<Progress> progress;

  // shape computation inside the viewer
  ShapeJob job;
  int extraTrials = 1;          // neighbouring trial periods on each side
  bool useOutConw = true;
  double targetPeriod = 0;      // [h]
  std::string savedDir;
  int pgMetric = 0;             // periodogram y: 0 chi2, 1 rms, 2 dark facet
  bool autoBest = true;         // live: recompute the best shape automatically
  bool smoothToggled = false;
  bool jobIsAuto = false;
  double lastAutoCheck = -1e9, autoDoneP = 0, autoDoneChi2 = 0, finalDoneP = 0, finalDoneChi2 = 0;
  int autoInterval = 10;        // seconds between automatic checks

  // smooth model
  bool smooth = true, shownSmoothModel = false, smoothPending = false;
  Mesh smoothMesh;
  std::future<Mesh> smoothFuture;

  // BOINC integration
  std::string boincDir;
  char boincBuf[512] = "";
  std::vector<BoincTask> tasks;
  std::string boincErr;
  double lastSlotScan = -1e9;
  BoincTask current;                          // the slot that is open (wuName empty if none)
  bool currentGone = false;                   // BOINC removed the shown task or put another one into its slot
  bool currentCached = false;                 // the shown task was opened from the finished-tasks cache
  std::time_t currentFinished = 0;            // its finish time (cached task)
  bool currentWasRunning = false;             // cached copy of the running task that was shown when it finished
  // finished-tasks cache (outside the BOINC folder)
  int cacheMax = 100;                         // tasks kept, oldest deleted first; 0 = do not cache
  std::vector<CachedTask> cached;             // cacheList(), refreshed after every change
  int namesCached = 0;                        // asteroid names in the cache (refreshed with the list)
  // running tasks seen in the slots, by WU name: the resolved project files stay readable after BOINC
  // empties the slot, until the result is reported, so a task that finished between two scans is still cached
  struct Seen { BoincTask task; std::string inPath, outPath; int total = 0; };
  std::map<std::string, Seen> seen;
  std::map<std::string, SbdbInfo> taskInfo;   // designation -> SBDB answer
  std::map<std::string, std::future<SbdbInfo>> taskLookups;
  struct SlotProgress {
    std::string wuName;
    int total = 0, rows = -1;
    std::string inPath, outPath;  // resolved BOINC links (project files)
    long long outSize = -1;       // rows are counted again only when the output grows
  };
  std::map<std::string, SlotProgress> slotProgress;  // slotDir -> rows / total trial periods
  struct ScanResult { std::vector<BoincTask> tasks; std::map<std::string, SlotProgress> prog; std::string err; };
  std::future<ScanResult> scanFuture;  // slot scan running in the background (file reads stay off the UI thread)
  std::string jobInputPath;
  bool jobFinal = false;     // an all-poles shape of a finished run (stored with the cached task)
  std::string jobWu;         // WU name of the task the job was started for
  std::string pendingShapeWu;  // final shape computed before the task was cached: store it once cached
  bool autoSwitch = true;  // switch to the most advanced running task when the shown one is complete
  int switchInterval = 60; // auto switch between tasks interval [s]
  enum SwitchStrategy { SwitchFinished = 0, SwitchRunningMax = 1, SwitchRunningMin = 2 };
  int switchStrategy = SwitchRunningMax;
  double lastOpenTime = -1e9;
  double lastProgressTime = -1e9;  // when the shown task last wrote new rows
  int lastRows = -1;               // rows of the shown task at that time

  // object info from JPL SBDB
  SbdbInfo sbdb;
  std::future<SbdbInfo> sbdbFuture;
  bool sbdbPending = false;
  std::string sbdbFor;
  char desBuf[64] = "";

  std::string dataDir() const {
    if (in.ok) return u8p(resolveBoincLink(in.path)).parent_path().u8string();
    if (!out.paths.empty()) return u8p(out.paths[0]).parent_path().u8string();
    return ".";
  }

  // where "Save shape files" writes: next to the input, but never into the BOINC folders
  // (Documents\asteroid_viewer\<WU name> for a BOINC task)
  std::string saveDir() const {
    if (current.wuName.empty()) return dataDir();
    std::string home = envUtf8("USERPROFILE");
    if (home.empty()) home = envUtf8("HOME");
    std::error_code ec;
    fs::path base = !home.empty() ? u8p(home) / "Documents" / "asteroid_viewer" : fs::temp_directory_path(ec) / "asteroid_viewer";
    return (base / current.wuName).u8string();
  }

  // designation from the file names (period_search_in_20360, BOINC ps_<des>_input_...)
  std::string guessDesignation() const {
    if (!current.designation.empty()) return current.designation;  // BOINC slot: from <wu_name>
    std::vector<std::string> names;
    if (in.ok) { names.push_back(in.path); names.push_back(resolveBoincLink(in.path)); }
    for (auto& p : out.paths) { names.push_back(p); names.push_back(resolveBoincLink(p)); }
    for (auto& n : names) {
      std::string d = designationFromFilename(n);
      if (!d.empty()) return d;
    }
    return "";
  }

  void lookupObject(const std::string& des, bool force = false) {
    if (des.empty() || sbdbPending) return;
    sbdbFor = des;
    std::snprintf(desBuf, sizeof desBuf, "%s", des.c_str());
    sbdbPending = true;
    sbdb = SbdbInfo();
    std::string dir = dataDir();
    sbdbFuture = std::async(std::launch::async, [des, dir, force] { return sbdbLookup(des, dir, force); });
  }

  void pollLookup() {
    if (sbdbPending && sbdbFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      sbdb = sbdbFuture.get();
      sbdbPending = false;
      msg(sbdb.ok ? "SBDB: " + sbdb.fullname + " - " + sbdb.orbitClass + " (" + sbdb.source + ")"
                  : "SBDB lookup of '" + sbdb.designation + "' failed: " + sbdb.error);
    }
  }

  // trial-period window of the loaded input around targetPeriod
  bool trialWindow(double& p0, double& p1, int& count) const { return trialWindow(targetPeriod, extraTrials, p0, p1, count); }
  bool trialWindow(double target, int extra, double& p0, double& p1, int& count) const {
    int N = in.totalTrials();
    if (N <= 0 || target <= 0) return false;
    int k = 0;
    for (int i = 1; i < N; i++)
      if (std::fabs(in.trialPeriod(i) - target) < std::fabs(in.trialPeriod(k) - target)) k = i;
    int k0 = std::max(0, k - extra), k1 = std::min(N - 1, k + extra);
    double step = 0.5 / (in.jdMax - in.jdMin) / 24 * in.perStepCoef;
    p0 = in.trialPeriod(k0);
    p1 = 1 / (1 / in.trialPeriod(k1) - 0.5 * step);  // half a step beyond the last trial: no rounding loss
    count = k1 - k0 + 1;
    return true;
  }

  double outConw() const {
    if (out.files.empty() || out.files[0].rows == 0 || out.files[0].rows > in.totalTrials()) return 0;
    return out.files[0].firstConw;
  }

  void startShape() { startShape(targetPeriod, extraTrials, false, false); }
  // isAuto: started by the live mode for the best row (no tab switch, message prefix)
  // singlePole: start only from (poleLambda, poleBeta), the row's own pole - about 10x faster
  void startShape(double target, int extra, bool isAuto, bool singlePole, double poleLambda = 0, double poleBeta = 0) {
    double p0, p1;
    int cnt;
    if (!in.ok || out.rows.empty() || job.running() || !trialWindow(target, extra, p0, p1, cnt)) return;
    double conw = useOutConw ? outConw() : 0;
    jobInputPath = in.path;
    job.start(resolveBoincLink(in.path), p0, p1, conw, smooth ? in.lmax : 0, in.mmax, singlePole, poleLambda, poleBeta);
    jobIsAuto = isAuto;
    jobFinal = !singlePole && runFinished();  // automatic final shape or "Compute shape" on a finished run
    jobWu = current.wuName;
    savedDir.clear();
    msg(std::string(isAuto ? (singlePole ? "auto (1 trial, start pole of the best row): " : "auto, run finished (all 10 poles): ") : "") + "computing shape: " + std::to_string(cnt) + " trial period(s) around " +
        std::to_string(target) + " h" +
        (conw > 0 ? ", convexity weight " + std::to_string(conw) : ", optimising the convexity weight"));
  }

  // live mode: every 10 s, compute the shape of the best period if it changed
  // true when the loaded period_search_out is complete for the loaded input
  bool runFinished() const {
    int N = in.totalTrials();
    return N > 0 && !out.files.empty() && out.files[0].rows >= N;
  }

  // live mode: every autoInterval seconds (and right after an automatic shape finished),
  // compute the shape of the best period if it changed. When the period_search run is complete,
  // compute the best shape once more with all 10 starting poles (exact reproduction of the row).
  void autoShape(double now) {
    if (!live || !autoBest || !in.ok || out.best < 0 || now - lastAutoCheck < autoInterval) return;
    lastAutoCheck = now;
    if (job.running()) return;
    if (in.totalTrials() <= 0 || out.files.empty() || out.files[0].rows > in.totalTrials()) return;  // input/output mismatch
    const OutputRow& b = out.rows[out.best];
    if (runFinished()) {
      if (std::fabs(b.period - finalDoneP) < 1e-9 && std::fabs(b.chi2 - finalDoneChi2) < 1e-12) return;
      finalDoneP = autoDoneP = b.period;
      finalDoneChi2 = autoDoneChi2 = b.chi2;
      startShape(b.period, 0, true, false);
      return;
    }
    if (std::fabs(b.period - autoDoneP) < 1e-9 && std::fabs(b.chi2 - autoDoneChi2) < 1e-12) return;
    autoDoneP = b.period;
    autoDoneChi2 = b.chi2;
    startShape(b.period, 0, true, true, b.lambda, b.beta);
  }


  // shown mesh: smooth model if wanted and available
  void refreshMesh(MeshRenderer& r) {
    bool useSmooth = smooth && smoothMesh.ok;
    shownSmoothModel = useSmooth;
    r.upload(useSmooth ? smoothMesh : mesh, smooth);
  }

  // smooth model for a model loaded from files (cg_dump.txt + period_search_in)
  void requestSmooth() {
    smoothMesh = Mesh();
    if (!smooth || !model.ok || model.shapeCoef.empty() || !in.ok || in.lmax <= 0 || smoothPending) return;
    smoothPending = true;
    auto coef = model.shapeCoef;
    int L = in.lmax, M = in.mmax;
    auto warm = std::make_shared<Mesh>(mesh);
    smoothFuture = std::async(std::launch::async, [coef, L, M, warm] {
      Mesh out;
      std::string err;
      if (!smoothModelMesh(coef, L, M, warm->ok ? warm.get() : nullptr, out, err)) out = Mesh();
      return out;
    });
  }

  void pollSmooth(MeshRenderer& r) {
    if (smoothPending && smoothFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      smoothMesh = smoothFuture.get();
      smoothPending = false;
      if (smoothMesh.ok) msg("smooth model ready (" + std::to_string(smoothMesh.tri.size() / 3) + " triangles)");
      refreshMesh(r);
    }
  }

  void pollJob(MeshRenderer& r) {
    if (!job.collect()) return;
    if (jobIsAuto) lastAutoCheck = -1e9;  // check again right away: the best row may have changed meanwhile
    if (job.ok && jobFinal && !jobWu.empty()) cacheShape(jobWu);  // even if another task is shown by now
    if (jobInputPath != in.path) {         // another task was opened meanwhile
      msg("discarded a shape computed for " + jobInputPath);
      return;
    }
    if (job.ok) {
      mesh = job.mink.mesh;
      mesh.path = "(computed in the viewer)";
      smoothMesh = job.smoothMesh;
      refreshMesh(r);
      model = job.model();
      gauss = job.gauss();
      recompute();
      char b[200];
      std::snprintf(b, sizeof b, "shape computed in %.0f s: P = %.6f h, lambda %.1f, beta %.1f, rms %.6f, dark facet %.2f%%",
                    job.seconds, job.hooks.bestPeriod, job.hooks.bestLambda, job.hooks.bestBeta, job.hooks.bestDev,
                    100 * job.mink.darkFacet);
      msg(b);
    } else if (job.cancelled) {
      msg("shape computation cancelled");
    } else {
      msg("ERROR shape computation: " + job.error);
    }
  }

  static std::string& gStartTabRef();

  void clearAll(MeshRenderer& r) {
    in = InputData(); out = OutputData(); mesh = Mesh(); model = ModelParams(); gauss = GaussImage();
    obsOrder.clear(); obsRank.clear();  // playObs stays on: the next task plays as well
    if (orbitFuture.valid()) orbitFuture.wait();
    orbitFuture = std::future<OrbitFit>();
    orbitFit = OrbitFit();
    orbitFitPath.clear();
    outWatch.clear(); progress.clear(); selectedRow = -1;
    smoothMesh = Mesh();
    current = BoincTask();
    currentGone = currentCached = currentWasRunning = false;
    lastProgressTime = -1e9;
    lastRows = -1;
    sbdb = SbdbInfo(); sbdbFor.clear(); desBuf[0] = 0;
    autoDoneP = autoDoneChi2 = finalDoneP = finalDoneChi2 = 0;
    lastAutoCheck = -1e9;
    targetPeriod = 0;
    obsIdx = 0; timeOffset = 0;
    refreshMesh(r);
    recompute();
  }

  void openTask(const BoincTask& task, MeshRenderer& r) {
    BoincTask t = task;  // copy: `task` may live in `tasks`
    lastOpenTime = glfwGetTime();
    msg("opening BOINC slot " + std::to_string(t.slot) + " (" + t.slotDir + ")");
    clearAll(r);
    try {
      load({t.slotDir}, r);
    } catch (std::exception& e) {
      msg(std::string("ERROR opening the task: ") + e.what());
    }
    current = t;
    currentGone = false;
    currentCached = false;
    currentWasRunning = false;
    std::string des = guessDesignation();
    if (!des.empty() && des != sbdbFor) lookupObject(des);
  }

  // ---- finished-tasks cache ---------------------------------------------------
  void refreshCache() {
    cached = cacheList();
    namesCached = namesCacheCount();
    for (auto& c : cached) requestTaskInfo(c.designation, c.dir);  // names and orbit classes for the list
  }

  static int countRows(const std::string& path) {
    std::ifstream f(u8p(path), std::ios::binary);
    if (!f) return -1;
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return (int)std::count(text.begin(), text.end(), '\n');
  }

  // stores the final shape model solution of a finished task in its cache folder (<task>/shape/)
  void cacheShape(const std::string& wu) {
    if (cacheMax <= 0) return;
    if (!cacheHas(wu)) { pendingShapeWu = wu; return; }  // the task gets cached at the next slot scan
    pendingShapeWu.clear();
    std::string dir = (u8p(cacheTaskDir(wu)) / "shape").u8string(), err;
    if (!job.save(dir, err)) { msg("ERROR caching the shape of " + wu + ": " + err); return; }
    {  // closed (written) before the list is read again below
      std::ofstream f(u8p(dir) / "solution.txt");
      char b[256];
      std::snprintf(b, sizeof b, "period=%.8f\nlambda=%.3f\nbeta=%.3f\nrms=%.6f\nrow_period=%.10f\nrow_chi2=%.10e\n", job.hooks.bestPeriod,
                    job.hooks.bestLambda, job.hooks.bestBeta, job.hooks.bestDev, finalDoneP, finalDoneChi2);
      f << b;
      if (!f.flush()) { msg("ERROR writing " + dir + "/solution.txt"); return; }
    }
    msg("cached the shape model solution of " + wu);
    refreshCache();
  }

  // copies a finished task (all trial periods written) into the cache, once
  void cacheTask(const Seen& s, int rows) {
    if (cacheMax <= 0 || s.total <= 0 || rows < s.total || cacheHas(s.task.wuName)) return;
    std::string err;
    if (!cacheStore(s.task.wuName, s.task.resultName, s.task.designation, s.inPath, s.outPath, rows, s.total, err)) {
      msg("ERROR caching " + s.task.wuName + ": " + err);
      return;
    }
    int del = cacheTrim(cacheMax);
    msg("cached finished task " + s.task.wuName + (del ? " (" + std::to_string(del) + " old removed)" : ""));
    refreshCache();
    if (pendingShapeWu == s.task.wuName && !job.running() && job.ok && jobWu == pendingShapeWu) cacheShape(pendingShapeWu);
  }

  // after a slot scan: cache the tasks that are complete, and those that left their slot complete
  void updateCache() {
    std::map<std::string, bool> present;
    for (auto& t : tasks) {
      if (!t.periodSearch || t.wuName.empty()) continue;
      present[t.wuName] = true;
      auto sp = slotProgress.find(t.slotDir);
      if (sp == slotProgress.end() || sp->second.wuName != t.wuName) continue;
      Seen& s = seen[t.wuName];
      s.task = t;
      s.total = sp->second.total;
      if (sp->second.total > 0) s.inPath = sp->second.inPath;  // readable (it was parsed)
      if (sp->second.rows >= 0) s.outPath = sp->second.outPath;
      if (sp->second.rows >= s.total && !s.inPath.empty() && !s.outPath.empty()) cacheTask(s, sp->second.rows);
    }
    for (auto it = seen.begin(); it != seen.end();) {
      if (present.count(it->first)) { ++it; continue; }
      if (!it->second.inPath.empty() && !it->second.outPath.empty())
        cacheTask(it->second, countRows(it->second.outPath));  // gone from the slot: last chance
      it = seen.erase(it);
    }
  }

  void setCacheMax(int n) {
    cacheMax = std::max(0, std::min(100000, n));
    settingsSet("cache_max", std::to_string(cacheMax));
    if (cacheMax > 0) {
      int del = cacheTrim(cacheMax);
      if (del) { msg("removed " + std::to_string(del) + " old cached task(s)"); refreshCache(); }
    }
  }

  // clears the finished tasks and the asteroid names (JPL SBDB answers); the names are looked up again
  void clearCache() {
    int n = cacheClear();
    int m = namesCacheClear();
    sbdbForgetFailures();
    taskRetryAt.clear();
    for (auto it = taskInfo.begin(); it != taskInfo.end();)  // running lookups finish on their own
      it = taskLookups.count(it->first) ? std::next(it) : taskInfo.erase(it);
    msg("cleared the cache: " + std::to_string(n) + " finished task(s), " + std::to_string(m) + " asteroid name(s)");
    refreshCache();
    namesCached = namesCacheCount();
  }

  // The shown running task has finished and is in the cache: show the cached copy instead, in place
  // (same data, so the shape and plots stay; BOINC may now delete or reuse the slot files).
  void adoptCached(MeshRenderer& r) {
    if (current.wuName.empty() || currentCached || !(currentGone || runFinished())) return;
    static double lastCheck = -1e9;  // a file check: at most once per second
    if (glfwGetTime() - lastCheck < 1.0) return;
    lastCheck = glfwGetTime();
    if (!cacheHas(current.wuName)) return;
    const CachedTask* c = nullptr;
    for (auto& x : cached)
      if (x.wuName == current.wuName) c = &x;
    if (!c) { refreshCache(); for (auto& x : cached) if (x.wuName == current.wuName) c = &x; }
    if (!c) return;
    CachedTask cc = *c;
    std::string inP = (u8p(cc.dir) / "period_search_in").u8string(), outP = (u8p(cc.dir) / "period_search_out").u8string();
    if (out.files.size() != 1) {  // unusual layout: just reopen it from the cache
      openCached(cc, r);
      currentWasRunning = true;
      return;
    }
    if (jobInputPath == in.path) jobInputPath = inP;  // a running shape computation belongs to the same task
    in.path = inP;
    out.paths = {outP};
    out.files[0].path = outP;
    current.slotDir = cc.dir;
    current.slot = -1;
    currentCached = true;
    currentGone = false;
    currentWasRunning = true;
    currentFinished = cc.finished;
    syncWatches(glfwGetTime());
    msg("task " + cc.wuName + " finished: showing its cached copy");
  }

  void openCached(const CachedTask& c0, MeshRenderer& r) {
    CachedTask c = c0;  // copy: `c0` may live in `cached`
    lastOpenTime = glfwGetTime();
    msg("opening cached task " + c.wuName + " (" + c.dir + ")");
    clearAll(r);
    try {
      load({c.dir}, r);
      fs::path sh = u8p(c.dir) / "shape";
      std::error_code ec;
      if (fs::exists(sh / "cg_dump.txt", ec)) {  // cached shape model solution: no need to compute it again
        load({(sh / "asteroid_shape.obj").u8string(), (sh / "cg_dump.txt").u8string(), (sh / "shape_dump.txt").u8string()}, r);
        if (out.best >= 0) {  // the final shape of this run is done
          finalDoneP = autoDoneP = out.rows[out.best].period;
          finalDoneChi2 = autoDoneChi2 = out.rows[out.best].chi2;
        }
        msg("loaded the cached shape model solution");
      }
    } catch (std::exception& e) {
      msg(std::string("ERROR opening the cached task: ") + e.what());
    }
    current = BoincTask();
    current.wuName = c.wuName;
    current.resultName = c.resultName;
    current.designation = c.designation;
    current.slotDir = c.dir;
    currentCached = true;
    currentWasRunning = false;
    currentFinished = c.finished;
    std::string des = guessDesignation();
    if (!des.empty() && des != sbdbFor) lookupObject(des);
  }

  void setBoincDir(const std::string& d, bool save) {
    boincDir = d;
    std::snprintf(boincBuf, sizeof boincBuf, "%s", d.c_str());
    if (save) settingsSet("boinc_dir", d);
    lastSlotScan = -1e9;
  }

  // Slot scan work: reads the slots and the progress of each period_search task. Runs in a background
  // thread (no App state is touched); `prev` lets it skip what has not changed since the last scan.
  static ScanResult scanWork(const std::string& dir, std::map<std::string, SlotProgress> prev) {
    ScanResult r;
    try {
      r.tasks = boincScanSlots(dir, r.err);
      for (auto& t : r.tasks) {
        if (!t.periodSearch) continue;
        SlotProgress sp = prev[t.slotDir];
        if (sp.wuName != t.wuName || sp.total <= 0) {  // new work unit in the slot (BOINC reuses slot folders)
          sp = SlotProgress();
          sp.wuName = t.wuName;
          sp.inPath = resolveBoincLink((u8p(t.slotDir) / "period_search_in").u8string());
          sp.outPath = resolveBoincLink((u8p(t.slotDir) / "period_search_out").u8string());
          InputData d;
          std::string err;
          sp.total = readInput(sp.inPath, d, err) ? d.totalTrials() : 0;
        }
        std::error_code ec;
        auto size = fs::file_size(u8p(sp.outPath), ec);
        if (ec) { sp.rows = -1; sp.outSize = -1; }
        else if ((long long)size != sp.outSize) {  // grew (or changed): count the rows again
          sp.outSize = (long long)size;
          sp.rows = countRows(sp.outPath);
        }
        r.prog[t.slotDir] = sp;
      }
    } catch (std::exception& e) {
      r.err = std::string("BOINC scan failed: ") + e.what();
    }
    return r;
  }

  // rescan the BOINC slots every 5 s in the background; `force` scans now and waits for the result
  void scanBoinc(double now, bool force = false) {
    if (boincDir.empty()) return;
    if (force) {
      if (scanFuture.valid()) scanFuture.wait();
      pollScan();
      lastSlotScan = now;
      applyScan(scanWork(boincDir, slotProgress));
      return;
    }
    pollScan();
    if (scanFuture.valid() || now - lastSlotScan < 5.0) return;
    lastSlotScan = now;
    std::string dir = boincDir;
    auto prev = slotProgress;
    scanFuture = std::async(std::launch::async, [dir, prev] { return scanWork(dir, prev); });
  }

  void pollScan() {
    if (scanFuture.valid() && scanFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      try {
        applyScan(scanFuture.get());
      } catch (std::exception& e) {
        boincErr = std::string("BOINC scan failed: ") + e.what();
        msg(boincErr);
      }
    }
  }

  // takes over a finished scan (UI thread): task list, progress, asteroid lookups, cache, shown task
  void applyScan(ScanResult r) {
    boincErr = r.err;
    tasks = std::move(r.tasks);
    slotProgress = std::move(r.prog);
    for (auto& t : tasks) {
      if (!t.periodSearch) continue;
      requestTaskInfo(t.designation, t.slotDir);
    }
    // most advanced first (period_search tasks before others)
    std::stable_sort(tasks.begin(), tasks.end(), [this](const BoincTask& a, const BoincTask& b) {
      if (a.periodSearch != b.periodSearch) return a.periodSearch;
      return taskFraction(a) > taskFraction(b);
    });
    updateCache();
    if (!current.wuName.empty() && !currentGone && !currentCached) {  // keep the shown task's state current
      bool found = false;
      for (auto& t : tasks)
        if (t.slotDir == current.slotDir && t.wuName == current.wuName) { current = t; found = true; }
      if (!found) markCurrentGone();
    }
  }

  // The shown task left its slot. BOINC reuses slot folders, so the slot's period_search_in/out may now
  // belong to a new work unit: stop reading them, keep the finished task's data on screen.
  void markCurrentGone() {
    if (currentGone) return;
    currentGone = true;
    msg("BOINC task " + current.wuName + " left slot " + std::to_string(current.slot) + "; its data stays shown");
  }

  // true while the open BOINC slot still holds the task that was opened (cheap: reads init_data.xml)
  bool currentSlotValid() {
    if (current.wuName.empty() || currentCached) return true;  // not a running BOINC task
    if (currentGone) return false;
    BoincTask t;
    if (!boincReadSlot(current.slotDir, t) || t.wuName != current.wuName) { markCurrentGone(); return false; }
    return true;
  }

  // fraction done of a task: rows written / trial periods, else BOINC's checkpoint value
  double taskFraction(const BoincTask& t) const {
    auto sp = slotProgress.find(t.slotDir);
    if (sp != slotProgress.end() && sp->second.wuName == t.wuName && sp->second.total > 0 && sp->second.rows >= 0)
      return std::min(1.0, (double)sp->second.rows / sp->second.total);
    return t.fractionDone >= 0 ? t.fractionDone : 0;
  }

  // the running period_search task with the most (or, with `leastDone`, the least) work done, except `skip`;
  // complete tasks (only waiting for BOINC to upload them) are skipped; nullptr if none
  const BoincTask* bestTask(const std::string& skipWu = "", bool leastDone = false) const {
    const BoincTask* best = nullptr;
    double bestF = 0;
    for (auto& t : tasks) {
      if (!t.periodSearch || t.wuName.empty() || t.wuName == skipWu) continue;
      double f = taskFraction(t);
      if (f >= 1.0) continue;
      if (leastDone && f < 0.01) continue;  // min strategy: only tasks with at least 1 % done
      bool better = !best || (leastDone ? f < bestF : f > bestF) || (f == bestF && t.slot < best->slot);
      if (better) { best = &t; bestF = f; }
    }
    return best;
  }

  static const char* strategyName(int s) {
    return s == SwitchFinished ? "finished tasks" : s == SwitchRunningMin ? "running tasks (min % done)" : "running tasks (max % done)";
  }

  // auto switch: every `switchInterval` seconds show the task the strategy picks:
  //   finished tasks             - the next task of the finished-tasks cache (newest to oldest, then around)
  //   running tasks (max % done) - the running task with the most work done
  //   running tasks (min % done) - the running task with the least work done
  // A running task that has just completed keeps its place until its final shape has been computed.
  // seconds the shown task has made no progress (no new rows), counted from when it was opened
  double idleFor() const { return glfwGetTime() - std::max(lastOpenTime, lastProgressTime); }

  // what an automatic switch waits for now ("" = nothing but the interval)
  const char* switchHold() const {
    if (job.running()) return "shape";
    if (!current.wuName.empty() && (!currentCached || currentWasRunning) && !currentGone && runFinished() && live && autoBest &&
        finalDoneP == 0)
      return "final shape";  // final shape of the run that has just finished not started yet
    return "";
  }

  // auto switch: when the shown task has made no progress for `switchInterval` seconds, show the task the
  // strategy picks; a task that keeps writing rows stays. `force` (the "next" button) switches right away
  // to another task.
  void autoSwitchTask(MeshRenderer& r, bool force = false) {
    if (!force) {
      if (!autoSwitch || idleFor() < switchInterval || *switchHold()) return;
      if (current.wuName.empty() && in.ok) return;  // files opened by hand: leave them alone
    }
    const char* how = force ? "next" : "auto switch";

    if (switchStrategy == SwitchFinished) {
      if (cached.empty()) return;
      size_t next = 0;
      if (currentCached)
        for (size_t i = 0; i < cached.size(); i++)
          if (cached[i].dir == current.slotDir) { next = (i + 1) % cached.size(); break; }
      if (currentCached && cached[next].dir == current.slotDir) { lastOpenTime = glfwGetTime(); return; }  // only one
      CachedTask c = cached[next];
      msg(std::string(how) + " (" + strategyName(switchStrategy) + "): opening " + c.wuName);
      openCached(c, r);
      return;
    }

    if (boincDir.empty()) return;
    scanBoinc(glfwGetTime(), true);  // choose from fresh progress, not the last 5 s scan
    bool showingRunning = !currentCached && !currentGone && !current.wuName.empty();
    const BoincTask* next = bestTask(force && showingRunning ? current.wuName : "", switchStrategy == SwitchRunningMin);
    if (!next) { if (!force) lastOpenTime = glfwGetTime(); return; }
    if (showingRunning && next->wuName == current.wuName) {  // already shown: keep it, wait another interval
      lastOpenTime = glfwGetTime();
      return;
    }
    for (auto& t : tasks)  // record the candidates in the log, so the choice can be checked
      if (t.periodSearch) {
        char l[64];
        std::snprintf(l, sizeof l, "  slot %d %5.1f%% ", t.slot, 100 * taskFraction(t));
        logLine(l + t.wuName);
      }
    BoincTask copy = *next;
    char pct[48];
    std::snprintf(pct, sizeof pct, " (%.1f%% done; no progress for %.0f s)", 100 * taskFraction(copy), idleFor());
    msg(std::string(how) + " (" + strategyName(switchStrategy) + "): " +
        (current.wuName.empty() ? std::string() : current.wuName + " -> ") + copy.wuName + pct);
    openTask(copy, r);
  }

  // name and orbit class of a listed task (once; after a network failure again 2 minutes later)
  std::map<std::string, double> taskRetryAt;
  void requestTaskInfo(const std::string& des, const std::string& dir) {
    if (des.empty() || taskInfo.count(des) || taskLookups.count(des)) return;
    auto r = taskRetryAt.find(des);
    if (r != taskRetryAt.end() && glfwGetTime() < r->second) return;
    taskLookups[des] = std::async(std::launch::async, [des, dir] { return sbdbLookup(des, dir); });
  }

  void pollTaskLookups() {
    for (auto it = taskLookups.begin(); it != taskLookups.end();) {
      if (it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        SbdbInfo info = it->second.get();
        if (info.netError) {  // no answer: don't remember it, ask again later
          if (!taskRetryAt.count(it->first)) msg("SBDB lookup of '" + it->first + "' failed: " + info.error);
          taskRetryAt[it->first] = glfwGetTime() + 120;
          it = taskLookups.erase(it);
          continue;
        }
        taskRetryAt.erase(it->first);
        taskInfo[it->first] = info;
        if (it->first == sbdbFor && !sbdb.ok && !sbdbPending && taskInfo[it->first].ok) sbdb = taskInfo[it->first];
        it = taskLookups.erase(it);
      } else {
        ++it;
      }
    }
  }

  bool hasRotation() const { return model.ok && in.ok; }
  double periodDays() const { return model.ok ? model.periodHours() / 24.0 : 1.0; }
  double currentJd() const { return in.ok ? in.obs[obsIdx].jd + timeOffset : 0; }

  void msg(const std::string& s) {
    log.push_back(s);
    std::fprintf(stderr, "%s\n", s.c_str());
    logLine(s);
  }

  void loadFile(const std::string& p, FileKind k, MeshRenderer& r) {
    std::string err;
    bool ok = false;
    switch (k) {
      case FileKind::Input: ok = readInput(p, in, err); if (ok) { obsIdx = 0; timeOffset = 0; } break;
      case FileKind::Output:
        if (std::find(out.paths.begin(), out.paths.end(), p) != out.paths.end()) { ok = true; break; }
        ok = readOutput(p, out, err);
        selectedRow = -1;
        break;
      case FileKind::Obj: ok = readObj(p, mesh, err); break;
      case FileKind::Model: ok = readModel(p, model, err); break;
      case FileKind::Gauss: ok = readGauss(p, gauss, err); break;
      default: err = "unrecognised file: " + p; break;
    }
    if (ok && k == FileKind::Output && out.files.back().rows == 0)
      msg("loaded " + p + " (no rows yet - calculation starting?)");
    else
      msg(ok ? "loaded " + p : "ERROR " + err);
  }

  static Watch stat(const std::string& p) {
    Watch w;
    std::error_code ec;
    std::string r = resolveBoincLink(p);
    w.exists = fs::exists(u8p(r), ec);
    if (w.exists) {
      w.mtime = fs::last_write_time(u8p(r), ec);
      w.size = fs::file_size(u8p(r), ec);
    }
    return w;
  }

  static double toSystemSeconds(fs::file_time_type t) {
    using namespace std::chrono;
    auto sys = time_point_cast<system_clock::duration>(t - fs::file_time_type::clock::now() + system_clock::now());
    return duration<double>(sys.time_since_epoch()).count();
  }

  // keep one watch/progress entry per output file, record the current state
  void syncWatches(double now) {
    outWatch.resize(out.files.size());
    progress.resize(out.files.size());
    for (size_t i = 0; i < out.files.size(); i++) {
      outWatch[i] = stat(out.files[i].path);
      if (outWatch[i].exists) progress[i].lastWrite = toSystemSeconds(outWatch[i].mtime);
      auto& h = progress[i].hist;
      if (h.empty() || h.back().second != out.files[i].rows) h.push_back({now, out.files[i].rows});
    }
    if (in.ok) inWatch = stat(in.path);
  }

  // poll files once a second; reload what changed
  // where period_search will write its output for the loaded input (it may not exist yet)
  std::vector<std::string> expectedOutputs() const {
    std::vector<std::string> c;
    if (!in.ok) return c;
    fs::path p = u8p(in.path);
    std::string name = p.filename().u8string();
    size_t k = name.rfind("_in");
    if (k != std::string::npos) {  // period_search_in_20360 -> period_search_out_20360
      std::string o = name.substr(0, k) + "_out" + name.substr(k + 3);
      c.push_back((p.parent_path() / o).u8string());
    }
    std::string plain = (p.parent_path() / "period_search_out").u8string();  // also BOINC slot folders
    if (std::find(c.begin(), c.end(), plain) == c.end()) c.push_back(plain);
    return c;
  }

  // load the expected output as soon as it appears (checked every second, Live or not)
  void scanForOutput(double now) {
    if (!in.ok || !out.files.empty() || now - lastOutScan < 1.0) return;
    lastOutScan = now;
    if (!currentSlotValid()) return;
    for (auto& c : expectedOutputs()) {
      std::error_code ec;
      if (!fs::exists(u8p(resolveBoincLink(c)), ec)) continue;
      std::string err;
      if (readOutput(c, out, err)) {
        msg("found " + c + (out.files.back().rows ? "" : " (no rows yet)"));
        selectedRow = -1;
        syncWatches(now);
      }
      return;
    }
  }

  void poll(double now) {
    scanForOutput(now);
    if (!live || now - lastPoll < 1.0) return;
    lastPoll = now;
    if (lastRows < 0) {  // first poll of this task: rows already there are no progress
      lastRows = 0;
      for (auto& f : out.files) lastRows += f.rows;
    }
    if (!currentSlotValid()) return;  // the slot now holds another work unit: don't load its files
    bool outChanged = false;
    for (size_t i = 0; i < out.files.size() && i < outWatch.size(); i++) {
      Watch w = stat(out.files[i].path);
      if (w.exists != outWatch[i].exists || w.mtime != outWatch[i].mtime || w.size != outWatch[i].size) outChanged = true;
    }
    if (outChanged) {
      int rowsNow = 0;
      for (auto& f : out.files) rowsNow += f.rows;
      if (lastRows >= 0 && rowsNow > lastRows) lastProgressTime = now;  // the run made progress
      lastRows = rowsNow;
      double selP = selectedRow >= 0 && selectedRow < (int)out.rows.size() ? out.rows[selectedRow].period : -1;
      std::string err;
      if (!reloadOutputs(out, err)) msg("ERROR " + err);
      selectedRow = -1;
      if (selP > 0)
        for (int i = 0; i < (int)out.rows.size(); i++)
          if (out.rows[i].period == selP) selectedRow = i;
      for (size_t i = 0; i < out.files.size(); i++) {
        auto& h = progress[i].hist;
        if (!h.empty() && out.files[i].rows < h.back().second) h.clear();  // file restarted
        while (h.size() > 2 && now - h.front().first > 900) h.pop_front();  // 15 min window for the rate
      }
      syncWatches(now);
    }
    if (in.ok) {
      Watch w = stat(in.path);
      if (w.mtime != inWatch.mtime || w.size != inWatch.size) {
        std::string err;
        InputData d;
        if (readInput(in.path, d, err)) { in = d; msg("reloaded " + in.path); recompute(); }
        inWatch = w;
      }
    }
  }

  void load(const std::vector<std::string>& paths, MeshRenderer& r) {
    std::vector<std::string> dirs;
    for (auto& p : paths) {
      FileKind k = classify(p);
      if (k == FileKind::Directory) dirs.push_back(p);
      else loadFile(p, k, r);
    }
    for (auto& d : dirs) {
      BoincTask t;
      if (boincReadSlot(d, t)) {
        current = t;
        msg("BOINC slot " + std::to_string(t.slot) + ": " + t.wuName);
      }
      std::vector<std::string> files;
      std::error_code ec;
      fs::directory_iterator it(u8p(d), ec), end;
      for (; !ec && it != end; it.increment(ec)) {  // never throws
        std::error_code ec3;
        if (it->is_regular_file(ec3) && !ec3) files.push_back(it->path().u8string());
      }
      if (ec) msg("cannot list " + d + ": " + ec.message());
      std::sort(files.begin(), files.end());
      bool hadOut = !out.files.empty(), hadIn = in.ok;
      for (auto& f : files) {
        FileKind k = classify(f);
        std::string name = u8p(f).filename().u8string();
        std::error_code ec2;
        bool outExists = fs::exists(u8p(resolveBoincLink(f)), ec2);  // BOINC stub may point to a file not written yet
        bool known = k == FileKind::Obj || k == FileKind::Model || k == FileKind::Gauss ||
                     (name == "period_search_in" && !hadIn) ||
                     (name.rfind("period_search_out", 0) == 0 && !hadOut && outExists);
        if (known) loadFile(f, k, r);
      }
    }
    recompute();
    smoothMesh = Mesh();
    refreshMesh(r);
    requestSmooth();
    syncWatches(glfwGetTime());
    if (in.ok && in.path != orbitFitPath) startOrbitFit();  // once per input file
    std::string des = guessDesignation();
    if (!des.empty() && des != sbdbFor) lookupObject(des);
  }

  void recompute() {
    facetsFromGauss = gauss.ok;
    facets = gauss.ok ? ::facetsFromGauss(gauss) : (mesh.ok ? facetsFromMesh(mesh) : Facets());
    fit = Fit();
    if (in.ok && model.ok && !facets.area.empty()) {
      fit = fitLightcurves(in, facets, model);
      msg("model rms " + std::to_string(fit.rms) + (facetsFromGauss ? " (Gaussian image)" : " (OBJ facets)"));
    }
    if (obsIdx >= (int)in.obs.size()) obsIdx = 0;
    if (!hasRotation() && viewMode == 1) viewMode = 0;
  }

  // body -> world rotation. World = ecliptic frame when a rotation model is loaded.
  M3 bodyToWorld() const {
    if (hasRotation())
      return eclToBody(model, in.epoch(), in.phi0deg * M_PI / 180, currentJd()).T();
    M3 r;
    double c = std::cos(spinAngle), s = std::sin(spinAngle);
    r.m[0][0] = c; r.m[0][1] = -s; r.m[1][0] = s; r.m[1][1] = c;
    return r;
  }

  // raw model brightness -> relative brightness of the selected lightcurve
  double relScale() const {
    if (!fit.ok) return 1;
    int lc = in.obs[obsIdx].lc;
    return fit.lcScale[lc] / in.lcs[lc].mean;
  }
};

static App* gApp = nullptr;
static std::string gStartTab;  // --tab: select a tab on the first frame
std::string& App::gStartTabRef() { return gStartTab; }
static ImGuiTabItemFlags tabFlag(const char* name) {
  return gStartTab == name ? ImGuiTabItemFlags_SetSelected : 0;
}

static void dropCallback(GLFWwindow*, int n, const char** paths) {
  for (int i = 0; i < n; i++) gApp->pendingDrops.push_back(paths[i]);
}

static void* getProc(const char* name) { return (void*)glfwGetProcAddress(name); }

// --------------------------------------------------------------------------
// calendar date (UT) of a Julian date (Meeus, Astronomical Algorithms ch. 7; any epoch, no time_t)
static void jdToCalendar(double jd, int& y, int& mo, int& d, int& h, int& mi) {
  double z = std::floor(jd + 0.5), f = jd + 0.5 - z;
  double a = z;
  if (z >= 2299161) { double al = std::floor((z - 1867216.25) / 36524.25); a = z + 1 + al - std::floor(al / 4); }
  double b = a + 1524, c = std::floor((b - 122.1) / 365.25), dd = std::floor(365.25 * c), e = std::floor((b - dd) / 30.6001);
  double day = b - dd - std::floor(30.6001 * e) + f;
  d = (int)day;
  mo = (int)(e < 14 ? e - 1 : e - 13);
  y = (int)(mo > 2 ? c - 4716 : c - 4715);
  int minutes = (int)std::lround((day - d) * 1440);
  if (minutes >= 1440) minutes = 1439;  // keep the same day
  h = minutes / 60; mi = minutes % 60;
}
static std::string jdDate(double jd, bool withTime = true) {
  int y, mo, d, h, mi;
  jdToCalendar(jd, y, mo, d, h, mi);
  char b[40];
  if (withTime) std::snprintf(b, sizeof b, "%04d-%02d-%02d %02d:%02d UT", y, mo, d, h, mi);
  else std::snprintf(b, sizeof b, "%04d-%02d-%02d", y, mo, d);
  return b;
}
// ImPlot axis labels: JD - 2400000 shown as a date
// (data: the visible span in days; long spans show year-month only)
static int jdAxisFormatter(double v, char* buf, int size, void* data) {
  int y, mo, d, h, mi;
  jdToCalendar(v + 2400000, y, mo, d, h, mi);
  double span = data ? *(const double*)data : 0;
  if (span > 3 * 365) return std::snprintf(buf, size, "%04d-%02d", y, mo);
  if (span < 3) return std::snprintf(buf, size, "%02d-%02d %02d:%02d", mo, d, h, mi);
  return std::snprintf(buf, size, "%04d-%02d-%02d", y, mo, d);
}

struct Projector {
  M4 vp;
  float x0, y0, w, h;
  bool project(V3 p, ImVec2& s) const {
    float c[4];
    for (int i = 0; i < 4; i++) c[i] = vp(i, 0) * (float)p.x + vp(i, 1) * (float)p.y + vp(i, 2) * (float)p.z + vp(i, 3);
    if (c[3] <= 1e-6f) return false;
    s = ImVec2(x0 + (c[0] / c[3] * 0.5f + 0.5f) * w, y0 + (0.5f - c[1] / c[3] * 0.5f) * h);
    return true;
  }
};

static void arrow(ImDrawList* dl, const Projector& pr, V3 from, V3 to, ImU32 col, const char* label) {
  ImVec2 a, b;
  if (!pr.project(from, a) || !pr.project(to, b)) return;
  dl->AddLine(a, b, col, 2.5f);
  ImVec2 d(b.x - a.x, b.y - a.y);
  float l = std::sqrt(d.x * d.x + d.y * d.y);
  if (l > 1) {
    d = ImVec2(d.x / l, d.y / l);
    ImVec2 n(-d.y, d.x);
    dl->AddTriangleFilled(ImVec2(b.x + d.x * 9, b.y + d.y * 9), ImVec2(b.x + n.x * 5, b.y + n.y * 5),
                          ImVec2(b.x - n.x * 5, b.y - n.y * 5), col);
  }
  if (label) dl->AddText(ImVec2(b.x + 8, b.y - 8), col, label);
}

// ---- solar system view (main 3D view): orbits as projected lines, the shape model drawn at the asteroid --------
struct Kepler;
static V3 asteroidHelioPos(const App& a, double jd, bool& ok);
static void drawSolarSystem(const App& a, const Projector& pr, double jd);

static void writePPM(const char* path, int w, int h) {
  std::vector<unsigned char> px((size_t)w * h * 4);
  glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
  FILE* f = fopenUtf8(path, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%d %d\n255\n", w, h);
  for (int y = h - 1; y >= 0; y--)
    for (int x = 0; x < w; x++) std::fwrite(&px[((size_t)y * w + x) * 4], 1, 3, f);
  std::fclose(f);
}


static std::string fmtDuration(double s) {
  char b[64];
  if (s < 0) return "?";
  if (s < 90) std::snprintf(b, sizeof b, "%.0f s", s);
  else if (s < 5400) std::snprintf(b, sizeof b, "%.0f min", s / 60);
  else if (s < 2 * 86400) std::snprintf(b, sizeof b, "%dh %02dm", (int)(s / 3600), (int)std::fmod(s / 60, 60));
  else std::snprintf(b, sizeof b, "%.1f days", s / 86400);
  return b;
}

// overall fraction done (first file), -1 if unknown; used for the window title
static double overallFraction(const App& a) {
  int N = a.in.totalTrials();
  if (N <= 0 || a.out.files.empty() || a.out.files[0].rows > N) return -1;
  return std::min(1.0, (double)a.out.files[0].rows / N);
}

static void drawProgress(App& a) {
  int N = a.in.totalTrials();
  double nowApp = glfwGetTime();
  double nowSys = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  for (size_t i = 0; i < a.out.files.size(); i++) {
    const auto& f = a.out.files[i];
    const App::Progress* pr = i < a.progress.size() ? &a.progress[i] : nullptr;
    ImGui::PushID((int)i);
    std::string name = u8p(f.path).filename().u8string();
    double age = pr && pr->lastWrite > 0 ? nowSys - pr->lastWrite : -1;
    // rows per second from the rows seen arriving while the viewer was open
    double rate = 0;
    if (pr && pr->hist.size() >= 2) {
      auto& h = pr->hist;
      double dt = h.back().first - h.front().first;
      if (dt > 0) rate = (h.back().second - h.front().second) / dt;
    }
    // does the loaded period_search_in describe this output? (rows must lie inside its scan)
    bool match = N > 0 && f.rows <= N;
    if (match && f.rows > 0) {
      double step = a.in.trialPeriod(1) - a.in.trialPeriod(0);
      double lo = 1e300, hi = -1e300;
      for (auto& r : a.out.rows) { lo = std::min(lo, r.period); hi = std::max(hi, r.period); }
      if (a.out.files.size() == 1 && (lo < a.in.perStart - 3 * step || hi > a.in.perEnd + 3 * step)) match = false;
    }
    if (N > 0 && !match) {
      ImGui::Text("%s: %d rows", name.c_str(), f.rows);
      ImGui::TextColored(ImVec4(1, 0.6f, 0.35f, 1), "period_search_in (scan %.4f - %.4f h) does not match this output;",
                         a.in.perStart, a.in.perEnd);
      ImGui::TextColored(ImVec4(1, 0.6f, 0.35f, 1), "load the input file used for this run to see the progress.");
    } else if (N > 0) {
      double frac = std::min(1.0, (double)f.rows / N);
      char label[96];
      std::snprintf(label, sizeof label, "%d / %d periods  (%.1f%%)", f.rows, N, 100 * frac);
      ImVec4 col = f.rows >= N ? ImVec4(0.3f, 0.75f, 0.4f, 1) : ImVec4(0.26f, 0.59f, 0.98f, 1);
      ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
      ImGui::ProgressBar((float)frac, ImVec2(-1, 0), label);
      ImGui::PopStyleColor();
      if (a.out.files.size() > 1) ImGui::TextDisabled("%s", name.c_str());
      if (f.rows >= N) {
        ImGui::TextColored(ImVec4(0.4f, 0.85f, 0.5f, 1), "finished");
        if (age >= 0) { ImGui::SameLine(); ImGui::TextDisabled("(last row written %s ago)", fmtDuration(age).c_str()); }
      } else {
        double next = a.in.trialPeriod(f.rows);
        if (f.rows == 0)
          ImGui::TextColored(ImVec4(1, 0.8f, 0.4f, 1), "no rows yet - period_search optimises the convexity weight first");
        else
          ImGui::Text("now computing trial P = %.5f h   (scan %.4f - %.4f h)", next, a.in.perStart, a.in.perEnd);
        if (rate > 0) {
          ImGui::Text("speed %.2f periods/min   remaining ~%s", rate * 60, fmtDuration((N - f.rows) / rate).c_str());
        } else if (a.live) {
          ImGui::TextDisabled("speed: measured once new rows arrive");
        }
        if (age >= 0) {
          bool stale = age > 600 && (rate <= 0 || age > 5.0 / std::max(rate, 1e-9));
          ImGui::TextColored(stale ? ImVec4(1, 0.55f, 0.35f, 1) : ImVec4(0.6f, 0.6f, 0.6f, 1),
                             stale ? "last row written %s ago - stopped or suspended?" : "last row written %s ago",
                             fmtDuration(age).c_str());
        }
      }
    } else {
      ImGui::Text("%s: %d rows", name.c_str(), f.rows);
      ImGui::TextDisabled("load the matching period_search_in to see the progress");
    }
    ImGui::PopID();
  }
  ImGui::Checkbox("Live (re-read files every second)", &a.live);
  ImGui::SameLine();
  ImGui::BeginDisabled(!a.live);
  ImGui::Checkbox("auto shape of best period, every", &a.autoBest);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70);
  if (ImGui::DragInt("##autoint", &a.autoInterval, 0.5f, 1, 3600, "%d s", ImGuiSliderFlags_AlwaysClamp)) a.lastAutoCheck = -1e9;
  if (ImGui::IsItemDeactivatedAfterEdit()) settingsSet("auto_interval", std::to_string(a.autoInterval));
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("Check interval (drag or double-click to type, 1-3600 s).\n"
                      "While Live is on: if the best period changed, compute its shape - fast: only the trial period of the\n"
                      "best row and one starting pole (the row's own lambda/beta). Checked again right after each automatic\n"
                      "shape. When the period_search run is complete: once more with all 10 poles (exact reproduction).");
  ImGui::EndDisabled();
  if (a.live && a.autoBest && a.out.best >= 0) {
    ImGui::SameLine();
    double left = a.autoInterval - (glfwGetTime() - a.lastAutoCheck);
    if (a.job.running()) ImGui::TextDisabled("(computing)");
    else if (left > 0 && left < 1e6) ImGui::TextDisabled("(next %.0f s)", left);
  }
  ImGui::Checkbox("Full scan range", &a.fullRange);
  ImGui::SameLine();
  ImGui::Checkbox("Auto-fit", &a.followAxes);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fit the axes of all plots to their data (turn off to zoom/pan freely)");
  (void)nowApp;
}



// progress of the running shape computation (also shown with the section collapsed)
static void drawJobProgress(App& a) {
  auto s = a.job.snapshot();
  double frac = 0;
  char label[160];
  if (s.phase == "building the polyhedron") {
    frac = 1;
    std::snprintf(label, sizeof label, "Minkowski: iteration %d, area mismatch %.1e", s.minkIter, s.minkMismatch);
  } else if (s.stage == 0) {
    frac = (s.n - 1 + (s.pole - 1 + s.iter / 50.0) / 10.0) / 10.0;
    std::snprintf(label, sizeof label, "convexity weight: trial %d, pole %d/10, iteration %d, rms %.6f", s.n, s.pole, s.iter, s.rms);
  } else {
    int idx = std::max(0, s.n - 1);
    frac = s.total > 0 ? (idx + (std::max(0, s.pole - 1) + std::min(1.0, s.iter / 50.0)) / s.nPoles) / s.total : 0;
    std::snprintf(label, sizeof label, "P %.5f h (%d/%d)  pole %d/%d  iteration %d  rms %.6f", s.trialP, s.n, s.total, s.pole,
                  s.nPoles, s.iter, s.rms);
  }
  ImGui::ProgressBar((float)std::min(1.0, frac), ImVec2(-80, 0), label);
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) a.job.cancel();
  ImGui::TextDisabled("%s%s, %.0f s", a.jobIsAuto ? "auto (best period): " : "", s.phase.c_str(), s.elapsed);
}

static void drawCompute(App& a) {
  // collapsible; the progress bar of a running computation stays visible when collapsed
  if (!ImGui::CollapsingHeader("Shape model solution (period_search + Minkowski)###compute", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (a.job.running()) drawJobProgress(a);
    return;
  }
  if (!a.in.ok) { ImGui::TextDisabled("load period_search_in to compute a shape here"); return; }
  if (a.targetPeriod <= 0) {
    if (a.selectedRow >= 0) a.targetPeriod = a.out.rows[a.selectedRow].period;
    else if (a.out.best >= 0) a.targetPeriod = a.out.rows[a.out.best].period;
    else if (a.model.ok) a.targetPeriod = a.model.periodHours();
    else a.targetPeriod = 0.5 * (a.in.perStart + a.in.perEnd);
  }
  bool running = a.job.running();
  ImGui::BeginDisabled(running);
  ImGui::SetNextItemWidth(130);
  ImGui::InputDouble("period [h]", &a.targetPeriod, 0, 0, "%.6f");
  if (a.out.best >= 0) {
    ImGui::SameLine();
    if (ImGui::SmallButton("best")) a.targetPeriod = a.out.rows[a.out.best].period;
  }
  if (a.selectedRow >= 0) {
    ImGui::SameLine();
    if (ImGui::SmallButton("selected")) a.targetPeriod = a.out.rows[a.selectedRow].period;
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(90);
  ImGui::SliderInt("+- trials", &a.extraTrials, 0, 5);
  double p0, p1;
  int cnt = 0;
  bool okWin = a.trialWindow(p0, p1, cnt);
  double cw = a.outConw();
  if (cw > 0) ImGui::Checkbox(("convexity weight " + std::to_string(cw).substr(0, 4) + " from period_search_out").c_str(), &a.useOutConw);
  ImGui::EndDisabled();
  if (cw <= 0 || !a.useOutConw) ImGui::TextDisabled("the convexity weight is optimised first (adds up to a few minutes)");
  if (okWin) {
    double step = 0.5 / (a.in.jdMax - a.in.jdMin) / 24 * a.in.perStepCoef;
    ImGui::TextDisabled("%d trial period(s): %.5f - %.5f h, 10 starting poles each", cnt, p0, 1 / (1 / p1 + 0.5 * step));
  }

  if (!running) {
    bool haveOut = !a.out.rows.empty();
    ImGui::BeginDisabled(!okWin || !haveOut);
    if (ImGui::Button("Compute shape")) a.startShape();
    ImGui::EndDisabled();
    if (!haveOut) {
      ImGui::SameLine();
      ImGui::TextDisabled("available once period_search_out has rows");
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Load the period_search_out of this run (or wait for its first row while it is running).");
    }
    if (a.live && a.autoBest && a.out.best >= 0) {
      ImGui::SameLine();
      ImGui::TextDisabled("live: best-period shape updated automatically");
    }
    if (a.job.ok && !a.job.hooks.area.empty()) {
      ImGui::SameLine();
      char dir[64];
      std::snprintf(dir, sizeof dir, "shape_%.4f", a.job.hooks.bestPeriod);
      std::string target = (u8p(a.saveDir()) / dir).u8string();
      if (ImGui::Button("Save shape files")) {
        std::string err;
        if (a.job.save(target, err)) { a.savedDir = target; a.msg("saved " + target); }
        else a.msg("ERROR " + err);
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("asteroid_shape.obj, shape_dump.txt, cg_dump.txt -> %s", target.c_str());
      if (!a.savedDir.empty()) { ImGui::SameLine(); ImGui::TextDisabled("saved to %s", a.savedDir.c_str()); }
    }
  } else {
    drawJobProgress(a);
  }
}


// ---- orbits (heliocentric ecliptic J2000, AU) -----------------------------------------
struct Kepler { double a, e, i, om, w, M0, n, epoch; };  // deg, deg/day, JD

static V3 keplerPos(const Kepler& k, double E) {  // position for eccentric anomaly E [rad]
  const double d = M_PI / 180;
  double xo = k.a * (std::cos(E) - k.e), yo = k.a * std::sqrt(1 - k.e * k.e) * std::sin(E);
  double cw = std::cos(k.w * d), sw = std::sin(k.w * d), co = std::cos(k.om * d), so = std::sin(k.om * d);
  double ci = std::cos(k.i * d), si = std::sin(k.i * d);
  double x1 = cw * xo - sw * yo, y1 = sw * xo + cw * yo;  // in the orbit plane, from the node
  return V3{co * x1 - so * ci * y1, so * x1 + co * ci * y1, si * y1};
}

static V3 keplerAt(const Kepler& k, double jd) {
  double M = std::fmod((k.M0 + k.n * (jd - k.epoch)) * M_PI / 180, 2 * M_PI);
  double E = M;
  for (int it = 0; it < 30; it++) E -= (E - k.e * std::sin(E) - M) / (1 - k.e * std::cos(E));
  return keplerPos(k, E);
}

// planets: JPL "Approximate positions of the planets" (J2000 mean elements, valid 1800-2050)
static Kepler planetElements(int p) {
  struct P { double a, e, i, L, varpi, om, Ldot; };
  static const P t[] = {{0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593, 149472.67411175},
                        {0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255, 58517.81538729},
                        {1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0, 35999.37244981},
                        {1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891, 19140.30268499},
                        {5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909, 3034.74612775}};
  const P& q = t[p];
  return Kepler{q.a, q.e, q.i, q.om, q.varpi - q.om, q.L - q.varpi, q.Ldot / 36525.0, 2451545.0};
}

static void plotOrbitLine(const char* name, const Kepler& k, int view, ImVec4 col, float w = 1.5f) {
  const int N = 361;
  std::vector<double> x(N), y(N);
  for (int j = 0; j < N; j++) {
    V3 p = keplerPos(k, 2 * M_PI * j / (N - 1));
    x[j] = p.x;
    y[j] = view == 0 ? p.y : p.z;
  }
  ImPlot::SetNextLineStyle(col, w);
  ImPlot::PlotLine(name, x.data(), y.data(), N);
}

// heliocentric position of the asteroid at `jd`: from period_search_in at the shown observation (exact),
// otherwise from the JPL SBDB elements
// the asteroid's orbit: fitted to the observation positions (preferred: it goes through them), else JPL's elements
static bool asteroidKepler(const App& a, Kepler& k, const char** source = nullptr) {
  if (a.orbitFit.ok) {
    k = Kepler{a.orbitFit.a, a.orbitFit.e, a.orbitFit.i, a.orbitFit.om, a.orbitFit.w, a.orbitFit.M0, a.orbitFit.n, a.orbitFit.epoch};
    if (source) *source = "fitted to the observations";
    return true;
  }
  if (a.sbdb.hasOrbit) {
    k = Kepler{a.sbdb.ea, a.sbdb.ee, a.sbdb.ei, a.sbdb.eom, a.sbdb.ew, a.sbdb.ema, a.sbdb.en, a.sbdb.epoch};
    if (source) *source = "JPL SBDB";
    return true;
  }
  return false;
}

// at an observation: its exact position from period_search_in; between observations: on the orbit
static bool atObservation(const App& a) { return a.in.ok && std::fabs(a.timeOffset) < 0.02; }

static V3 asteroidHelioPos(const App& a, double jd, bool& ok) {
  ok = true;
  if (atObservation(a)) return V3{} - a.in.obs[a.obsIdx].sun;
  Kepler k;
  if (asteroidKepler(a, k)) return keplerAt(k, jd);
  if (a.in.ok) return V3{} - a.in.obs[a.obsIdx].sun;
  ok = false;
  return V3{};
}

static V3 earthHelioPos(const App& a, double jd) {
  if (atObservation(a)) return a.in.obs[a.obsIdx].earth - a.in.obs[a.obsIdx].sun;
  return keplerAt(planetElements(2), jd);
}

static void projLine(ImDrawList* dl, const Projector& pr, V3 p, V3 q, ImU32 col, float w) {
  ImVec2 a, b;
  if (pr.project(p, a) && pr.project(q, b)) dl->AddLine(a, b, col, w);
}

static void projOrbit(ImDrawList* dl, const Projector& pr, const Kepler& k, ImU32 col, ImU32 colBelow, float w) {
  const int N = 360;
  V3 prev = keplerPos(k, 0);
  for (int j = 1; j <= N; j++) {
    V3 p = keplerPos(k, 2 * M_PI * j / N);
    projLine(dl, pr, prev, p, (prev.z + p.z) >= 0 ? col : colBelow, w);  // dimmer below the ecliptic
    prev = p;
  }
}

// date, observation and its brightness at the top right of the 3D view (all views)
static void drawPlayInfo(const App& a, const Projector& pr, double jd) {
  ImDrawList* dl = ImGui::GetBackgroundDrawList();
  {
    std::string d = jdDate(jd);
    ImVec2 ts = ImGui::CalcTextSize(d.c_str());
    dl->AddText(ImVec2(pr.x0 + pr.w - ts.x - 14, pr.y0 + 12), IM_COL32(220, 220, 230, 230), d.c_str());
    if (a.in.ok && (a.playObs || !a.obsOrder.empty())) {  // which observation, and its brightness
      int n = (int)a.in.obs.size(), r = a.obsRank.size() == (size_t)n ? a.obsRank[a.obsIdx] : a.obsIdx;
      const Observation& o = a.in.obs[a.obsIdx];
      char l1[96], l2[96];
      std::snprintf(l1, sizeof l1, "observation %d / %d", r + 1, n);
      bool between = !atObservation(a) && std::fabs(a.timeOffset) > 0.5;  // time-lapse between two observations
      if (between && a.obsRank.size() == (size_t)n) {
        int nr = r + 1 < n ? r + 1 : 0;
        double nj = a.in.obs[a.obsOrder[nr]].jd;
        std::snprintf(l1, sizeof l1, "between observations %d and %d", r + 1, nr + 1);
        std::snprintf(l2, sizeof l2, "next: %s (in %.0f days)", jdDate(nj, false).c_str(), std::max(0.0, nj - jd));
      } else if (a.fit.ok)
        std::snprintf(l2, sizeof l2, "observed %.3f   model %.3f", a.fit.relObs[a.obsIdx], a.fit.relModel[a.obsIdx]);
      else
        std::snprintf(l2, sizeof l2, "observed %.3f", o.brightness / a.in.lcs[o.lc].mean);
      std::string l3 = a.in.lcs[o.lc].label;
      float y = pr.y0 + 12 + ts.y + 4;
      for (const char* l : {(const char*)l1, (const char*)l2, l3.c_str()}) {
        ImVec2 s2 = ImGui::CalcTextSize(l);
        dl->AddText(ImVec2(pr.x0 + pr.w - s2.x - 14, y), IM_COL32(170, 180, 200, 220), l);
        y += s2.y + 2;
      }
      if (a.fit.ok && !between) {  // brightness bars: observed (green) and model (white), relative brightness 0..2
        float x0 = pr.x0 + pr.w - 174, bw = 160;
        auto bar = [&](double v, ImU32 c) {
          float w = (float)std::clamp(v / 2.0, 0.0, 1.0) * bw;
          dl->AddRectFilled(ImVec2(x0, y + 2), ImVec2(x0 + w, y + 9), c);
          y += 11;
        };
        dl->AddRect(ImVec2(x0 - 1, y + 1), ImVec2(x0 + bw + 1, y + 21), IM_COL32(120, 130, 150, 120));
        bar(a.fit.relObs[a.obsIdx], IM_COL32(150, 230, 150, 220));
        bar(a.fit.relModel[a.obsIdx], IM_COL32(235, 235, 235, 220));
      }
    }
  }
}

static void drawSolarSystem(const App& a, const Projector& pr, double jd) {
  ImDrawList* dl = ImGui::GetBackgroundDrawList();
  ImVec2 sp;
  // ecliptic reference: faint circles every AU up to Jupiter, vernal equinox direction
  for (int r = 1; r <= 5; r++) {
    V3 prev{(double)r, 0, 0};
    for (int j = 1; j <= 120; j++) {
      double t = 2 * M_PI * j / 120;
      V3 p{r * std::cos(t), r * std::sin(t), 0};
      projLine(dl, pr, prev, p, IM_COL32(120, 130, 150, 28), 1);
      prev = p;
    }
  }
  projLine(dl, pr, V3{0, 0, 0}, V3{5.6, 0, 0}, IM_COL32(120, 130, 150, 60), 1);
  if (pr.project(V3{5.7, 0, 0}, sp)) dl->AddText(sp, IM_COL32(150, 160, 180, 160), "vernal equinox");
  static const char* names[] = {"Mercury", "Venus", "Earth", "Mars", "Jupiter"};
  static const ImU32 cols[] = {IM_COL32(150, 150, 150, 150), IM_COL32(215, 190, 130, 150), IM_COL32(80, 150, 255, 220),
                               IM_COL32(230, 100, 80, 170), IM_COL32(205, 165, 115, 170)};
  for (int p = 0; p < 5; p++) {
    Kepler k = planetElements(p);
    projOrbit(dl, pr, k, cols[p], cols[p], p == 2 ? 1.6f : 1.0f);
    if (pr.project(keplerAt(k, jd), sp)) {
      dl->AddCircleFilled(sp, p == 2 ? 5 : 4, cols[p] | IM_COL32_A_MASK);
      dl->AddText(ImVec2(sp.x + 7, sp.y - 7), cols[p] | IM_COL32_A_MASK, names[p]);
    }
  }
  if (pr.project(V3{0, 0, 0}, sp)) {
    dl->AddCircleFilled(sp, 8, IM_COL32(255, 215, 80, 255));
    dl->AddCircle(sp, 12, IM_COL32(255, 215, 80, 90), 0, 2);
  }
  Kepler ak;
  if (asteroidKepler(a, ak)) {
    Kepler k = ak;
    // the orbit lit up where the asteroid can be observed from the Earth at this moment: solar elongation
    // (angle Sun - Earth - asteroid) above 90 deg = in the night sky, reflecting sunlight towards us; the arc
    // brightens towards opposition and moves with the Earth
    V3 E = earthHelioPos(a, jd);
    const int N = 360;
    V3 prev = keplerPos(k, 0);
    for (int j = 1; j <= N; j++) {
      V3 p = keplerPos(k, 2 * M_PI * j / N), mid = (prev + p) * 0.5;
      double el = std::acos(std::clamp(dot(normalize(V3{} - E), normalize(mid - E)), -1.0, 1.0)) * 180 / M_PI;
      double t = std::clamp((el - 90) / 60, 0.0, 1.0);
      if (el < 90) projLine(dl, pr, prev, p, IM_COL32(150, 150, 160, 110), 1.2f);
      else projLine(dl, pr, prev, p, IM_COL32(255, 245 - (int)(35 * t), 200 - (int)(110 * t), 150 + (int)(105 * t)), 2.0f + 1.0f * (float)t);
      prev = p;
    }
    if (pr.project(keplerAt(k, jd), sp)) dl->AddCircle(sp, 9, IM_COL32(255, 220, 130, 230), 0, 1.5f);  // where it is now
    // perihelion
    if (pr.project(keplerPos(k, 0), sp)) dl->AddText(ImVec2(sp.x + 5, sp.y), IM_COL32(200, 200, 200, 160), "q");
  }
  if (a.in.ok) {  // asteroid positions at the observations (dots) and the shown observation's line of sight
    size_t step = std::max<size_t>(1, a.in.obs.size() / 1500);
    for (size_t k = 0; k < a.in.obs.size(); k += step)  // while playing: observations already passed brighter
      if (pr.project(V3{} - a.in.obs[k].sun, sp))
        dl->AddCircleFilled(sp, 1.6f, a.playObs && a.in.obs[k].jd > jd ? IM_COL32(150, 230, 150, 50) : IM_COL32(150, 230, 150, 150));
    bool okp;
    V3 ast = asteroidHelioPos(a, jd, okp), earth = earthHelioPos(a, jd);
    projLine(dl, pr, earth, ast, IM_COL32(90, 160, 255, 150), 1.2f);
    projLine(dl, pr, V3{0, 0, 0}, ast, IM_COL32(255, 215, 80, 70), 1.0f);
    if (pr.project(earth, sp)) dl->AddCircle(sp, 7, IM_COL32(90, 160, 255, 255), 0, 1.5f);
  }
  bool ok;
  V3 ast = asteroidHelioPos(a, jd, ok);
  if (ok && pr.project(ast, sp)) {
    if (!a.mesh.ok) dl->AddCircleFilled(sp, 5, IM_COL32(240, 70, 70, 255));
    std::string name = a.sbdb.ok ? a.sbdb.fullname : std::string("asteroid");
    dl->AddText(ImVec2(sp.x + 12, sp.y + 6), IM_COL32(255, 220, 130, 255), name.c_str());
  }
}

// scatter whose legend entry has the marker's colour
static void orbitPoints(const char* name, const double* x, const double* y, int n, ImPlotMarker m, float size, ImVec4 col) {
  ImPlot::SetNextLineStyle(col);
  ImPlot::SetNextMarkerStyle(m, size, col, 0, col);
  ImPlot::PlotScatter(name, x, y, n);
}

// Line decimation for long series (x sorted): per screen column keep the min and the max, plus the first and last
// point and the global extremes (so auto-fit sees the same range). Keeps big periodograms fast to draw.
static void decimateMinMax(const std::vector<double>& x, const std::vector<double>& y, double x0, double x1, int buckets,
                           std::vector<double>& ox, std::vector<double>& oy) {
  size_t n = x.size();
  ox.clear(); oy.clear();
  if (n == 0) return;
  size_t gmin = 0, gmax = 0;
  for (size_t i = 1; i < n; i++) { if (y[i] < y[gmin]) gmin = i; if (y[i] > y[gmax]) gmax = i; }
  std::vector<size_t> keep = {0, n - 1, gmin, gmax};
  size_t i0 = std::lower_bound(x.begin(), x.end(), x0) - x.begin(), i1 = std::upper_bound(x.begin(), x.end(), x1) - x.begin();
  if (i0 > 0) i0--;
  if (i1 < n) i1++;
  double w = (x1 - x0) / std::max(1, buckets);
  for (size_t i = i0; i < i1;) {
    long b = w > 0 ? (long)std::floor((x[i] - x0) / w) : 0;
    size_t lo = i, hi = i, j = i;
    for (; j < i1 && (w > 0 ? (long)std::floor((x[j] - x0) / w) : 0) == b; j++) {
      if (y[j] < y[lo]) lo = j;
      if (y[j] > y[hi]) hi = j;
    }
    keep.push_back(lo);
    keep.push_back(hi);
    i = j;
  }
  std::sort(keep.begin(), keep.end());
  keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
  for (size_t k : keep) { ox.push_back(x[k]); oy.push_back(y[k]); }
}

// all plots follow the Auto-fit checkbox (on by default)
static ImPlotAxisFlags fitFlag(const App& a) { return a.followAxes ? ImPlotAxisFlags_AutoFit : 0; }

static void drawBoinc(App& a, MeshRenderer& rend) {
  bool any = false;
  for (auto& t : a.tasks) any = any || t.periodSearch;
  char hdr[96];
  int n = 0;
  for (auto& t : a.tasks) n += t.periodSearch;
  std::snprintf(hdr, sizeof hdr, "BOINC tasks (%d running)###boinc", n);
  // auto switch settings: always visible (also with the task list collapsed)
  if (ImGui::Checkbox("auto switch", &a.autoSwitch)) {
    settingsSet("auto_switch", a.autoSwitch ? "1" : "0");
    a.lastOpenTime = -1e9;  // apply right away
  }
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Every interval, show the task the strategy picks.\n"
                      "A running task that has just completed stays until its final shape is computed.");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(220);
  ImGui::BeginDisabled(!a.autoSwitch);
  if (ImGui::BeginCombo("##strategy", App::strategyName(a.switchStrategy))) {
    for (int i : {App::SwitchFinished, App::SwitchRunningMax, App::SwitchRunningMin}) {
      if (ImGui::Selectable(App::strategyName(i), a.switchStrategy == i) && a.switchStrategy != i) {
        a.switchStrategy = i;
        settingsSet("switch_strategy", std::to_string(i));
        a.lastOpenTime = -1e9;  // apply right away
      }
    }
    ImGui::EndCombo();
  }
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("finished tasks: go through the finished-tasks cache, newest first\n"
                      "running tasks (max %% done): the running task with the most work done\n"
                      "running tasks (min %% done): the running task with the least work done");
  ImGui::TextUnformatted("auto switch between tasks interval");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  ImGui::DragInt("##switchint", &a.switchInterval, 0.5f, 30, 3600, "%d s", ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::IsItemDeactivatedAfterEdit()) settingsSet("task_switch_interval", std::to_string(a.switchInterval));
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("Switch when the shown task has made no progress (no new rows) for this long.\n"
                      "A task that keeps writing rows stays. 30-3600 s, default 60.");
  ImGui::EndDisabled();
  ImGui::SameLine();
  char nextLab[48];
  const char* hold = a.switchHold();
  if (!a.autoSwitch) std::snprintf(nextLab, sizeof nextLab, "next###nexttask");
  else if (*hold && a.idleFor() >= a.switchInterval) std::snprintf(nextLab, sizeof nextLab, "next (%s)###nexttask", hold);
  else std::snprintf(nextLab, sizeof nextLab, "next (%ds)###nexttask", (int)std::ceil(std::max(0.0, a.switchInterval - a.idleFor())));
  if (ImGui::Button(nextLab)) a.autoSwitchTask(rend, true);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Show the next task now (by the strategy).\nThe countdown restarts whenever the shown task writes new rows.");
  if (!ImGui::CollapsingHeader(hdr)) return;  // collapsed by default
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 140);
  bool enter = ImGui::InputTextWithHint("##boincdir", "BOINC data folder", a.boincBuf, sizeof a.boincBuf,
                                        ImGuiInputTextFlags_EnterReturnsTrue);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("BOINC data folder (contains slots\\ and projects\\); saved for next time");
  ImGui::SameLine();
  if (ImGui::Button("Apply") || enter) a.setBoincDir(a.boincBuf, true);
  ImGui::SameLine();
  if (ImGui::Button("Default")) a.setBoincDir(boincDefaultDataDir(), true);
  if (!a.boincErr.empty()) ImGui::TextColored(ImVec4(1, 0.6f, 0.35f, 1), "%s", a.boincErr.c_str());
  else if (!any) ImGui::TextDisabled("no period_search task running in %s\\slots", a.boincDir.c_str());
  if (!any) return;
  ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit;
  if (ImGui::BeginTable("tasks", 5, tf)) {
    ImGui::TableSetupColumn("slot");
    ImGui::TableSetupColumn("asteroid", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("class");
    ImGui::TableSetupColumn("progress");
    ImGui::TableSetupColumn("WU name");
    ImGui::TableHeadersRow();
    for (auto& t : a.tasks) {
      if (!t.periodSearch) continue;
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      bool sel = t.slotDir == a.current.slotDir;
      char lab[32];
      std::snprintf(lab, sizeof lab, "%d##slot%d", t.slot, t.slot);
      if (ImGui::Selectable(lab, sel, ImGuiSelectableFlags_SpanAllColumns) && !sel) a.openTask(t, rend);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("open this task\\n%s", t.slotDir.c_str());
      ImGui::TableNextColumn();
      auto it = a.taskInfo.find(t.designation);
      if (it != a.taskInfo.end() && it->second.ok) ImGui::TextUnformatted(it->second.fullname.c_str());
      else if (a.taskLookups.count(t.designation)) ImGui::TextDisabled("%s ...", t.designation.c_str());
      else ImGui::TextUnformatted(t.designation.empty() ? "?" : t.designation.c_str());
      ImGui::TableNextColumn();
      if (it != a.taskInfo.end() && it->second.ok) {
        ImGui::Text("%s (%s)", it->second.orbitClass.c_str(), it->second.orbitCode.c_str());
      }
      ImGui::TableNextColumn();
      auto sp = a.slotProgress.find(t.slotDir);
      if ((sp != a.slotProgress.end() && sp->second.wuName == t.wuName && sp->second.total > 0 && sp->second.rows >= 0) ||
          t.fractionDone >= 0)
        ImGui::Text("%5.1f%%", 100 * a.taskFraction(t));  // same value the sort and the auto switch use
      else
        ImGui::TextDisabled("-");
      ImGui::TableNextColumn();
      ImGui::TextDisabled("%s", t.wuName.c_str());
    }
    ImGui::EndTable();
  }
}

// finished tasks kept outside the BOINC folder
static void drawCache(App& a, MeshRenderer& rend) {
  char hdr[96];
  std::snprintf(hdr, sizeof hdr, "Finished tasks (%d cached)###cache", (int)a.cached.size());
  static double lastNames = -1e9;  // the names count changes as lookups finish
  if (!ImGui::CollapsingHeader(hdr)) return;  // collapsed by default
  if (glfwGetTime() - lastNames > 5) { lastNames = glfwGetTime(); a.namesCached = namesCacheCount(); }
  ImGui::TextUnformatted("keep");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  int n = a.cacheMax;
  ImGui::DragInt("##cachemax", &n, 1.0f, 0, 100000, n == 0 ? "off" : "%d tasks", ImGuiSliderFlags_AlwaysClamp);
  if (n != a.cacheMax) a.cacheMax = n;
  if (ImGui::IsItemDeactivatedAfterEdit()) a.setCacheMax(a.cacheMax);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Finished BOINC tasks kept in the cache (default 100); the oldest are deleted first.\n"
                      "0 = do not cache. Folder: %s", cacheDir().c_str());
  ImGui::SameLine();
  ImGui::BeginDisabled(a.cached.empty() && a.namesCached == 0);
  if (ImGui::Button("Clear cache")) ImGui::OpenPopup("Clear the cache?");
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("Deletes the cached finished tasks and the cached asteroid names\n(JPL SBDB answers, looked up again when needed)");
  ImGui::SameLine();
  ImGui::TextDisabled("%d asteroid names cached", a.namesCached);
  if (ImGui::BeginPopupModal("Clear the cache?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Delete %d cached finished tasks and %d cached asteroid names?", (int)a.cached.size(), a.namesCached);
    ImGui::TextDisabled("%s", cacheRootDir().c_str());
    if (ImGui::Button("Delete", ImVec2(100, 0))) {
      if (a.currentCached) a.clearAll(rend);  // its files are about to go
      a.clearCache();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(100, 0))) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
  if (a.cached.empty()) {
    ImGui::TextDisabled(a.cacheMax > 0 ? "no finished tasks yet - they are copied here when they complete"
                                       : "caching is off");
    return;
  }
  ImGuiTableFlags fl = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
  float h = std::min(10, (int)a.cached.size() + 1) * ImGui::GetTextLineHeightWithSpacing() + 6;
  if (ImGui::BeginTable("cachetbl", 6, fl, ImVec2(0, h))) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("finished");
    ImGui::TableSetupColumn("asteroid");
    ImGui::TableSetupColumn("class");
    ImGui::TableSetupColumn("shape P [h]");
    ImGui::TableSetupColumn("quality");
    ImGui::TableSetupColumn("WU", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < a.cached.size(); i++) {
      const CachedTask& c = a.cached[i];
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      char when[32] = "?";
      std::time_t ft = c.finished;
      if (std::tm* tm = std::localtime(&ft)) std::strftime(when, sizeof when, "%Y-%m-%d %H:%M", tm);
      char lab[64];
      std::snprintf(lab, sizeof lab, "%s##cache%d", when, (int)i);
      bool sel = a.currentCached && a.current.slotDir == c.dir;
      if (ImGui::Selectable(lab, sel, ImGuiSelectableFlags_SpanAllColumns) && !sel) {
        a.openCached(c, rend);
        ImGui::EndTable();
        return;  // `cached` entries stay valid, but stop drawing this frame's table
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("open this finished task\n%d of %d trial periods\n%s", c.rows, c.total, c.dir.c_str());
      ImGui::TableNextColumn();
      auto it = a.taskInfo.find(c.designation);
      if (it != a.taskInfo.end() && it->second.ok) ImGui::TextUnformatted(it->second.fullname.c_str());
      else ImGui::TextUnformatted(c.designation.empty() ? "?" : c.designation.c_str());
      ImGui::TableNextColumn();
      if (it != a.taskInfo.end() && it->second.ok) ImGui::Text("%s (%s)", it->second.orbitClass.c_str(), it->second.orbitCode.c_str());
      ImGui::TableNextColumn();
      if (c.shapePeriod > 0) ImGui::Text("%.5f", c.shapePeriod);
      else ImGui::TextDisabled("-");
      ImGui::TableNextColumn();
      if (c.quality.ok) {
        const RunQuality& q = c.quality;
        ImVec4 col = q.grade == 2 ? ImVec4(0.4f, 0.85f, 0.4f, 1) : q.grade == 1 ? ImVec4(0.95f, 0.8f, 0.3f, 1) : ImVec4(1, 0.45f, 0.4f, 1);
        ImGui::TextColored(col, "%s", q.label());
        if (ImGui::IsItemHovered()) {
          char dk[32];
          if (q.dark >= 0) std::snprintf(dk, sizeof dk, "%.2f %%", q.dark); else std::snprintf(dk, sizeof dk, "unknown");
          ImGui::SetTooltip("period: %s (%d interval%s below chi2_min x 1.1, %.0f %% of the scanned periods)\n"
                            "dark facet of the best period: %s (should be < 1 %%)\n"
                            "rms of the best period: %.6f%s\n\n"
                            "good: unique period (one narrow interval) and dark facet < 1 %%\n"
                            "fair: up to 3 competing periods, dark facet < 2 %%\n"
                            "poor: otherwise",
                            q.minima > 1 ? "ambiguous" : q.belowFrac > 0.5 ? "not constrained (flat periodogram)" : "unique",
                            q.minima, q.minima == 1 ? "" : "s", 100 * q.belowFrac, dk, q.rms,
                            c.shapeRms > 0 ? ("\nrms of the cached shape: " + std::to_string(c.shapeRms)).c_str() : "");
        }
      } else {
        ImGui::TextDisabled("?");
      }
      ImGui::TableNextColumn();
      ImGui::TextDisabled("%s", c.wuName.c_str());
    }
    ImGui::EndTable();
  }
}

// --------------------------------------------------------------------------
static void drawPanel(App& a, MeshRenderer& rend, float panelX, float panelW, float height) {
  ImGui::SetNextWindowPos(ImVec2(panelX, 0));
  ImGui::SetNextWindowSize(ImVec2(panelW, height));
  ImGui::Begin("Results", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

  // ---- object (JPL SBDB) -----------------------------------------------------
  if (a.sbdb.ok) {
    ImGui::TextColored(ImVec4(1, 0.85f, 0.45f, 1), "%s", a.sbdb.fullname.c_str());
    ImGui::SameLine();
    ImGui::Text("  %s (%s)%s%s", a.sbdb.orbitClass.c_str(), a.sbdb.orbitCode.c_str(), a.sbdb.neo ? "  NEO" : "",
                a.sbdb.pha ? "  PHA" : "");
    ImGui::TextDisabled("a = %s AU   e = %s   i = %s deg   (%s)", a.sbdb.a.c_str(), a.sbdb.e.c_str(), a.sbdb.i.c_str(),
                        a.sbdb.source.c_str());
  } else if (a.sbdbPending) {
    ImGui::TextDisabled("looking up %s in JPL SBDB...", a.sbdbFor.c_str());
  } else if (!a.sbdb.error.empty()) {
    ImGui::TextColored(ImVec4(1, 0.6f, 0.35f, 1), "%s: %s", a.sbdb.designation.c_str(), a.sbdb.error.c_str());
  }
  if (!a.current.wuName.empty()) {
    ImGui::Text("WU");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1, 1), "%s", a.current.wuName.c_str());
    ImGui::SameLine();
    if (a.currentCached) {
      char when[32] = "";
      if (std::tm* tm = std::localtime(&a.currentFinished)) std::strftime(when, sizeof when, "%Y-%m-%d %H:%M", tm);
      ImGui::TextDisabled("finished %s (cached)", when);
    } else {
      ImGui::TextDisabled("BOINC slot %d", a.current.slot);
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("result %s\n%s", a.current.resultName.c_str(), a.current.slotDir.c_str());
  }

  { PROF("ui boinc"); drawBoinc(a, rend); }
  { PROF("ui cache"); drawCache(a, rend); }

  // ---- files -------------------------------------------------------------
  if (ImGui::CollapsingHeader("Files", a.in.ok && a.mesh.ok ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
    auto row = [](const char* what, bool ok, const std::string& p) {
      ImGui::TextColored(ok ? ImVec4(0.5f, 0.9f, 0.5f, 1) : ImVec4(0.6f, 0.6f, 0.6f, 1), "%-8s %s", what,
                         ok ? u8p(p).filename().u8string().c_str() : "-");
      if (ok && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", p.c_str());
    };
    row("input", a.in.ok, a.in.path);
    row("output", !a.out.rows.empty(), a.out.paths.empty() ? "" : a.out.paths[0] + (a.out.paths.size() > 1 ? " (+" + std::to_string(a.out.paths.size() - 1) + ")" : ""));
    row("shape", a.mesh.ok, a.mesh.path);
    row("model", a.model.ok, a.model.path);
    row("gauss", a.gauss.ok, a.gauss.path);
    ImGui::SetNextItemWidth(120);
    bool go = ImGui::InputTextWithHint("##des", "designation", a.desBuf, sizeof a.desBuf, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    ImGui::BeginDisabled(a.sbdbPending);
    if ((ImGui::Button("Look up in JPL SBDB") || go) && a.desBuf[0]) a.lookupObject(a.desBuf, true);  // by hand: ask the API now
    ImGui::EndDisabled();
    if (a.sbdbPending) { ImGui::SameLine(); ImGui::TextDisabled("asking JPL..."); }
    ImGui::SetNextItemWidth(panelW - 140);
    bool enter = ImGui::InputTextWithHint("##path", "file or folder path (or drop files on the window)", a.pathBuf,
                                          sizeof a.pathBuf, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((ImGui::Button("Load") || enter) && a.pathBuf[0]) a.load({a.pathBuf}, rend);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) a.clearAll(rend);
    if (ImGui::TreeNode("Log")) {
      for (auto it = a.log.rbegin(); it != a.log.rend() && it - a.log.rbegin() < 30; ++it) ImGui::TextWrapped("%s", it->c_str());
      ImGui::TreePop();
    }
  }

  // ---- solution summary ---------------------------------------------------
  if (a.model.ok) {
    ImGui::SeparatorText("Shape model solution");
    ImGui::Text("P = %.6f h   lambda = %.1f deg   beta = %.1f deg", a.model.periodHours(), a.model.lambdaDeg(), a.model.betaDeg());
    if (a.fit.ok) ImGui::Text("rms of model vs data = %.6f  (%s)", a.fit.rms, a.facetsFromGauss ? "Gaussian image" : "OBJ facets");
  }
  { PROF("ui compute"); drawCompute(a); }
  if (a.out.files.empty() && a.in.ok) {
    ImGui::SeparatorText("period_search_out");
    auto ex = a.expectedOutputs();
    ImGui::TextDisabled("waiting for %s ...", ex.empty() ? "period_search_out" : u8p(ex[0]).filename().u8string().c_str());
    if (ImGui::IsItemHovered()) {
      std::string t = "Checked every second; loaded as soon as it appears:";
      for (auto& e : ex) t += "\n" + e;
      ImGui::SetTooltip("%s", t.c_str());
    }
  }
  if (!a.out.files.empty()) {
    ImGui::SeparatorText("period_search_out");
    { PROF("ui progress"); drawProgress(a); }
    if (!a.out.rows.empty()) {
      auto& b = a.out.rows[a.out.best];
      ImGui::Text("best so far: P = %.6f h  chi2 = %.4f  rms = %.6f  lambda/beta = %.0f/%.0f  dark = %.1f%%", b.period,
                  b.chi2, b.rms, b.lambda, b.beta, b.dark);
    }
    if (a.selectedRow >= 0) {
      auto& s = a.out.rows[a.selectedRow];
      ImGui::TextColored(ImVec4(1, 0.65f, 0.2f, 1), "selected: P = %.6f h  chi2 = %.4f  lambda/beta = %.0f/%.0f  dark = %.1f%%",
                         s.period, s.chi2, s.lambda, s.beta, s.dark);
    }
  }

  // ---- periodogram --------------------------------------------------------
  ProfScope profPg("ui periodogram+tabs");
  // collapsible (open by default); collapsed, the tabs below get the room
  bool pgOpen = !a.out.rows.empty() && ImGui::CollapsingHeader("Periodogram###pgram", ImGuiTreeNodeFlags_DefaultOpen);
  if (pgOpen && ImPlot::BeginPlot("##Periodogram", ImVec2(-1, 240))) {
    size_t n = a.out.rows.size();
    std::vector<double> p(n), c(n);
    const char* metricName[] = {"chi2", "rms", "dark facet [%]"};
    for (size_t i = 0; i < n; i++) {
      auto& r = a.out.rows[i];
      p[i] = r.period;
      c[i] = a.pgMetric == 0 ? r.chi2 : (a.pgMetric == 1 ? r.rms : r.dark);
    }
    ImPlotAxisFlags yf = a.followAxes ? ImPlotAxisFlags_AutoFit : 0;
    bool range = a.fullRange && a.in.totalTrials() > 0 && !a.out.files.empty() && a.out.files[0].rows <= a.in.totalTrials() &&
                 a.out.rows.front().period > a.in.perStart - 0.05 * (a.in.perEnd - a.in.perStart) &&
                 a.out.rows.back().period < a.in.perEnd + 0.05 * (a.in.perEnd - a.in.perStart);
    ImPlot::SetupAxes("period [h]", metricName[a.pgMetric], range ? 0 : (a.followAxes ? ImPlotAxisFlags_AutoFit : 0), yf);
    if (range) ImPlot::SetupAxisLimits(ImAxis_X1, a.in.perStart, a.in.perEnd, a.followAxes ? ImPlotCond_Always : ImPlotCond_Once);
    // long scans: draw the min/max per screen column instead of every row (same picture, far fewer vertices)
    int cols = std::max(50, (int)ImPlot::GetPlotSize().x);
    ImPlotRect lim = ImPlot::GetPlotLimits();
    ImPlot::SetNextLineStyle(ImVec4(0.55f, 0.6f, 0.7f, 1));
    if ((int)n > 3 * cols) {
      static std::vector<double> dx, dy;
      decimateMinMax(p, c, lim.X.Min, lim.X.Max, cols, dx, dy);
      ImPlot::PlotLine(metricName[a.pgMetric], dx.data(), dy.data(), (int)dx.size());
    } else {
      ImPlot::PlotLine(metricName[a.pgMetric], p.data(), c.data(), (int)n);
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f);
      ImPlot::PlotScatter("##pts", p.data(), c.data(), (int)n);
    }
    // significance threshold: chi2_min x 1.1 (same level in rms: rms_min x sqrt(1.1)); dark facet: 1 %
    double tv = a.pgMetric == 0 ? c[a.out.best] * 1.1 : (a.pgMetric == 1 ? c[a.out.best] * std::sqrt(1.1) : 1.0);
    double thr[2] = {tv, tv};
    double tx[2] = {p.front(), p.back()};
    ImPlot::SetNextLineStyle(ImVec4(0.9f, 0.3f, 0.3f, 0.6f));
    ImPlot::PlotLine(a.pgMetric == 2 ? "1 % limit" : "chi2_min x 1.1", tx, thr, 2);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 7, ImVec4(0.95f, 0.25f, 0.25f, 1));
    ImPlot::PlotScatter("best", &p[a.out.best], &c[a.out.best], 1);
    if (a.selectedRow >= 0) {
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 6, ImVec4(1, 0.65f, 0.2f, 1));
      ImPlot::PlotScatter("selected", &p[a.selectedRow], &c[a.selectedRow], 1);
    }
    if (a.model.ok) {
      double mp = a.model.periodHours();
      ImPlot::SetNextLineStyle(ImVec4(0.3f, 0.85f, 0.4f, 0.9f));
      ImPlot::PlotInfLines("shape model P", &mp, 1);
    }
    // part of the scan not computed yet (trial periods increase row by row)
    if (range && !a.out.files.empty() && a.out.files[0].rows < a.in.totalTrials()) {
      ImPlotRect lim = ImPlot::GetPlotLimits();
      double x0 = a.out.files[0].rows ? a.out.files[0].lastPeriod : a.in.perStart;
      double xs2[2] = {x0, a.in.perEnd}, lo[2] = {lim.Y.Min, lim.Y.Min}, hi[2] = {lim.Y.Max, lim.Y.Max};
      ImPlot::SetNextFillStyle(ImVec4(0.5f, 0.5f, 0.55f, 0.12f));
      ImPlot::PlotShaded("not computed yet", xs2, lo, hi, 2, ImPlotItemFlags_NoFit);
      double nextP = a.in.trialPeriod(a.out.files[0].rows);
      ImPlot::SetNextLineStyle(ImVec4(1, 0.8f, 0.3f, 0.8f));
      ImPlot::PlotInfLines("computing", &nextP, 1);
    }
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0)) {
      double x = ImPlot::GetPlotMousePos().x;
      int bestI = 0;
      for (size_t i = 1; i < n; i++)
        if (std::fabs(p[i] - x) < std::fabs(p[bestI] - x)) bestI = (int)i;
      a.selectedRow = bestI;
      a.targetPeriod = p[bestI];
    }
    ImPlot::EndPlot();
  }
  if (pgOpen) {
    ImGui::SetNextItemWidth(150);
    ImGui::Combo("periodogram y", &a.pgMetric, "chi2\0rms\0dark facet\0");
    ImGui::SameLine();
    ImGui::TextDisabled("click a point to select it");
  }

  // ---- tabs ---------------------------------------------------------------
  float plotH = std::max(200.0f, ImGui::GetContentRegionAvail().y - 30);
  if (ImGui::BeginTabBar("tabs")) {
    if (ImGui::BeginTabItem("Rotation", nullptr, tabFlag("Rotation"))) {
      PROF("tab rotation");
      if (a.hasRotation() && !a.facets.area.empty()) {
        // synthetic lightcurve over one rotation at the geometry of the selected observation
        const auto& o = a.in.obs[a.obsIdx];
        double ep = a.in.epoch(), phi0 = a.in.phi0deg * M_PI / 180, Pd = a.periodDays();
        double ph0 = rotPhase(a.model, ep, phi0, o.jd), sc = a.relScale();
        const int N = 200;
        std::vector<double> xs(N + 1), ys(N + 1);
        for (int k = 0; k <= N; k++) {
          double ph = (double)k / N;
          double jd = o.jd + (ph - ph0) * Pd;
          xs[k] = ph;
          ys[k] = brightness(a.facets, a.model, eclToBody(a.model, ep, phi0, jd), o.sun, o.earth) * sc;
        }
        std::vector<double> ox, oy;  // observations of the same night
        for (int j = 0; j < (int)a.in.obs.size(); j++)
          if (std::fabs(a.in.obs[j].jd - o.jd) < 0.5 && a.in.obs[j].lc == o.lc) {
            ox.push_back(rotPhase(a.model, ep, phi0, a.in.obs[j].jd));
            oy.push_back(a.in.obs[j].brightness / a.in.lcs[o.lc].mean);
          }
        double cph = rotPhase(a.model, ep, phi0, a.currentJd());
        double cy = brightness(a.facets, a.model, eclToBody(a.model, ep, phi0, a.currentJd()), o.sun, o.earth) * sc;
        if (ImPlot::BeginPlot("##rot", ImVec2(-1, plotH))) {
          ImPlot::SetupAxes("rotational phase", "relative brightness", 0, fitFlag(a));
          ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1, ImPlotCond_Always);
          ImPlot::PlotLine("model, this geometry", xs.data(), ys.data(), N + 1);
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4);
          ImPlot::PlotScatter("observed (same night)", ox.data(), oy.data(), (int)ox.size());
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 8, ImVec4(1, 0.65f, 0.2f, 1), 1, ImVec4(1, 0.65f, 0.2f, 1));
          ImPlot::PlotScatter("now", &cph, &cy, 1);
          ImPlot::EndPlot();
        }
      } else {
        ImGui::TextWrapped("Needs period_search_in, cg_dump.txt and a shape (OBJ or shape_dump.txt).");
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Lightcurve", nullptr, tabFlag("Lightcurve"))) {
      PROF("tab lightcurve");
      if (a.in.ok && ImPlot::BeginPlot("##lc", ImVec2(-1, plotH))) {
        size_t n = a.in.obs.size();
        std::vector<double> t(n), y(n);
        for (size_t i = 0; i < n; i++) {
          t[i] = a.in.obs[i].jd - 2400000;
          y[i] = a.in.obs[i].brightness / a.in.lcs[a.in.obs[i].lc].mean;
        }
        ImPlot::SetupAxes("date (UT)", "relative brightness", fitFlag(a), fitFlag(a));
        static double lcSpan = 0;  // visible span of the last frame, for the label format
        ImPlot::SetupAxisFormat(ImAxis_X1, jdAxisFormatter, &lcSpan);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f);
        ImPlot::PlotScatter("observed", t.data(), y.data(), (int)n);
        if (a.fit.ok) {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 3, ImVec4(0.3f, 0.85f, 0.4f, 1));
          ImPlot::PlotScatter("model", t.data(), a.fit.relModel.data(), (int)n);
        }
        double cur = a.currentJd() - 2400000;
        ImPlot::SetNextLineStyle(ImVec4(1, 0.65f, 0.2f, 1));
        ImPlot::PlotInfLines("selected", &cur, 1);
        if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0)) {
          ImPlotPoint m = ImPlot::GetPlotMousePos();
          int bi = 0;
          double bd = 1e300;
          ImPlotRect lim = ImPlot::GetPlotLimits();
          for (size_t i = 0; i < n; i++) {
            double dx = (t[i] - m.x) / lim.X.Size(), dy = (y[i] - m.y) / lim.Y.Size();
            if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; bi = (int)i; }
          }
          a.obsIdx = bi;
          a.timeOffset = 0;
        }
        if (ImPlot::IsPlotHovered()) {  // nearest observation: date, brightness, lightcurve
          ImPlotPoint m = ImPlot::GetPlotMousePos();
          ImPlotRect lim = ImPlot::GetPlotLimits();
          int bi = -1;
          double bd = 1e300;
          for (size_t i = 0; i < n; i++) {
            double dx = (t[i] - m.x) / lim.X.Size(), dy = (y[i] - m.y) / lim.Y.Size();
            if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; bi = (int)i; }
          }
          if (bi >= 0 && bd < 0.0004) {
            const auto& ob = a.in.obs[bi];
            ImGui::SetTooltip("%s\nrelative brightness %.3f\nlightcurve %d: %s\n(click to show this observation)", jdDate(ob.jd).c_str(), y[bi],
                              ob.lc + 1, a.in.lcs[ob.lc].label.c_str());
          }
        }
        lcSpan = ImPlot::GetPlotLimits().X.Size();
        ImPlot::EndPlot();
        ImGui::TextDisabled("observations %s - %s, %d lightcurves, %d points", jdDate(a.in.jdMin, false).c_str(),
                            jdDate(a.in.jdMax, false).c_str(), (int)a.in.lcs.size(), (int)n);
      } else if (!a.in.ok) ImGui::Text("No period_search_in loaded.");
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Folded", nullptr, tabFlag("Folded"))) {
      PROF("tab folded");
      if (a.fit.ok && ImPlot::BeginPlot("##fold", ImVec2(-1, plotH))) {
        size_t n = a.in.obs.size();
        std::vector<double> ph(n);
        double ep = a.in.epoch(), phi0 = a.in.phi0deg * M_PI / 180;
        for (size_t i = 0; i < n; i++) ph[i] = rotPhase(a.model, ep, phi0, a.in.obs[i].jd);
        ImPlot::SetupAxes("rotational phase", "relative brightness", 0, fitFlag(a));
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1, ImPlotCond_Always);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f);
        ImPlot::PlotScatter("observed", ph.data(), a.fit.relObs.data(), (int)n);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 3, ImVec4(0.3f, 0.85f, 0.4f, 1));
        ImPlot::PlotScatter("model", ph.data(), a.fit.relModel.data(), (int)n);
        ImPlot::EndPlot();
      } else if (!a.fit.ok) ImGui::TextWrapped("Needs period_search_in, cg_dump.txt and a shape.");
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Residuals", nullptr, tabFlag("Residuals"))) {
      PROF("tab residuals");
      if (a.fit.ok && ImPlot::BeginPlot("##res", ImVec2(-1, plotH))) {
        size_t n = a.in.obs.size();
        std::vector<double> x(n), r(n);
        for (size_t i = 0; i < n; i++) { x[i] = a.fit.relModel[i]; r[i] = a.fit.relObs[i] - a.fit.relModel[i]; }
        ImPlot::SetupAxes("model (relative)", "observed - model", fitFlag(a), fitFlag(a));
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f);
        ImPlot::PlotScatter("residual", x.data(), r.data(), (int)n);
        ImPlot::EndPlot();
      } else if (!a.fit.ok) ImGui::TextWrapped("Needs period_search_in, cg_dump.txt and a shape.");
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Pole", nullptr, tabFlag("Pole"))) {
      PROF("tab pole");
      if (!a.out.rows.empty() && ImPlot::BeginPlot("##pole", ImVec2(-1, plotH))) {
        size_t n = a.out.rows.size();
        std::vector<double> l(n), b(n);
        for (size_t i = 0; i < n; i++) { l[i] = a.out.rows[i].lambda; b[i] = a.out.rows[i].beta; }
        ImPlot::SetupAxes("lambda [deg]", "beta [deg]", fitFlag(a), fitFlag(a));
        if (!a.followAxes) {
          ImPlot::SetupAxisLimits(ImAxis_X1, 0, 360, ImPlotCond_Once);
          ImPlot::SetupAxisLimits(ImAxis_Y1, -90, 90, ImPlotCond_Once);
        }
        if (n > 3000) {  // many rows: one marker per 3x3 screen pixels is enough
          ImPlotRect lim = ImPlot::GetPlotLimits();
          ImVec2 sz = ImPlot::GetPlotSize();
          double cx = std::max(1e-9, (lim.X.Max - lim.X.Min) / std::max(1.0f, sz.x) * 3), cy = std::max(1e-9, (lim.Y.Max - lim.Y.Min) / std::max(1.0f, sz.y) * 3);
          std::vector<long long> seen;
          seen.reserve(n);
          std::vector<double> tl, tb;
          for (size_t i = 0; i < n; i++) {
            long long key = (long long)std::floor((l[i] - lim.X.Min) / cx) * 1000003LL + (long long)std::floor((b[i] - lim.Y.Min) / cy);
            seen.push_back(key);
          }
          std::vector<size_t> idx(n);
          for (size_t i = 0; i < n; i++) idx[i] = i;
          std::sort(idx.begin(), idx.end(), [&](size_t p1, size_t p2) { return seen[p1] < seen[p2]; });
          for (size_t k = 0; k < n; k++)
            if (k == 0 || seen[idx[k]] != seen[idx[k - 1]]) { tl.push_back(l[idx[k]]); tb.push_back(b[idx[k]]); }
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0.55f, 0.6f, 0.7f, 0.8f));
          ImPlot::PlotScatter("all periods", tl.data(), tb.data(), (int)tl.size());
        } else {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0.55f, 0.6f, 0.7f, 0.8f));
          ImPlot::PlotScatter("all periods", l.data(), b.data(), (int)n);
        }
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 8, ImVec4(0.95f, 0.25f, 0.25f, 1));
        ImPlot::PlotScatter("best", &l[a.out.best], &b[a.out.best], 1);
        if (a.selectedRow >= 0) {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 7, ImVec4(1, 0.65f, 0.2f, 1));
          ImPlot::PlotScatter("selected", &l[a.selectedRow], &b[a.selectedRow], 1);
        }
        if (a.model.ok) {
          double ml = a.model.lambdaDeg(), mb = a.model.betaDeg();
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Asterisk, 10, ImVec4(0.3f, 0.85f, 0.4f, 1), 2);
          ImPlot::PlotScatter("shape model", &ml, &mb, 1);
        }
        ImPlot::EndPlot();
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Orbit", nullptr, tabFlag("Orbit"))) {
      PROF("tab orbit");
      static int view = 0;  // 0 top (ecliptic x-y), 1 side (x-z)
      ImGui::RadioButton("top view (ecliptic)", &view, 0);
      ImGui::SameLine();
      ImGui::RadioButton("side view", &view, 1);
      const SbdbInfo& sb = a.sbdb;
      Kepler ak;
      const char* asrc = "";
      bool hasK = asteroidKepler(a, ak, &asrc);
      if (hasK)
        ImGui::TextDisabled("a %.3f AU  e %.3f  i %.2f deg  node %.1f deg  peri %.1f deg  period %.2f yr  (%s)", ak.a, ak.e, ak.i,
                            ak.om, ak.w, 360.0 / ak.n / 365.25, asrc);
      else
        ImGui::TextDisabled("orbit elements: %s", a.sbdbPending ? "looking up..." : "not available (JPL SBDB)");
      if (a.in.ok) {
        ImGui::SameLine();
        ImGui::TextDisabled("   positions on %s", jdDate(a.currentJd()).c_str());
      }
      if ((hasK || a.in.ok) && ImPlot::BeginPlot("##orbit", ImVec2(-1, plotH - 2 * ImGui::GetFrameHeightWithSpacing()), ImPlotFlags_Equal)) {
        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
        ImPlot::SetupAxes("x [AU]", view == 0 ? "y [AU]" : "z [AU]", fitFlag(a), fitFlag(a));
        double jd = a.in.ok ? a.currentJd() : 2451545.0 + (std::time(nullptr) / 86400.0 - 10957.5);
        static const char* names[] = {"Mercury", "Venus", "Earth", "Mars", "Jupiter"};
        static const ImVec4 cols[] = {{0.6f, 0.6f, 0.6f, 0.6f}, {0.85f, 0.75f, 0.5f, 0.6f}, {0.3f, 0.6f, 1, 0.9f},
                                      {0.9f, 0.4f, 0.3f, 0.7f}, {0.8f, 0.65f, 0.45f, 0.7f}};
        for (int p = 0; p < 5; p++) {
          Kepler k = planetElements(p);
          plotOrbitLine(names[p], k, view, cols[p], p == 2 ? 1.5f : 1.0f);
          V3 q = keplerAt(k, jd);
          double qx = q.x, qy = view == 0 ? q.y : q.z;
          orbitPoints(names[p], &qx, &qy, 1, ImPlotMarker_Circle, p == 2 ? 5 : 4, cols[p]);
        }
        double zero = 0;
        orbitPoints("##Sun", &zero, &zero, 1, ImPlotMarker_Circle, 7, ImVec4(1, 0.85f, 0.2f, 1));
        ImVec4 ac(0.95f, 0.95f, 0.95f, 1);
        if (hasK) {
          Kepler k = ak;
          plotOrbitLine("asteroid orbit", k, view, ac, 2.0f);
        }
        if (a.in.ok) {
          // positions at the observations, straight from period_search_in (asteroid-centric vectors)
          std::vector<double> ox, oy;
          size_t step = std::max<size_t>(1, a.in.obs.size() / 3000);
          for (size_t k = 0; k < a.in.obs.size(); k += step) {
            const Observation& o = a.in.obs[k];
            ox.push_back(-o.sun.x);
            oy.push_back(view == 0 ? -o.sun.y : -o.sun.z);
          }
          orbitPoints("asteroid (lightcurves)", ox.data(), oy.data(), (int)ox.size(), ImPlotMarker_Circle, 2.5f, ImVec4(0.6f, 0.9f, 0.6f, 0.8f));
          const Observation& o = a.in.obs[a.obsIdx];
          V3 ast = V3{} - o.sun, earth = o.earth - o.sun;
          double lx[2] = {earth.x, ast.x}, ly[2] = {view == 0 ? earth.y : earth.z, view == 0 ? ast.y : ast.z};
          ImPlot::SetNextLineStyle(ImVec4(0.3f, 0.6f, 1, 0.6f), 1);
          ImPlot::PlotLine("##line of sight", lx, ly, 2);
          orbitPoints("Earth (observation)", &lx[0], &ly[0], 1, ImPlotMarker_Circle, 5, ImVec4(0.3f, 0.6f, 1, 1));
          orbitPoints("asteroid (observation)", &lx[1], &ly[1], 1, ImPlotMarker_Diamond, 8, ImVec4(0.95f, 0.25f, 0.25f, 1));
        } else if (hasK) {
          Kepler k = ak;
          V3 q = keplerAt(k, jd);
          double qx = q.x, qy = view == 0 ? q.y : q.z;
          orbitPoints("asteroid (today)", &qx, &qy, 1, ImPlotMarker_Diamond, 8, ImVec4(0.95f, 0.25f, 0.25f, 1));
        }
        ImPlot::EndPlot();
      }
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
    gStartTab.clear();
  }
  ImGui::End();
}

static void drawControls(App& a) {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.72f);
  ImGui::Begin("View", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
  if (!a.mesh.ok) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "No shape loaded - drop an .obj or a shape folder here");
  ImGui::RadioButton("Free orbit", &a.viewMode, 0);
  ImGui::SameLine();
  ImGui::BeginDisabled(!a.hasRotation());
  ImGui::RadioButton("Seen from Earth", &a.viewMode, 1);
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!a.in.ok && !a.sbdb.hasOrbit);
  ImGui::RadioButton("Solar system", &a.viewMode, 2);
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("The asteroid's orbit with the Sun and the planets (Mercury to Jupiter), the shape model drawn\n"
                      "at the asteroid's position at the shown observation (enlarged). Drag to rotate, wheel to zoom.");
  ImGui::RadioButton("Studio", &a.shading, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Photometric", &a.shading, 1);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Lambert + Lommel-Seeliger, the scattering law used by period_search");
  if (ImGui::Checkbox("Smooth model", &a.smooth)) a.smoothToggled = true;
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Fitted shape function evaluated on a finer grid (800 facets) and smooth shading.\n"
                      "Off: the polyhedron exactly as period_search fits it (288 facets).");
  if (a.smooth && a.smoothPending) { ImGui::SameLine(); ImGui::TextDisabled("(building...)"); }
  ImGui::SameLine();
  ImGui::Checkbox("Facet edges", &a.edges);
  ImGui::SameLine();
  ImGui::Checkbox("Axes", &a.showAxes);
  if (a.viewMode == 0 && a.hasRotation()) {
    ImGui::SameLine();
    ImGui::Checkbox("Sun light", &a.sunLight);
  }
  if (a.shading == 1) {
    ImGui::SetNextItemWidth(160);
    ImGui::SliderFloat("exposure", &a.exposure, 0.3f, 5.0f);
  }
  ImGui::Separator();
  if (ImGui::Button(a.playing ? "Pause" : "Play")) a.playing = !a.playing;
  ImGui::SameLine();
  ImGui::SetNextItemWidth(160);
  ImGui::SliderFloat("rot/s", &a.speed, 0.01f, 1.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  if (a.in.ok) {
    int n = (int)a.in.obs.size();
    ImGui::SetNextItemWidth(260);
    if (ImGui::SliderInt("observation", &a.obsIdx, 0, n - 1)) a.timeOffset = 0;
    ImGui::SameLine();
    if (ImGui::ArrowButton("##prev", ImGuiDir_Left) && a.obsIdx > 0) { a.obsIdx--; a.timeOffset = 0; }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##next", ImGuiDir_Right) && a.obsIdx < n - 1) { a.obsIdx++; a.timeOffset = 0; }
    if (ImGui::Button(a.playObs ? "Stop###playobs" : "Play observations###playobs")) {
      a.playObs = !a.playObs;
      settingsSet("play_obs", a.playObs ? "1" : "0");
      if (a.playObs) {
        a.playing = false; a.obsAccum = 0; a.playJd = a.currentJd();
      }
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Go through the observations in time order. In the Solar system view with a known orbit\n"
                        "time runs continuously (days/s): the asteroid and the planets move along their orbits, the asteroid\n"
                        "turned as the rotation model says, and each observation is shown as it is passed.\n"
                        "In Free orbit and Seen from Earth (and without an orbit) it steps from observation to observation\n"
                        "(obs/s): the body as it was turned and lit at each measurement. Loops at the end.");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(130);
    Kepler kk;
    if (a.viewMode == 2 && asteroidKepler(a, kk))  // solar system: time-lapse along the orbit
      ImGui::SliderFloat("days/s", &a.daysPerSec, 1.0f, 365.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    else
      ImGui::SliderFloat("obs/s", &a.obsRate, 1.0f, 500.0f, "%.0f", ImGuiSliderFlags_Logarithmic);

    Observation o = a.in.obs[a.obsIdx];
    if (!atObservation(a)) {  // between observations (time-lapse): geometry from the orbits
      bool okp;
      V3 ast = asteroidHelioPos(a, a.currentJd(), okp);
      o.sun = V3{} - ast;
      o.earth = earthHelioPos(a, a.currentJd()) - ast;
    }
    double alpha = std::acos(std::fmax(-1.0, std::fmin(1.0, dot(normalize(o.sun), normalize(o.earth))))) * 180 / M_PI;
    ImGui::Text("%s   (JD %.5f, +%.3f h)", jdDate(a.currentJd()).c_str(), a.currentJd(), a.timeOffset * 24);
    if (o.lc >= 0 && o.lc < (int)a.in.lcs.size() && ImGui::IsItemHovered())
      ImGui::SetTooltip("lightcurve %d of %d: %s", o.lc + 1, (int)a.in.lcs.size(), a.in.lcs[o.lc].label.c_str());
    ImGui::Text("phase angle %.1f deg", alpha);
    ImGui::Text("r_sun %.3f AU   r_earth %.3f AU", norm(o.sun), norm(o.earth));
    if (a.hasRotation()) {
      double ph = rotPhase(a.model, a.in.epoch(), a.in.phi0deg * M_PI / 180, a.currentJd());
      ImGui::Text("rotational phase %.3f   P = %.4f h", ph, a.model.periodHours());
      float off = (float)(a.timeOffset / a.periodDays());
      ImGui::SetNextItemWidth(260);
      if (ImGui::SliderFloat("time [rotations]", &off, -1.0f, 1.0f)) a.timeOffset = off * a.periodDays();
    }
  }
  ImGui::TextDisabled("drag: rotate   wheel: zoom");
  ImGui::End();
}

// --------------------------------------------------------------------------
static int runApp(int argc, char** argv);

int main(int argc, char** argv) {
  std::set_terminate(onTerminate);  // first: report anything that goes wrong from here on
#ifdef _WIN32
  SetUnhandledExceptionFilter(onSehCrash);
#endif
  std::fprintf(stderr, "asteroid_viewer: starting\n");
  std::fflush(stderr);
  // 1st choice: a log next to the executable (no std::filesystem involved), else %APPDATA%\asteroid_viewer
  {
    std::string exe = argv[0];
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) exe = u8narrow(std::wstring(buf, n));
#endif
    size_t k = exe.find_last_of("/\\");
    std::string exeLog = (k == std::string::npos ? std::string() : exe.substr(0, k + 1)) + "asteroid_viewer.log";
    gLog = fopenUtf8(exeLog, "w");
    std::fprintf(stderr, "asteroid_viewer: log %s %s\n", exeLog.c_str(), gLog ? "" : "(cannot write)");
  }
  logLine("main() entered");
  if (!gLog) {
    try {
      std::error_code ec;
      fs::create_directories(u8p(logFilePath()).parent_path(), ec);
      gLog = fopenUtf8(logFilePath(), "w");
    } catch (...) {
    }
  }
  std::atexit(onExit);
  std::string cmd;
  for (int i = 0; i < argc; i++) cmd += std::string(i ? " " : "") + argv[i];
  logLine("command line: " + cmd);
  try {
    int r = runApp(argc, argv);
    gNormalExit = true;
    logLine("exit " + std::to_string(r));
    return r;
  } catch (std::exception& e) {
    fatal(std::string(e.what()) + " (during: " + gStep + ")");
  } catch (...) {
    fatal("unknown exception (during: " + gStep + ")");
  }
  gNormalExit = true;
  return 3;
}

static int runApp(int argc, char** argvAnsi) {
  // arguments as UTF-8 (the narrow argv uses the ANSI code page on Windows)
  std::vector<std::string> args = argsUtf8(argc, argvAnsi);
  std::vector<char*> argvU8;
  for (auto& a : args) argvU8.push_back(&a[0]);
  argvU8.push_back(nullptr);
  argc = (int)args.size();
  char** argv = argvU8.data();
  std::vector<std::string> paths;
  std::string screenshot;
  int frames = 3, winW = 1500, winH = 900, startObs = -1;
  int startView = -1, startShading = -1;
  bool startPlayObs = false;
  bool startEdges = false;
  double waitSec = 0, computeP = 0;
  bool waitCompute = false;
  int autoIntervalArg = 0;
  std::string boincArg;
  int slotArg = -1;
  bool noBoinc = false;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--screenshot" && i + 1 < argc) screenshot = argv[++i];
    else if (s == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
    else if (s == "--wait" && i + 1 < argc) waitSec = std::atof(argv[++i]);
    else if (s == "--compute" && i + 1 < argc) computeP = std::atof(argv[++i]);
    else if (s == "--wait-compute") waitCompute = true;
    else if (s == "--auto-interval" && i + 1 < argc) autoIntervalArg = std::atoi(argv[++i]);
    else if (s == "--boinc" && i + 1 < argc) boincArg = argv[++i];
    else if (s == "--slot" && i + 1 < argc) slotArg = std::atoi(argv[++i]);
    else if (s == "--no-boinc") noBoinc = true;
    else if (s == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &winW, &winH);
    else if (s == "--obs" && i + 1 < argc) startObs = std::atoi(argv[++i]);
    else if (s == "--view" && i + 1 < argc) { std::string v = argv[++i]; startView = v == "earth" ? 1 : v == "orbit" ? 2 : 0; }
    else if (s == "--photometric") startShading = 1;
    else if (s == "--edges") startEdges = true;
    else if (s == "--play-obs") startPlayObs = true;
    else if (s == "--tab" && i + 1 < argc) gStartTab = argv[++i];
    else if (s == "-h" || s == "--help") {
      std::printf("usage: asteroid_viewer [files or folders...]\n"
                  "  period_search_in, period_search_out*, *.obj, cg_dump.txt, shape_dump.txt\n"
                  "  options: --view free|earth|orbit --photometric --edges --obs N --size WxH --play-obs\n"
                  "           --tab Rotation|Lightcurve|Folded|Residuals|Pole|Orbit\n"
                  "           --compute P   (compute the shape for period P [h] at start)\n"
                  "           --auto-interval S   (seconds between automatic best-shape checks, default 10)\n"
                  "           --boinc DIR   (BOINC data folder, default C:\\ProgramData\\BOINC; also set in the app)\n"
                  "           --slot N   (open the task running in BOINC slot N)   --no-boinc   (skip BOINC)\n"
                  "           --screenshot out.ppm [--frames N] [--wait S] [--wait-compute]   (render and exit)\n");
      return 0;
    } else paths.push_back(s);
  }
  int pathsGiven = (int)paths.size();
  if (paths.empty()) paths.push_back(".");

  gStep = "initialising GLFW";
  logLine(gStep);
  glfwSetErrorCallback([](int code, const char* d) { logLine("GLFW error " + std::to_string(code) + ": " + d); });
  if (!glfwInit()) { fatal("glfwInit failed"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  gStep = "creating the OpenGL 3.3 window";
  logLine(gStep);
  GLFWwindow* win = glfwCreateWindow(winW, winH, "Asteroid Viewer - period_search", nullptr, nullptr);
  if (!win) { fatal("cannot create an OpenGL 3.3 window (graphics driver?)"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);
  if (const char* missing = psLoadGL(getProc)) { fatal(std::string("OpenGL function missing: ") + missing); return 1; }

  gStep = "initialising the user interface";
  logLine(gStep);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  float xs = 1, ys = 1;
  glfwGetWindowContentScale(win, &xs, &ys);
  ImFontConfig fc;
  fc.SizePixels = std::round(13.0f * std::max(1.0f, xs));
  io.Fonts->AddFontDefault(&fc);
  ImGui::StyleColorsDark();
  ImGui::GetStyle().ScaleAllSizes(std::max(1.0f, xs));
  ImGui_ImplGlfw_InitForOpenGL(win, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  MeshRenderer rend;
  std::string err;
  if (!rend.init(err)) { fatal("shader error: " + err); return 1; }

  App app;
  gApp = &app;
  glfwSetDropCallback(win, dropCallback);
  gStep = "reading settings";
  logLine("settings: " + logFilePath());
  if (!noBoinc) app.setBoincDir(!boincArg.empty() ? boincArg : settingsGet("boinc_dir", boincDefaultDataDir()), false);
  logLine("BOINC data folder: " + (noBoinc ? std::string("(disabled)") : app.boincDir));
  app.autoInterval = std::max(1, std::min(3600, std::atoi(settingsGet("auto_interval", "10").c_str())));
  gProf.always = !envUtf8("ASTEROID_VIEWER_PROFILE").empty();
  app.autoSwitch = settingsGet("auto_switch", "1") != "0";
  app.playObs = settingsGet("play_obs", "0") == "1";  // Play observations survives task switches and restarts
  app.cacheMax = std::max(0, std::min(100000, std::atoi(settingsGet("cache_max", "100").c_str())));
  app.refreshCache();
  app.switchInterval = std::max(30, std::min(3600, std::atoi(settingsGet("task_switch_interval", "60").c_str())));
  app.switchStrategy = std::max(0, std::min(2, std::atoi(settingsGet("switch_strategy", "1").c_str())));
  gStep = "loading files";
  app.load(paths, rend);
  gStep = "scanning BOINC slots";
  app.scanBoinc(glfwGetTime(), true);
  logLine("BOINC tasks found: " + std::to_string(app.tasks.size()) + (app.boincErr.empty() ? "" : " (" + app.boincErr + ")"));
  gStep = "opening a BOINC task";
  if (slotArg >= 0)
    for (auto& t : app.tasks)
      if (t.slot == slotArg) { app.openTask(t, rend); pathsGiven = 1; }
  if (!app.in.ok && pathsGiven == 0) {  // nothing to show: open the running task with the most work done
    const BoincTask* t = app.bestTask();
    if (!t)
      for (auto& x : app.tasks)
        if (x.periodSearch) { t = &x; break; }
    if (t) { BoincTask c = *t; app.openTask(c, rend); }
  }
  if (startObs >= 0 && app.in.ok) app.obsIdx = std::min(startObs, (int)app.in.obs.size() - 1);
  if (startView >= 0) app.viewMode = (startView == 1 && !app.hasRotation()) ? 0 : startView;
  if (startShading >= 0) app.shading = startShading;
  if (startEdges) app.edges = true;
  if (startPlayObs && app.in.ok) { app.playObs = true; app.playing = false; }
  if (!screenshot.empty()) app.playing = false;
  if (autoIntervalArg > 0) app.autoInterval = std::min(3600, autoIntervalArg);
  if (computeP > 0) { app.targetPeriod = computeP; app.startShape(); }

  gStep = "running";
  logLine("startup done");
  double last = glfwGetTime();
  int frame = 0, idleFrames = 0;
  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    if (!app.pendingDrops.empty()) {
      auto d = app.pendingDrops;
      app.pendingDrops.clear();
      app.load(d, rend);
    }
    double now = glfwGetTime(), dt = now - last;
    try {  // a failing update must not end the program: log it and carry on
      { PROF("poll"); app.poll(now); }
      { PROF("pollJob"); app.pollJob(rend); }
      app.pollOrbitFit();
      { PROF("pollLookup"); app.pollLookup(); }
      { PROF("pollSmooth"); app.pollSmooth(rend); }
      { PROF("scanBoinc"); app.scanBoinc(now); }
      { PROF("pollTaskLookups"); app.pollTaskLookups(); }
      { PROF("adoptCached"); app.adoptCached(rend); }
      { PROF("autoSwitchTask"); app.autoSwitchTask(rend); }
      { PROF("autoShape"); app.autoShape(now); }
    } catch (std::exception& e) {
      static std::string lastErr;
      if (lastErr != e.what()) { lastErr = e.what(); app.msg(std::string("ERROR (update): ") + e.what()); }
    }
    if (app.smoothToggled) {
      app.smoothToggled = false;
      if (app.smooth && !app.smoothMesh.ok) app.requestSmooth();
      app.refreshMesh(rend);
    }
    {
      static std::string lastTitle;
      double fr = overallFraction(app);
      std::string t = "Asteroid Viewer - " + (app.sbdb.ok ? app.sbdb.fullname : std::string("period_search"));
      if (fr >= 0) {
        char b[48];
        std::snprintf(b, sizeof b, fr >= 1 ? "  [finished]" : "  [%.1f%%]", 100 * fr);
        t += b;
      }
      if (t != lastTitle) { glfwSetWindowTitle(win, t.c_str()); lastTitle = t; }
    }
    last = now;
    Kepler loopK;
    if (app.playObs && app.in.ok && !app.in.obs.empty() && app.viewMode == 2 && asteroidKepler(app, loopK)) {  // time-lapse
      app.ensureObsOrder();
      app.playJd += dt * app.daysPerSec;
      if (app.playJd > app.in.jdMax || app.playJd < app.in.jdMin) app.playJd = app.in.jdMin;
      // the latest observation at or before the clock
      int lo = 0, hi = (int)app.obsOrder.size() - 1;
      while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (app.in.obs[app.obsOrder[mid]].jd <= app.playJd) lo = mid; else hi = mid - 1;
      }
      app.obsIdx = app.obsOrder[lo];
      app.timeOffset = app.playJd - app.in.obs[app.obsIdx].jd;
    } else if (app.playObs && app.in.ok && !app.in.obs.empty()) {  // observations in time order
      app.ensureObsOrder();
      app.obsAccum += dt * app.obsRate;
      int steps = (int)app.obsAccum;
      if (steps > 0) {
        app.obsAccum -= steps;
        int n = (int)app.obsOrder.size();
        int r = (app.obsRank[std::min(app.obsIdx, n - 1)] + steps) % n;
        app.obsIdx = app.obsOrder[r];
        app.timeOffset = 0;
      }
    } else if (app.playing) {
      if (app.hasRotation()) {
        app.timeOffset += dt * app.speed * app.periodDays();
        if (app.timeOffset > app.periodDays()) app.timeOffset -= app.periodDays();
      } else {
        app.spinAngle += dt * app.speed * 2 * M_PI;
      }
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int fbW, fbH, wW, wH;
    glfwGetFramebufferSize(win, &fbW, &fbH);
    glfwGetWindowSize(win, &wW, &wH);
    float panelW = std::min(std::max(420.0f, wW * 0.42f), (float)wW - 200);
    float viewW = wW - panelW;
    float sx = wW > 0 ? (float)fbW / wW : 1;

    // ---- camera input (3D area only) -------------------------------------
    if (!io.WantCaptureMouse && io.MousePos.x < viewW) {
      if (ImGui::IsMouseDragging(0) && (app.viewMode == 0 || app.viewMode == 2)) {
        app.yaw -= io.MouseDelta.x * 0.008;
        app.pitch = std::clamp(app.pitch + io.MouseDelta.y * 0.008, -1.55, 1.55);
      }
      if (io.MouseWheel != 0) {
        if (app.viewMode == 0) app.dist = std::clamp(app.dist * std::exp(-io.MouseWheel * 0.1), 1.2, 40.0);
        else if (app.viewMode == 2) app.orbitDist = std::clamp(app.orbitDist * std::exp(-io.MouseWheel * 0.1), 0.3, 80.0);
        else app.zoomEarth = std::clamp(app.zoomEarth * std::exp(io.MouseWheel * 0.1), 0.3, 5.0);
      }
    }

    // ---- 3D scene ------------------------------------------------------------
    double R = app.mesh.ok ? app.mesh.radius : 1.0;
    V3 eye, up{0, 0, 1}, lightDir, target{0, 0, 0};
    V3 sunW{1, 0, 0}, earthW{1, 0, 0};
    if (app.in.ok) {
      sunW = normalize(app.in.obs[app.obsIdx].sun);
      earthW = normalize(app.in.obs[app.obsIdx].earth);
      if (!atObservation(app)) {  // between observations: directions from the orbits
        bool okp;
        V3 ast = asteroidHelioPos(app, app.currentJd(), okp);
        if (okp) { sunW = normalize(V3{} - ast); earthW = normalize(earthHelioPos(app, app.currentJd()) - ast); }
      }
    }
    float fovy = 0.6f;
    double jdNow = app.in.ok ? app.currentJd() : 2451545.0 + (std::time(nullptr) / 86400.0 - 10957.5);
    bool solar = app.viewMode == 2 && (app.in.ok || app.sbdb.hasOrbit);
    V3 astPos{0, 0, 0};
    if (solar) {
      bool ok;
      astPos = asteroidHelioPos(app, jdNow, ok);
      if (app.orbitDist <= 0) {  // fit: the asteroid's aphelion (or its distance) and the Earth's orbit
        double r = app.sbdb.hasOrbit ? app.sbdb.ea * (1 + app.sbdb.ee) : norm(astPos);
        app.orbitDist = std::max(4.0, r * 3.2);
      }
    }
    if (solar) {
      eye = V3{std::cos(app.pitch) * std::cos(app.yaw), std::cos(app.pitch) * std::sin(app.yaw), std::sin(app.pitch)} * app.orbitDist;
      lightDir = normalize(V3{} - astPos);  // the body lit by the Sun at this moment
    } else if (app.viewMode == 1) {
      double d = R / std::tan(fovy / 2) * 1.8 / app.zoomEarth;
      eye = earthW * d;
      lightDir = sunW;
    } else {
      eye = V3{std::cos(app.pitch) * std::cos(app.yaw), std::cos(app.pitch) * std::sin(app.yaw), std::sin(app.pitch)} * (app.dist * R);
      if (app.sunLight && app.hasRotation()) lightDir = sunW;
      else lightDir = normalize(normalize(eye) + normalize(cross(up, eye)) * 0.6 + up * 0.5);
    }
    M4 view = lookAt(eye, target, up);
    M4 proj = solar ? perspective(fovy, viewW / std::max(1.0f, (float)wH), (float)(norm(eye) * 0.01), (float)(norm(eye) * 10))
                    : perspective(fovy, viewW / std::max(1.0f, (float)wH), (float)(norm(eye) * 0.05), (float)(norm(eye) * 4 + 10 * R));
    M4 vp = proj * view;
    M3 b2w = app.bodyToWorld();

    glViewport(0, 0, fbW, fbH);
    glClearColor(0.055f, 0.06f, 0.075f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, (int)(viewW * sx), fbH);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_MULTISAMPLE);
    if (app.mesh.ok && solar) {  // the shape at the asteroid's position, enlarged to a constant on-screen size
      M4 m = fromM3(b2w);
      double sc = 0.022 * norm(eye - astPos) / std::max(1e-9, R);
      for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++) m(r, c) = (float)(m(r, c) * sc);
      m(0, 3) = (float)astPos.x; m(1, 3) = (float)astPos.y; m(2, 3) = (float)astPos.z;
      rend.draw(m, vp, lightDir, normalize(eye - astPos), app.shading, (float)app.model.cl, (float)app.model.cls, app.exposure, app.edges);
    } else if (app.mesh.ok) {
      rend.draw(fromM3(b2w), vp, lightDir, normalize(eye), app.shading, (float)app.model.cl, (float)app.model.cls,
                app.exposure, app.edges);
    }
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    // ---- overlays: axes and directions ---------------------------------------
    if (solar) drawSolarSystem(app, Projector{vp, 0, 0, viewW, (float)wH}, jdNow);
    if (solar || (app.in.ok && app.playObs)) drawPlayInfo(app, Projector{vp, 0, 0, viewW, (float)wH}, jdNow);
    if (app.showAxes && app.mesh.ok && !solar) {
      Projector pr{vp, 0, 0, viewW, (float)wH};
      ImDrawList* dl = ImGui::GetBackgroundDrawList();
      V3 spin = b2w * V3{0, 0, 1};
      bool behind = dot(spin, normalize(eye)) < -0.3;  // arrow hidden behind the body: dim it
      arrow(dl, pr, spin * (R * 1.05), spin * (R * 1.7), behind ? IM_COL32(235, 80, 80, 90) : IM_COL32(235, 80, 80, 255), "spin");
      if (app.viewMode == 0 && app.hasRotation()) {
        arrow(dl, pr, sunW * (R * 1.3), sunW * (R * 2.0), IM_COL32(250, 210, 80, 255), "Sun");
        arrow(dl, pr, earthW * (R * 1.3), earthW * (R * 2.0), IM_COL32(90, 170, 255, 255), "Earth");
      }
    }

    if (!app.current.wuName.empty()) {
      float y = (float)wH - ImGui::GetFontSize() - 12 - (app.sbdb.ok ? ImGui::GetFontSize() * 2.6f + 14 : 0);
      ImGui::GetBackgroundDrawList()->AddText(ImVec2(15, y), IM_COL32(150, 190, 255, 255), ("WU " + app.current.wuName).c_str());
    }
    if (app.sbdb.ok) {
      ImDrawList* dl = ImGui::GetBackgroundDrawList();
      std::string l1 = app.sbdb.fullname, l2 = app.sbdb.orbitClass + " (" + app.sbdb.orbitCode + ")" +
                                               (app.sbdb.neo ? "  NEO" : "") + (app.sbdb.pha ? "  PHA" : "");
      float fs1 = ImGui::GetFontSize() * 1.6f;
      dl->AddText(nullptr, fs1, ImVec2(14, (float)wH - fs1 - ImGui::GetFontSize() - 18), IM_COL32(255, 220, 130, 255), l1.c_str());
      dl->AddText(ImVec2(15, (float)wH - ImGui::GetFontSize() - 12), IM_COL32(200, 200, 210, 255), l2.c_str());
    }
    { PROF("ui controls"); drawControls(app); }
    { PROF("ui panel"); drawPanel(app, rend, viewW, panelW, (float)wH); }

    double tRender = glfwGetTime();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    idleFrames = app.job.running() || app.smoothPending ? 0 : idleFrames + 1;
    if (startView == 1 && app.hasRotation()) { app.viewMode = 1; startView = -1; }
    if (!screenshot.empty() && ++frame >= frames && glfwGetTime() >= waitSec && !(waitCompute && idleFrames < 3)) {
      writePPM(screenshot.c_str(), fbW, fbH);
      std::printf("wrote %s\n", screenshot.c_str());
      break;
    }
    gProf.add("render", (glfwGetTime() - tRender) * 1000);
    glfwSwapBuffers(win);
    gProf.frame((glfwGetTime() - now) * 1000, glfwGetTime());
  }

  rend.destroy();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}
