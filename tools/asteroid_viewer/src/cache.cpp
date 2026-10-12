#include "cache.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "u8.h"
#include "boinc.h"

namespace fs = std::filesystem;

std::string cacheRootDir() {
  std::string o = envUtf8("ASTEROID_VIEWER_CACHE");
  if (!o.empty()) return o;
  std::string base;
#ifdef _WIN32
  base = envUtf8("LOCALAPPDATA");
  if (base.empty()) base = envUtf8("APPDATA");
#else
  base = envUtf8("XDG_CACHE_HOME");
  if (base.empty()) {
    std::string home = envUtf8("HOME");
    if (!home.empty()) base = (u8p(home) / ".cache").u8string();
  }
#endif
  std::error_code ec;
  fs::path b = !base.empty() ? u8p(base) : fs::temp_directory_path(ec);
  return (b / "asteroid_viewer").u8string();
}

std::string cacheDir() { return (u8p(cacheRootDir()) / "finished_tasks").u8string(); }
std::string namesCacheDir() { return (u8p(cacheRootDir()) / "asteroid_names").u8string(); }

int namesCacheCount() {
  int n = 0;
  std::error_code ec;
  fs::directory_iterator it(u8p(namesCacheDir()), ec), end;
  for (; !ec && it != end; it.increment(ec)) n += it->path().extension() == ".json";
  return n;
}

int namesCacheClear() {
  int n = namesCacheCount();
  std::error_code ec;
  fs::remove_all(u8p(namesCacheDir()), ec);
  return n;
}

// WU names are plain ASCII, but keep the folder name safe anyway
static std::string folderName(const std::string& wu) {
  std::string s;
  for (char c : wu) s += (std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.') ? c : '_';
  return s.empty() ? "unnamed" : s;
}

std::string cacheTaskDir(const std::string& wuName) { return (u8p(cacheDir()) / folderName(wuName)).u8string(); }

bool cacheHas(const std::string& wuName) {
  std::error_code ec;
  return fs::exists(u8p(cacheDir()) / folderName(wuName) / "task.txt", ec);
}

static std::string qualityLines(const RunQuality& q) {
  if (!q.ok) return "";
  char b[200];
  std::snprintf(b, sizeof b, "q_grade=%d\nq_minima=%d\nq_below=%.4f\nq_dark=%.4f\nq_rms=%.8f\n", q.grade, q.minima, q.belowFrac,
                q.dark, q.rms);
  return b;
}

bool cacheStore(const std::string& wuName, const std::string& resultName, const std::string& designation,
                const std::string& inPath, const std::string& outPath, int rows, int total, std::string& err) {
  std::error_code ec;
  fs::path root = u8p(cacheDir());
  fs::path dst = root / folderName(wuName);
  fs::path tmp = root / (folderName(wuName) + ".partial");
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp, ec);
  if (ec) { err = "cannot create " + u8s(tmp) + ": " + ec.message(); return false; }
  fs::copy_file(u8p(inPath), tmp / "period_search_in", fs::copy_options::overwrite_existing, ec);
  if (ec) { err = "cannot copy " + inPath + ": " + ec.message(); fs::remove_all(tmp, ec); return false; }
  fs::copy_file(u8p(outPath), tmp / "period_search_out", fs::copy_options::overwrite_existing, ec);
  if (ec) { err = "cannot copy " + outPath + ": " + ec.message(); fs::remove_all(tmp, ec); return false; }
  {
    std::ofstream f(tmp / "task.txt");
    f << "wu_name=" << wuName << "\nresult_name=" << resultName << "\ndesignation=" << designation
      << "\nfinished=" << (long long)std::time(nullptr) << "\nrows=" << rows << "\ntotal=" << total << "\n";
    OutputData od;
    std::string e2;
    if (readOutput(outPath, od, e2)) f << qualityLines(assessRun(od));
    if (!f) { err = "cannot write task.txt"; fs::remove_all(tmp, ec); return false; }
  }
  fs::remove_all(dst, ec);
  fs::rename(tmp, dst, ec);  // a half-written copy never looks like a cached task
  if (ec) { err = "cannot rename " + u8s(tmp) + ": " + ec.message(); fs::remove_all(tmp, ec); return false; }
  return true;
}

std::vector<CachedTask> cacheList() {
  std::vector<CachedTask> r;
  std::error_code ec;
  fs::directory_iterator it(u8p(cacheDir()), ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    std::error_code e2;
    if (!it->is_directory(e2)) continue;
    std::ifstream f(it->path() / "task.txt");
    if (!f) continue;
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      auto p = line.find('=');
      if (p != std::string::npos) kv[line.substr(0, p)] = line.substr(p + 1);
    }
    CachedTask c;
    c.dir = it->path().u8string();
    c.wuName = kv["wu_name"];
    c.resultName = kv["result_name"];
    c.designation = designationFromWuName(c.wuName.empty() ? it->path().filename().u8string() : kv["wu_name"]);
    if (c.designation.empty()) c.designation = kv["designation"];
    c.finished = (std::time_t)std::atoll(kv["finished"].c_str());
    c.rows = std::atoi(kv["rows"].c_str());
    c.total = std::atoi(kv["total"].c_str());
    {  // cached shape model solution (shape/solution.txt: period=...)
      std::ifstream sf(it->path() / "shape" / "solution.txt");
      std::string l;
      while (std::getline(sf, l))
        if (l.rfind("period=", 0) == 0) c.shapePeriod = std::atof(l.c_str() + 7);
        else if (l.rfind("rms=", 0) == 0) c.shapeRms = std::atof(l.c_str() + 4);
    }
    if (kv.count("q_grade")) {  // stored with the task: no need to read the whole output
      c.quality.ok = true;
      c.quality.grade = std::atoi(kv["q_grade"].c_str());
      c.quality.minima = std::atoi(kv["q_minima"].c_str());
      c.quality.belowFrac = std::atof(kv["q_below"].c_str());
      c.quality.dark = std::atof(kv["q_dark"].c_str());
      c.quality.rms = std::atof(kv["q_rms"].c_str());
    } else {  // older entry: assess once and remember it in task.txt
      OutputData od;
      std::string err;
      if (readOutput((it->path() / "period_search_out").u8string(), od, err)) {
        c.quality = assessRun(od);
        std::ofstream(it->path() / "task.txt", std::ios::app) << qualityLines(c.quality);
      }
    }
    if (c.wuName.empty()) c.wuName = it->path().filename().u8string();
    r.push_back(c);
  }
  std::sort(r.begin(), r.end(), [](const CachedTask& a, const CachedTask& b) {
    return a.finished != b.finished ? a.finished > b.finished : a.wuName < b.wuName;
  });
  return r;
}

int cacheTrim(int keep) {
  auto all = cacheList();
  int n = 0;
  for (size_t i = std::max(0, keep); i < all.size(); i++) {
    std::error_code ec;
    if (fs::remove_all(u8p(all[i].dir), ec) > 0 && !ec) n++;
  }
  return n;
}

int cacheClear() { return cacheTrim(0); }
