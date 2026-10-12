// Cache of finished BOINC tasks outside the BOINC data folder.
// BOINC deletes a task's files once the result is reported; the viewer keeps a copy of
// period_search_in / period_search_out of every finished task so it can be opened later.
//
// It also keeps the JPL SBDB answers (asteroid names), so each asteroid is looked up once.
//
//   root:    %LOCALAPPDATA%\asteroid_viewer (Windows), $XDG_CACHE_HOME (or ~/.cache)/asteroid_viewer
//            (ASTEROID_VIEWER_CACHE overrides it)
//   tasks:   <root>/finished_tasks/<WU name>/
//   names:   <root>/asteroid_names/<designation>.json
#pragma once
#include <ctime>
#include <string>
#include <vector>

#include "data.h"

struct CachedTask {
  std::string wuName, resultName, designation;
  std::string dir;        // folder holding period_search_in / period_search_out / task.txt
  std::time_t finished = 0;
  int rows = 0, total = 0;
  double shapePeriod = 0;  // > 0: the shape model solution is cached too (<dir>/shape/)
  double shapeRms = 0;     // rms of that shape's fit
  RunQuality quality;      // of the run (period_search_out)
};

std::string cacheRootDir();
std::string cacheDir();       // finished tasks
std::string namesCacheDir();  // JPL SBDB answers
int namesCacheCount();
int namesCacheClear();        // returns the number of names deleted

// copies the input and output of a finished task into the cache; false (and err) on failure
bool cacheStore(const std::string& wuName, const std::string& resultName, const std::string& designation,
                const std::string& inPath, const std::string& outPath, int rows, int total, std::string& err);

bool cacheHas(const std::string& wuName);

// folder of a cached task (whether or not it exists)
std::string cacheTaskDir(const std::string& wuName);

// all cached tasks, most recently finished first
std::vector<CachedTask> cacheList();

// keeps the `keep` most recently finished tasks, deletes the rest; returns the number deleted
int cacheTrim(int keep);

// deletes every cached task; returns the number deleted
int cacheClear();
