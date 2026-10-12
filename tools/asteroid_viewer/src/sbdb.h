// Object name and orbit class from the JPL Small-Body Database API
// (https://ssd-api.jpl.nasa.gov/doc/sbdb.html).
#pragma once
#include <string>

struct SbdbInfo {
  bool ok = false;
  std::string designation;  // what was asked for
  std::string fullname;     // e.g. "20360 (1998 HV144)"
  std::string orbitClass, orbitCode;  // e.g. "Main-belt Asteroid", "MBA"
  std::string kind;
  bool neo = false, pha = false;
  std::string a, e, i, q, per;        // orbital elements as strings (AU, -, deg, AU, days)
  // heliocentric ecliptic J2000 elements for drawing the orbit (valid if hasOrbit)
  bool hasOrbit = false;
  double ea = 0, ee = 0, ei = 0, eom = 0, ew = 0, ema = 0, en = 0, epoch = 0;  // AU, -, deg, deg, deg, deg, deg/day, JD
  std::string error;
  std::string source;                 // "cache", "JPL SBDB"
  bool netError = false;              // no answer from the API (network, HTTP error): worth trying again later
};

// Asteroid designation from a file name, e.g.
//   period_search_in_20360 -> 20360,  ps_260930_input_110501_2 -> 110501 (BOINC work unit)
std::string designationFromFilename(const std::string& path);

// Blocking: reads <dir>/<des>_rdbd.json or the user cache if present, otherwise downloads
// and caches. Environment variable SBDB_API_URL overrides the endpoint (for tests).
// force: ask the API even if the last attempt for this designation failed a moment ago (lookup button)
SbdbInfo sbdbLookup(const std::string& designation, const std::string& dataDir, bool force = false);

// forget the network failures, so the next lookups ask the API again
void sbdbForgetFailures();

// Parses the JSON returned by the API.
SbdbInfo sbdbParse(const std::string& json);
