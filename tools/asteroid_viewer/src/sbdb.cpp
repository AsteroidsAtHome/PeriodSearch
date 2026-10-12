#include "u8.h"
#include "sbdb.h"
#include "cache.h"
#include <chrono>
#include <mutex>
#include <thread>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#ifndef WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY
#define WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY 4
#endif
#endif

namespace fs = std::filesystem;

// ---------------------------------------------------------------- tiny JSON
namespace {
struct J {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  bool b = false;
  std::string s;  // number text or string
  std::vector<J> a;
  std::map<std::string, J> o;
  const J& operator[](const std::string& k) const {
    static J none;
    auto it = o.find(k);
    return it == o.end() ? none : it->second;
  }
  std::string str() const { return t == Str || t == Num ? s : (t == Bool ? (b ? "true" : "false") : ""); }
};

struct Parser {
  const std::string& x;
  size_t p = 0;
  explicit Parser(const std::string& s) : x(s) {}
  void ws() { while (p < x.size() && std::isspace((unsigned char)x[p])) p++; }
  bool parse(J& v) {
    ws();
    if (p >= x.size()) return false;
    char c = x[p];
    if (c == '{') {
      v.t = J::Obj; p++; ws();
      if (p < x.size() && x[p] == '}') { p++; return true; }
      while (true) {
        J k;
        ws();
        if (!parseStr(k)) return false;
        ws();
        if (p >= x.size() || x[p] != ':') return false;
        p++;
        J val;
        if (!parse(val)) return false;
        v.o[k.s] = std::move(val);
        ws();
        if (p < x.size() && x[p] == ',') { p++; continue; }
        if (p < x.size() && x[p] == '}') { p++; return true; }
        return false;
      }
    }
    if (c == '[') {
      v.t = J::Arr; p++; ws();
      if (p < x.size() && x[p] == ']') { p++; return true; }
      while (true) {
        J val;
        if (!parse(val)) return false;
        v.a.push_back(std::move(val));
        ws();
        if (p < x.size() && x[p] == ',') { p++; continue; }
        if (p < x.size() && x[p] == ']') { p++; return true; }
        return false;
      }
    }
    if (c == '"') return parseStr(v);
    if (x.compare(p, 4, "true") == 0) { v.t = J::Bool; v.b = true; p += 4; return true; }
    if (x.compare(p, 5, "false") == 0) { v.t = J::Bool; v.b = false; p += 5; return true; }
    if (x.compare(p, 4, "null") == 0) { v.t = J::Null; p += 4; return true; }
    size_t s = p;
    while (p < x.size() && (std::isdigit((unsigned char)x[p]) || std::strchr("+-.eE", x[p]))) p++;
    if (p == s) return false;
    v.t = J::Num; v.s = x.substr(s, p - s);
    return true;
  }
  bool parseStr(J& v) {
    if (p >= x.size() || x[p] != '"') return false;
    p++;
    v.t = J::Str;
    while (p < x.size() && x[p] != '"') {
      if (x[p] == '\\' && p + 1 < x.size()) {
        char e = x[++p];
        if (e == 'n') v.s += '\n';
        else if (e == 't') v.s += '\t';
        else if (e == 'u' && p + 4 < x.size()) {
          unsigned cp = (unsigned)std::strtoul(x.substr(p + 1, 4).c_str(), nullptr, 16);
          p += 4;
          if (cp < 0x80) v.s += (char)cp;
          else if (cp < 0x800) { v.s += (char)(0xC0 | (cp >> 6)); v.s += (char)(0x80 | (cp & 0x3F)); }
          else { v.s += (char)(0xE0 | (cp >> 12)); v.s += (char)(0x80 | ((cp >> 6) & 0x3F)); v.s += (char)(0x80 | (cp & 0x3F)); }
        } else v.s += e;
        p++;
      } else v.s += x[p++];
    }
    if (p >= x.size()) return false;
    p++;
    return true;
  }
};
}  // namespace

SbdbInfo sbdbParse(const std::string& json) {
  SbdbInfo r;
  J root;
  Parser ps(json);
  if (!ps.parse(root) || root.t != J::Obj) {
    std::string snip = json.substr(0, 80);
    for (auto& ch : snip) if (ch == '\n' || ch == '\r') ch = ' ';
    r.error = "unexpected answer (not JSON): " + snip;
    return r;
  }
  if (root["object"].t != J::Obj) {
    // errors: {"code":"400","message":...} or {"code":"300","list":[...]} (several matches)
    std::string code = root["code"].str(), msg = root["message"].str();
    if (code == "300" && root["list"].t == J::Arr) {
      r.error = "several objects match:";
      for (size_t i = 0; i < root["list"].a.size() && i < 5; i++) r.error += " " + root["list"].a[i]["name"].str() + ";";
    } else r.error = msg.empty() ? "object not found in SBDB" : msg;
    return r;
  }
  const J& o = root["object"];
  r.fullname = o["fullname"].str();
  r.orbitClass = o["orbit_class"]["name"].str();
  r.orbitCode = o["orbit_class"]["code"].str();
  r.kind = o["kind"].str();
  r.neo = o["neo"].t == J::Bool && o["neo"].b;
  r.pha = o["pha"].t == J::Bool && o["pha"].b;
  for (auto& el : root["orbit"]["elements"].a) {
    std::string n = el["name"].str(), v = el["value"].str();
    if (n == "a") r.a = v;
    else if (n == "e") r.e = v;
    else if (n == "i") r.i = v;
    else if (n == "q") r.q = v;
    else if (n == "per") r.per = v;
    double d = std::atof(v.c_str());
    if (n == "a") r.ea = d;
    else if (n == "e") r.ee = d;
    else if (n == "i") r.ei = d;
    else if (n == "om") r.eom = d;
    else if (n == "w") r.ew = d;
    else if (n == "ma") r.ema = d;
    else if (n == "n") r.en = d;
  }
  r.epoch = std::atof(root["orbit"]["epoch"].str().c_str());
  if (r.en <= 0 && r.ea > 0) r.en = 0.9856076686 / std::pow(r.ea, 1.5);  // Kepler's 3rd law [deg/day]
  r.hasOrbit = r.ea > 0 && r.ee >= 0 && r.ee < 1 && r.epoch > 0;
  r.ok = !r.fullname.empty();
  if (!r.ok) r.error = "unexpected SBDB answer";
  return r;
}

std::string designationFromFilename(const std::string& path) {
  std::string name = u8p(path).filename().u8string();
  std::smatch m;
  static const std::regex boinc("ps_[0-9]+_input_([0-9]+)");  // BOINC WU: ps_<batch date>_input_<asteroid>_<chunk>
  static const std::regex suffix("(?:_in|_out|_input|_output)_([0-9]+)(?:\\D|$)");  // period_search_in_20360
  static const std::regex digits("([0-9]{3,})");
  if (std::regex_search(name, m, boinc)) return m[1];
  if (std::regex_search(name, m, suffix)) return m[1];
  std::string last;
  for (auto it = std::sregex_iterator(name.begin(), name.end(), digits); it != std::sregex_iterator(); ++it) last = (*it)[1];
  return last;
}

static std::string urlEncode(const std::string& s) {
  std::string r;
  char b[4];
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') r += (char)c;
    else { std::snprintf(b, sizeof b, "%%%02X", c); r += b; }
  }
  return r;
}

#ifdef _WIN32
static std::wstring widen(const std::string& s) {
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n ? n - 1 : 0, L'\0');
  if (n) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
  return w;
}

static bool httpGet(const std::string& url, std::string& body, std::string& err) {
  URL_COMPONENTS uc{};
  uc.dwStructSize = sizeof uc;
  wchar_t host[256], path[2048];
  uc.lpszHostName = host; uc.dwHostNameLength = 256;
  uc.lpszUrlPath = path; uc.dwUrlPathLength = 2048;
  wchar_t extra[2048];
  uc.lpszExtraInfo = extra; uc.dwExtraInfoLength = 2048;
  std::wstring wurl = widen(url);
  if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) { err = "bad URL"; return false; }
  std::wstring full = std::wstring(path, uc.dwUrlPathLength) + std::wstring(extra, uc.dwExtraInfoLength);
  HINTERNET s = WinHttpOpen(L"AsteroidViewer/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!s) s = WinHttpOpen(L"AsteroidViewer/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!s) { err = "WinHttpOpen failed"; return false; }
  WinHttpSetTimeouts(s, 10000, 10000, 15000, 30000);
  HINTERNET c = WinHttpConnect(s, std::wstring(host, uc.dwHostNameLength).c_str(), uc.nPort, 0);
  HINTERNET r = c ? WinHttpOpenRequest(c, L"GET", full.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                       uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0)
                  : nullptr;
  bool ok = r && WinHttpSendRequest(r, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
            WinHttpReceiveResponse(r, nullptr);
  if (ok) {
    DWORD status = 0, len = sizeof status;
    WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &len,
                        WINHTTP_NO_HEADER_INDEX);
    if (status >= 400 && status != 400) { err = "HTTP status " + std::to_string(status); ok = false; }
  }
  if (ok) {
    DWORD avail = 0;
    while (WinHttpQueryDataAvailable(r, &avail) && avail > 0) {
      std::string chunk(avail, '\0');
      DWORD got = 0;
      if (!WinHttpReadData(r, &chunk[0], avail, &got)) break;
      body.append(chunk.data(), got);
    }
  } else if (err.empty()) {
    err = "HTTP request failed (error " + std::to_string(GetLastError()) + ")";
  }
  if (r) WinHttpCloseHandle(r);
  if (c) WinHttpCloseHandle(c);
  WinHttpCloseHandle(s);
  return ok && !body.empty();
}
#else
static bool httpGet(const std::string& url, std::string& body, std::string& err) {
  std::string cmd = "curl -s -m 20 -w '\\n%{http_code}' \"" + url + "\"";  // status code on the last line
  std::unique_ptr<FILE, int (*)(FILE*)> p(popen(cmd.c_str(), "r"), pclose);
  if (!p) { err = "cannot run curl"; return false; }
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, p.get())) > 0) body.append(buf, n);
  size_t nl = body.rfind('\n');
  int status = nl == std::string::npos ? 0 : std::atoi(body.c_str() + nl + 1);
  if (nl != std::string::npos) body.resize(nl);
  if (status == 0) { err = "no answer (offline or blocked?)"; body.clear(); return false; }
  if (status >= 400 && status != 400) { err = "HTTP status " + std::to_string(status); body.clear(); return false; }
  if (body.empty()) err = "empty answer";
  return !body.empty();
}
#endif

static bool readFile(const fs::path& p, std::string& s) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  s.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return !s.empty();
}

static SbdbInfo sbdbLookupImpl(const std::string& des, const std::string& dataDir, bool force);

// never throws: runs in a background thread, an exception would end up in future::get()
SbdbInfo sbdbLookup(const std::string& des, const std::string& dataDir, bool force) {
  try {
    return sbdbLookupImpl(des, dataDir, force);
  } catch (std::exception& e) {
    SbdbInfo r;
    r.designation = des;
    r.error = std::string("lookup failed: ") + e.what();
    return r;
  } catch (...) {
    SbdbInfo r;
    r.designation = des;
    r.error = "lookup failed";
    return r;
  }
}

// network failures: an automatic retry of the same designation waits 2 minutes (keeps API traffic down when offline);
// the original error is kept so the message says what went wrong
struct Failure { std::chrono::steady_clock::time_point when; std::string error; };
static std::mutex gFailMutex;
static std::map<std::string, Failure> gFailed;
// one request at a time, at least 0.3 s apart: the JPL API refuses bursts of parallel requests
static std::mutex gNetMutex;
static std::chrono::steady_clock::time_point gLastRequest;

void sbdbForgetFailures() {
  std::lock_guard<std::mutex> lk(gFailMutex);
  gFailed.clear();
}

static SbdbInfo sbdbLookupImpl(const std::string& des, const std::string& dataDir, bool force) {
  SbdbInfo r;
  if (des.empty()) { r.error = "no designation"; return r; }
  std::string safe;
  for (char c : des) safe += std::isalnum((unsigned char)c) ? c : '_';
  std::error_code ec;
  // 1. the viewer's names cache (every answer of the API is kept, also "not found"), 2. graphic_app's file
  fs::path cacheFile = u8p(namesCacheDir()) / (safe + ".json");
  std::string json;
  if (readFile(cacheFile, json)) {
    r = sbdbParse(json);
    r.designation = des;
    r.source = "cache";
    return r;
  }
  if (!dataDir.empty() && readFile(u8p(dataDir) / (safe + "_rdbd.json"), json)) {
    r = sbdbParse(json);
    if (r.ok) {
      fs::create_directories(cacheFile.parent_path(), ec);
      std::ofstream(cacheFile, std::ios::binary) << json;
      r.designation = des; r.source = "cache"; return r;
    }
  }
  if (!force) {
    std::lock_guard<std::mutex> lk(gFailMutex);
    auto it = gFailed.find(des);
    auto age = std::chrono::steady_clock::now() - (it != gFailed.end() ? it->second.when : std::chrono::steady_clock::time_point());
    if (it != gFailed.end() && age < std::chrono::minutes(2)) {
      r.designation = des;
      r.netError = true;
      int left = (int)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::minutes(2) - age).count();
      r.error = it->second.error + " (next try in " + std::to_string(left) + " s)";
      return r;
    }
  }
  std::string env = envUtf8("SBDB_API_URL");
  std::string base = !env.empty() ? env : "https://ssd-api.jpl.nasa.gov/sbdb.api";
  std::string url = base + "?sstr=" + urlEncode(des), err;
  json.clear();
  bool got;
  {
    std::lock_guard<std::mutex> net(gNetMutex);
    auto wait = std::chrono::milliseconds(300) - (std::chrono::steady_clock::now() - gLastRequest);
    if (wait.count() > 0) std::this_thread::sleep_for(wait);
    got = httpGet(url, json, err);
    gLastRequest = std::chrono::steady_clock::now();
  }
  if (!got) {
    if (err.empty()) err = "empty answer from " + base;
    std::lock_guard<std::mutex> lk(gFailMutex);
    gFailed[des] = Failure{std::chrono::steady_clock::now(), err};
    r.designation = des; r.error = err; r.netError = true; return r;
  }
  {
    std::lock_guard<std::mutex> lk(gFailMutex);
    gFailed.erase(des);
  }
  r = sbdbParse(json);
  r.designation = des;
  r.source = "JPL SBDB";
  if (!r.ok && r.error.rfind("unexpected answer", 0) == 0) {  // e.g. an error page of a proxy: try again later
    r.netError = true;
    std::lock_guard<std::mutex> lk(gFailMutex);
    gFailed[des] = Failure{std::chrono::steady_clock::now(), r.error};
    return r;
  }
  if (r.ok || r.error.rfind("unexpected answer", 0) != 0) {  // a real answer: keep it, also "not found"
    fs::create_directories(cacheFile.parent_path(), ec);
    std::ofstream(cacheFile, std::ios::binary) << json;
  }
  return r;
}
