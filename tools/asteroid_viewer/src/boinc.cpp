#include "u8.h"
#include "boinc.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

static std::string readAll(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return "";
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static std::string xmlTag(const std::string& x, const std::string& tag) {
  std::string a = "<" + tag + ">", b = "</" + tag + ">";
  size_t i = x.find(a);
  if (i == std::string::npos) return "";
  i += a.size();
  size_t j = x.find(b, i);
  if (j == std::string::npos) return "";
  std::string v = x.substr(i, j - i);
  while (!v.empty() && std::isspace((unsigned char)v.back())) v.pop_back();
  while (!v.empty() && std::isspace((unsigned char)v.front())) v.erase(0, 1);
  return v;
}

// WU names: ps_<batch date YYMMDD>_input_<asteroid number>_<chunk>, e.g. ps_261007_input_165892_2
// -> 165892 (261007 is the batch, 2026-10-07, the same for many tasks)
std::string designationFromWuName(const std::string& wu) {
  // ps_<batch date>_input_<asteroid>_<chunk>, e.g. ps_261007_input_167069_29 -> 167069 (261007 = 2026-10-07)
  static const std::regex current("ps_[0-9]+_input_([0-9]+)(_[0-9]+)?");
  static const std::regex older("ps_([A-Za-z0-9]+)_input");  // fallback: ps_<asteroid>_input...
  std::smatch m;
  if (std::regex_search(wu, m, current)) return m[1];
  if (std::regex_search(wu, m, older)) return m[1];
  return "";
}

std::string boincDefaultDataDir() {
#ifdef _WIN32
  char buf[1024];
  DWORD size = sizeof buf;
  if (RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Space Sciences Laboratory, U.C. Berkeley\\BOINC Setup", "DATADIR",
                   RRF_RT_REG_SZ, nullptr, buf, &size) == ERROR_SUCCESS && buf[0]) {
    std::string d(buf);
    while (!d.empty() && (d.back() == '\\' || d.back() == '/')) d.pop_back();
    return d;
  }
  return "C:\\ProgramData\\BOINC";
#else
  std::error_code ec;
  if (fs::exists("/var/lib/boinc", ec)) return "/var/lib/boinc";
  if (fs::exists("/var/lib/boinc-client", ec)) return "/var/lib/boinc-client";  // older Debian/Ubuntu packages
  return "/var/lib/boinc";
#endif
}

bool boincReadSlot(const std::string& slotDir, BoincTask& t) {
  std::string init = readAll(u8p(slotDir) / "init_data.xml");
  if (init.empty()) return false;
  t = BoincTask();
  t.slotDir = slotDir;
  try { t.slot = std::stoi(u8p(slotDir).filename().u8string()); } catch (...) { t.slot = -1; }
  t.wuName = xmlTag(init, "wu_name");
  t.resultName = xmlTag(init, "result_name");
  t.appName = xmlTag(init, "app_name");
  t.projectDir = xmlTag(init, "project_dir");
  t.designation = designationFromWuName(t.wuName);
  std::error_code ec;
  t.periodSearch = fs::exists(u8p(slotDir) / "period_search_in", ec) || t.appName.find("period_search") != std::string::npos;
  std::string st = readAll(u8p(slotDir) / "boinc_task_state.xml");
  if (!st.empty()) {
    std::string f = xmlTag(st, "fraction_done"), c = xmlTag(st, "checkpoint_cpu_time");
    if (!f.empty()) t.fractionDone = std::atof(f.c_str());
    if (!c.empty()) t.cpuTime = std::atof(c.c_str());
  }
  return !t.wuName.empty();
}

std::vector<BoincTask> boincScanSlots(const std::string& dataDir, std::string& err) {
  std::vector<BoincTask> r;
  fs::path slots = u8p(dataDir) / "slots";
  std::error_code ec;
  if (!fs::is_directory(slots, ec)) {
    err = "no slots folder in " + dataDir;
    return r;
  }
  fs::directory_iterator it(slots, ec), end;
  if (ec) { err = "cannot list " + slots.u8string() + ": " + ec.message(); return r; }
  for (; it != end; it.increment(ec)) {  // increment(ec): never throws (e.g. access denied)
    if (ec) { err = ec.message(); break; }
    std::error_code ec2;
    if (!it->is_directory(ec2) || ec2) continue;
    try {
      BoincTask t;
      if (boincReadSlot(it->path().u8string(), t)) r.push_back(t);
    } catch (std::exception& e) {
      err = std::string("slot ") + it->path().filename().u8string() + ": " + e.what();
    }
  }
  std::sort(r.begin(), r.end(), [](const BoincTask& a, const BoincTask& b) {
    if (a.periodSearch != b.periodSearch) return a.periodSearch;
    return a.slot < b.slot;
  });
  return r;
}

// ---------------------------------------------------------------- settings
static fs::path settingsFile();
std::string logFilePath() { return (settingsFile().parent_path() / "asteroid_viewer.log").u8string(); }

static fs::path settingsFile() {
  std::string base;
#ifdef _WIN32
  base = envUtf8("APPDATA");  // UTF-8 (may contain non-ASCII letters of the user name)
#else
  base = envUtf8("XDG_CONFIG_HOME");
  if (base.empty()) {
    std::string home = envUtf8("HOME");
    if (!home.empty()) return u8p(home) / ".config" / "asteroid_viewer" / "settings.ini";
  }
#endif
  std::error_code ec;
  fs::path b = !base.empty() ? u8p(base) : fs::temp_directory_path(ec);
  return b / "asteroid_viewer" / "settings.ini";
}

static std::map<std::string, std::string> readSettings() {
  std::map<std::string, std::string> m;
  std::ifstream in(settingsFile());
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t k = line.find('=');
    if (k != std::string::npos && line[0] != '#') m[line.substr(0, k)] = line.substr(k + 1);
  }
  return m;
}

std::string settingsGet(const std::string& key, const std::string& def) {
  auto m = readSettings();
  auto it = m.find(key);
  return it == m.end() ? def : it->second;
}

void settingsSet(const std::string& key, const std::string& value) {
  auto m = readSettings();
  m[key] = value;
  std::error_code ec;
  fs::create_directories(settingsFile().parent_path(), ec);
  std::ofstream out(settingsFile());
  out << "# asteroid_viewer settings\n";
  for (auto& kv : m) out << kv.first << "=" << kv.second << "\n";
}
