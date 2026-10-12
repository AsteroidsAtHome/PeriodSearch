// BOINC client integration: find the Asteroids@home period_search tasks running in the
// slots of a BOINC data folder (default C:\ProgramData\BOINC on Windows).
#pragma once
#include <string>
#include <vector>

struct BoincTask {
  int slot = -1;
  std::string slotDir;
  std::string wuName, resultName, appName, projectDir;
  std::string designation;   // asteroid number from the WU name: ps_261007_input_167069_29 -> 167069
  double fractionDone = -1;  // from boinc_task_state.xml (last checkpoint), -1 if unknown
  double cpuTime = -1;       // checkpoint CPU time [s]
  bool periodSearch = false; // has period_search_in in the slot
};

// BOINC data folder: Windows registry (BOINC Setup DATADIR) or C:\ProgramData\BOINC;
// /var/lib/boinc-client elsewhere
std::string boincDefaultDataDir();

// all slots that hold a task (empty slots are skipped); period_search tasks first
std::vector<BoincTask> boincScanSlots(const std::string& dataDir, std::string& err);

// reads <wu_name> etc. of one slot folder; false if it is not a BOINC slot
bool boincReadSlot(const std::string& slotDir, BoincTask& t);

// designation from a WU name: ps_<batch date>_input_<asteroid>_<chunk>, e.g. ps_261007_input_167069_29 -> 167069
std::string designationFromWuName(const std::string& wu);

// log file next to the settings (%APPDATA%\asteroid_viewer\asteroid_viewer.log)
std::string logFilePath();

// simple key=value settings in the user's config folder
std::string settingsGet(const std::string& key, const std::string& def = "");
void settingsSet(const std::string& key, const std::string& value);
