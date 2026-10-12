#include "u8.h"
#include "data.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

static bool nextLine(std::istream& in, std::string& line) {
  if (!std::getline(in, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return true;
}

bool readInput(const std::string& path, InputData& d, std::string& err) {
  std::ifstream in(u8p(resolveBoincLink(path)));
  if (!in) { err = "cannot open " + path; return false; }
  d = InputData();
  d.path = path;
  std::vector<std::string> L;
  std::string line;
  while (nextLine(in, line)) L.push_back(line);
  if (L.size() < 16) { err = "too short for period_search_in: " + path; return false; }
  try {
    std::istringstream(L[0]) >> d.perStart >> d.perStepCoef >> d.perEnd;
    std::istringstream(L[1]) >> d.jd00;
    std::istringstream(L[2]) >> d.phi0deg;
    std::istringstream(L[3]) >> d.conw;
    std::istringstream(L[4]) >> d.lmax >> d.mmax;
    std::istringstream(L[5]) >> d.nrows;
    // lines 6..13: scattering, LS coeff, stop condition, iterations, Alamda
    size_t i = 14;
    int nlc = std::stoi(L[i++]);
    d.jdMin = 1e300; d.jdMax = -1e300;
    for (int k = 0; k < nlc; k++) {
      if (i >= L.size()) throw std::runtime_error("unexpected end of file");
      std::istringstream h(L[i++]);
      Lightcurve lc;
      int flag = 0;
      h >> lc.count >> flag;
      lc.calibrated = flag != 0;
      std::getline(h, lc.label);
      lc.first = (int)d.obs.size();
      double sum = 0;
      for (int j = 0; j < lc.count; j++) {
        if (i >= L.size()) throw std::runtime_error("unexpected end of file");
        std::istringstream r(L[i++]);
        Observation o;
        r >> o.jd >> o.brightness >> o.sun.x >> o.sun.y >> o.sun.z >> o.earth.x >> o.earth.y >> o.earth.z;
        if (r.fail()) throw std::runtime_error("bad data line " + std::to_string(i));
        o.lc = k;
        sum += o.brightness;
        d.jdMin = std::min(d.jdMin, o.jd);
        d.jdMax = std::max(d.jdMax, o.jd);
        d.obs.push_back(o);
      }
      lc.mean = lc.count ? sum / lc.count : 1;
      d.lcs.push_back(lc);
    }
    // weights: "lc_number weight" pairs after the last lightcurve (as read by period_search)
    for (; i < L.size(); i++) {
      std::istringstream w(L[i]);
      int lc = 0;
      double wt = 0;
      if (w >> lc >> wt && lc >= 1 && lc <= (int)d.lcs.size()) d.lcs[lc - 1].weight = wt;
    }
  } catch (std::exception& e) {
    err = path + ": " + e.what();
    return false;
  }
  d.ok = !d.obs.empty();
  if (!d.ok) err = "no observations in " + path;
  return d.ok;
}

std::string resolveBoincLink(const std::string& path) {
  std::ifstream in(u8p(path));
  char buf[1024] = {0};
  in.read(buf, sizeof buf - 1);
  std::string s(buf);
  auto a = s.find("<soft_link>"), b = s.find("</soft_link>");
  if (a == std::string::npos || b == std::string::npos || b <= a) return path;
  std::string target = s.substr(a + 11, b - a - 11);
  while (!target.empty() && std::isspace((unsigned char)target.back())) target.pop_back();
  while (!target.empty() && std::isspace((unsigned char)target.front())) target.erase(0, 1);
  fs::path t = u8p(target);
  if (t.is_relative()) t = u8p(path).parent_path() / t;
  return t.lexically_normal().u8string();
}

static void finishOutput(OutputData& d) {
  std::sort(d.rows.begin(), d.rows.end(), [](auto& a, auto& b) { return a.period < b.period; });
  d.best = d.rows.empty() ? -1 : 0;
  for (int i = 1; i < (int)d.rows.size(); i++)
    if (d.rows[i].chi2 < d.rows[d.best].chi2) d.best = i;
}

static bool readOutputRows(const std::string& path, OutputData& d, OutputFile& f, std::string& err) {
  std::ifstream in(u8p(resolveBoincLink(path)), std::ios::binary);
  if (!in) { err = "cannot open " + path; return false; }
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  size_t end = text.rfind('\n');
  text = end == std::string::npos ? std::string() : text.substr(0, end + 1);  // drop partial last line
  std::istringstream ss(text);
  std::string line;
  f.path = path;
  f.rows = 0;
  while (nextLine(ss, line)) {
    std::istringstream s(line);
    OutputRow r;
    if (s >> r.period >> r.rms >> r.chi2 >> r.dark >> r.lambda >> r.beta) {
      if (f.rows == 0) f.firstConw = r.dark;
      d.rows.push_back(r);
      f.rows++;
      f.lastPeriod = r.period;
    }
  }
  return true;
}

bool readOutput(const std::string& path, OutputData& d, std::string& err) {
  OutputFile f;
  if (!readOutputRows(path, d, f, err)) return false;
  d.paths.push_back(path);
  d.files.push_back(f);
  finishOutput(d);
  return true;
}

bool reloadOutputs(OutputData& d, std::string& err) {
  d.rows.clear();
  bool ok = true;
  for (auto& f : d.files) ok = readOutputRows(f.path, d, f, err) && ok;
  finishOutput(d);
  return ok;
}

int InputData::totalTrials() const {
  if (!ok || perStart <= 0 || perEnd <= 0 || jdMax <= jdMin) return 0;
  double fs0 = 1 / perStart, fe = 1 / perEnd, step = 0.5 / (jdMax - jdMin) / 24 * perStepCoef;
  return step > 0 ? (int)((fs0 - fe) / step) + 1 : 0;
}

double InputData::trialPeriod(int n) const {
  double step = 0.5 / (jdMax - jdMin) / 24 * perStepCoef;
  return 1 / (1 / perStart - n * step);
}

bool readObj(const std::string& path, Mesh& m, std::string& err) {
  std::ifstream in(u8p(path));
  if (!in) { err = "cannot open " + path; return false; }
  m = Mesh();
  m.path = path;
  std::string line;
  while (nextLine(in, line)) {
    std::istringstream s(line);
    std::string tag;
    s >> tag;
    if (tag == "v") {
      V3 v;
      s >> v.x >> v.y >> v.z;
      m.v.push_back(v);
    } else if (tag == "f") {
      std::vector<int> idx;
      std::string tok;
      while (s >> tok) {
        int k = std::stoi(tok.substr(0, tok.find('/')));
        idx.push_back(k < 0 ? (int)m.v.size() + k : k - 1);
      }
      for (size_t j = 1; j + 1 < idx.size(); j++) {  // triangle fan
        m.tri.push_back(idx[0]); m.tri.push_back(idx[j]); m.tri.push_back(idx[j + 1]);
      }
    }
  }
  if (m.v.empty() || m.tri.empty()) { err = "no geometry in " + path; return false; }
  for (int k : m.tri)
    if (k < 0 || k >= (int)m.v.size()) { err = "bad face index in " + path; return false; }
  m.radius = 0;
  for (auto& v : m.v) m.radius = std::max(m.radius, norm(v));
  m.ok = true;
  return true;
}

bool readModel(const std::string& path, ModelParams& p, std::string& err) {
  std::ifstream in(u8p(path));
  if (!in) { err = "cannot open " + path; return false; }
  int ncg = 0, ncoef = 0, nph = 0;
  in >> ncg >> ncoef >> nph;
  std::vector<double> cg(ncg + 1);
  for (int i = 1; i <= ncg; i++) in >> cg[i];
  if (!in || nph != 3 || ncg != ncoef + 5 + nph) { err = "unexpected cg_dump format: " + path; return false; }
  p = ModelParams();
  p.path = path;
  p.colat = cg[ncoef + 1];
  p.lambda = cg[ncoef + 2];
  p.omega = cg[ncoef + 3];
  for (int i = 0; i < 3; i++) p.phase[i] = cg[ncoef + 4 + i];
  p.cl = std::exp(cg[ncoef + 7]);
  p.cls = cg[ncoef + 8];
  p.shapeCoef.assign(cg.begin() + 1, cg.begin() + 1 + ncoef);
  p.ok = true;
  return true;
}

bool readGauss(const std::string& path, GaussImage& g, std::string& err) {
  std::ifstream in(u8p(path));
  if (!in) { err = "cannot open " + path; return false; }
  g = GaussImage();
  g.path = path;
  std::string line;
  while (nextLine(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream s(line);
    double a; V3 n;
    if (s >> a >> n.x >> n.y >> n.z) { g.area.push_back(a); g.normal.push_back(normalize(n)); }
  }
  g.ok = !g.area.empty();
  if (!g.ok) err = "no facets in " + path;
  return g.ok;
}

FileKind classify(const std::string& path) {
  std::error_code ec;
  if (fs::is_directory(u8p(path), ec)) return FileKind::Directory;
  std::string name = u8p(path).filename().u8string();
  std::string lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower.size() > 4 && lower.substr(lower.size() - 4) == ".obj") return FileKind::Obj;
  if (lower.rfind("cg_dump", 0) == 0) return FileKind::Model;
  if (lower.rfind("shape_dump", 0) == 0) return FileKind::Gauss;
  if (lower.find("_out") != std::string::npos) return FileKind::Output;
  if (lower.find("_in") != std::string::npos) return FileKind::Input;
  // sniff: 6 numeric columns -> output, otherwise try input
  std::ifstream in(u8p(resolveBoincLink(path)));
  std::string line;
  if (!nextLine(in, line)) return FileKind::Unknown;
  std::istringstream s(line);
  double x[6];
  int n = 0;
  while (n < 6 && s >> x[n]) n++;
  std::string rest;
  if (n == 6 && !(s >> rest)) return FileKind::Output;
  if (n >= 4) return FileKind::Input;
  return FileKind::Unknown;
}

RunQuality assessRun(const OutputData& d) {
  RunQuality q;
  if (d.best < 0 || d.rows.empty()) return q;
  const auto& r = d.rows;
  double lim = r[d.best].chi2 * 1.1;
  // separate period intervals below the threshold (neighbouring trial periods below it form one interval)
  bool inside = false;
  int nBelow = 0;
  for (size_t i = 0; i < r.size(); i++) {
    bool below = r[i].chi2 <= lim;
    nBelow += below;
    if (below && !inside) q.minima++;
    inside = below;
  }
  q.belowFrac = (double)nBelow / r.size();
  bool firstRow = d.files.size() == 1 && d.best == 0;  // its 4th column is the convexity weight
  q.dark = firstRow ? -1 : r[d.best].dark;
  q.rms = r[d.best].rms;
  bool unique = q.minima <= 1 && q.belowFrac <= 0.5, darkOk = q.dark >= 0 && q.dark < 1.0;
  if (unique && (darkOk || q.dark < 0)) q.grade = 2;
  else if (q.minima <= 3 && q.dark < 2.0) q.grade = 1;
  else q.grade = 0;
  q.ok = true;
  return q;
}
