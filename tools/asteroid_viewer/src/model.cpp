#include "model.h"
#include <cmath>

Facets facetsFromMesh(const Mesh& m) {
  Facets f;
  for (size_t i = 0; i + 2 < m.tri.size(); i += 3) {
    V3 a = m.v[m.tri[i]], b = m.v[m.tri[i + 1]], c = m.v[m.tri[i + 2]];
    V3 n = cross(b - a, c - a);
    double l = norm(n);
    if (l <= 0) continue;
    f.area.push_back(0.5 * l);
    f.normal.push_back(n * (1.0 / l));
  }
  return f;
}

Facets facetsFromGauss(const GaussImage& g) {
  Facets f;
  f.area = g.area;
  f.normal = g.normal;
  return f;
}

M3 eclToBody(const ModelParams& p, double epoch, double phi0rad, double jd) {
  double cb = std::cos(p.colat), sb = std::sin(p.colat), cl = std::cos(p.lambda), sl = std::sin(p.lambda);
  M3 B;
  B.m[0][0] = cb * cl; B.m[0][1] = cb * sl; B.m[0][2] = -sb;
  B.m[1][0] = -sl;     B.m[1][1] = cl;      B.m[1][2] = 0;
  B.m[2][0] = sb * cl; B.m[2][1] = sb * sl; B.m[2][2] = cb;
  double f = std::fmod(p.omega * (jd - epoch) + phi0rad, 2 * M_PI);
  double cf = std::cos(f), sf = std::sin(f);
  M3 F;
  F.m[0][0] = cf;  F.m[0][1] = sf; F.m[0][2] = 0;
  F.m[1][0] = -sf; F.m[1][1] = cf; F.m[1][2] = 0;
  F.m[2][0] = 0;   F.m[2][1] = 0;  F.m[2][2] = 1;
  return F * B;
}

double rotPhase(const ModelParams& p, double epoch, double phi0rad, double jd) {
  double f = std::fmod((p.omega * (jd - epoch) + phi0rad) / (2 * M_PI), 1.0);
  return f < 0 ? f + 1 : f;
}

double brightness(const Facets& f, const ModelParams& p, const M3& T, V3 sun, V3 earth) {
  V3 e = T * normalize(earth), e0 = T * normalize(sun);
  double br = 0;
  for (size_t i = 0; i < f.area.size(); i++) {
    double mu = dot(e, f.normal[i]), mu0 = dot(e0, f.normal[i]);
    if (mu > 1e-8 && mu0 > 1e-8) br += f.area[i] * mu * mu0 * (p.cl + p.cls / (mu + mu0));
  }
  double alpha = std::acos(std::fmax(-1.0, std::fmin(1.0, dot(normalize(earth), normalize(sun)))));
  double scale = 1 + p.phase[0] * std::exp(-alpha / p.phase[1]) + p.phase[2] * alpha;
  return br * scale;
}

Fit fitLightcurves(const InputData& in, const Facets& f, const ModelParams& p) {
  Fit r;
  if (!in.ok || !p.ok || f.area.empty()) return r;
  size_t n = in.obs.size();
  r.model.resize(n); r.relObs.resize(n); r.relModel.resize(n);
  r.lcScale.assign(in.lcs.size(), 1.0);
  double epoch = in.epoch(), phi0 = in.phi0deg * M_PI / 180, chi = 0;
  for (auto& lc : in.lcs) {
    double sumModel = 0;
    for (int j = lc.first; j < lc.first + lc.count; j++) {
      auto& o = in.obs[j];
      r.model[j] = brightness(f, p, eclToBody(p, epoch, phi0, o.jd), o.sun, o.earth);
      sumModel += r.model[j];
    }
    // relative LC: model rescaled so that its mean equals the observed mean (mrqcof.c)
    double k = lc.calibrated || sumModel <= 0 ? 1.0 : lc.mean * lc.count / sumModel;
    r.lcScale[&lc - &in.lcs[0]] = k;
    for (int j = lc.first; j < lc.first + lc.count; j++) {
      r.model[j] *= k;
      r.relObs[j] = in.obs[j].brightness / lc.mean;
      r.relModel[j] = r.model[j] / lc.mean;
      double dy = (in.obs[j].brightness - r.model[j]) / lc.mean;
      chi += lc.weight * dy * dy;  // weighted like period_search (mrqcof.c)
    }
  }
  r.rms = std::sqrt(chi / n);
  r.ok = true;
  return r;
}
