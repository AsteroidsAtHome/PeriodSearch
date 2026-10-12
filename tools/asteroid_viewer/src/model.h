// Photometric model identical to period_search (bright.c + matrix.c + blmatrix.c
// + phasec.c): Lambert + Lommel-Seeliger scattering on a convex body, exp-lin
// phase function, relative lightcurves normalised to their mean.
#pragma once
#include <vector>
#include "data.h"

struct Facets {  // area + outward normal per facet, body frame (z = spin axis)
  std::vector<double> area;
  std::vector<V3> normal;
};

Facets facetsFromMesh(const Mesh& m);
Facets facetsFromGauss(const GaussImage& g);

// ecliptic -> body rotation at time jd (period_search's tmat = Fz(phi) * B(colat, lambda))
M3 eclToBody(const ModelParams& p, double epoch, double phi0rad, double jd);
// rotational phase in [0,1)
double rotPhase(const ModelParams& p, double epoch, double phi0rad, double jd);

// disk-integrated brightness, sun/earth = asteroid-centric directions (ecliptic)
double brightness(const Facets& f, const ModelParams& p, const M3& eclToBodyM, V3 sun, V3 earth);

struct Fit {
  bool ok = false;
  std::vector<double> model;     // scaled like period_search (relative LCs matched to data mean)
  std::vector<double> relObs;    // obs / LC mean
  std::vector<double> relModel;  // model / LC mean (same normalisation as relObs)
  std::vector<double> lcScale;   // per-lightcurve factor applied to the raw model
  double rms = 0;                // comparable to column 2 of period_search_out
};
Fit fitLightcurves(const InputData& in, const Facets& f, const ModelParams& p);
