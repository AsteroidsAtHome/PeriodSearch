// Heliocentric orbit of the asteroid fitted to the positions in period_search_in
// (the Sun vectors of the observations): a two-body Kepler orbit, least squares.
#pragma once
#include "data.h"

struct OrbitFit {
  bool ok = false;
  double a = 0, e = 0, i = 0, om = 0, w = 0, M0 = 0;  // AU, -, deg, deg, deg, deg (mean anomaly at epoch)
  double n = 0, epoch = 0;                           // deg/day, JD
  double rms = 0;                                    // AU, positions vs the fitted orbit
  int points = 0;
};

// blocking (a fraction of a second); run it in a background thread
OrbitFit fitOrbit(const InputData& in);
