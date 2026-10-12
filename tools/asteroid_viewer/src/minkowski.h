// Minkowski problem: convex polyhedron from its Gaussian image (facet areas + normals).
#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <vector>
#include "data.h"

struct MinkowskiResult {
  bool ok = false;
  Mesh mesh;              // centred on the centre of mass, unit volume-equivalent radius, z = spin axis
  double darkFacet = 0;   // non-closure |sum A n| / sum A, added as one extra facet
  double areaMismatch = 0;  // 0.5 * sum |a_i - A_i| (normalised areas) after the fit
  int iterations = 0;
  std::string message;
};

struct MinkowskiOptions {
  // How the non-closure of the Gaussian image is removed:
  //  false: one extra "dark facet" (as period_search reports it)
  //  true : areas rescaled smoothly, A_i (1 + w.n_i), so that sum A n = 0 (no flat patch; smooth model)
  bool smoothClosure = false;
  const Mesh* warmStart = nullptr;  // start from the support function of this (similar) convex mesh
  double mismatchTol = 0;           // > 0: stop once the area mismatch is below this (faster, for display)
};

// progress(iteration, mismatch) is called now and then; returns early when *cancel becomes true
MinkowskiResult minkowski(const std::vector<double>& area, const std::vector<V3>& normal,
                          std::function<void(int, double)> progress = nullptr, std::atomic<bool>* cancel = nullptr,
                          const MinkowskiOptions& opt = MinkowskiOptions());
