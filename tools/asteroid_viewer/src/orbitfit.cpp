#include "orbitfit.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
const double kGauss = 0.01720209895;  // Gaussian gravitational constant [rad/day, AU^1.5]
const double D2R = M_PI / 180;

struct Pt { double t; V3 r; };

// position for elements p = {a, e, i, om, w, M0} (angles in rad) at time t (days from the epoch)
V3 kepler(const double* p, double t) {
  double a = p[0], e = std::clamp(p[1], 0.0, 0.97);
  double n = kGauss / std::pow(std::max(a, 0.05), 1.5);
  double M = std::fmod(p[5] + n * t, 2 * M_PI), E = M;
  for (int k = 0; k < 40; k++) {
    double d = (E - e * std::sin(E) - M) / (1 - e * std::cos(E));
    E -= d;
    if (std::fabs(d) < 1e-13) break;
  }
  double xo = a * (std::cos(E) - e), yo = a * std::sqrt(1 - e * e) * std::sin(E);
  double cw = std::cos(p[4]), sw = std::sin(p[4]), co = std::cos(p[3]), so = std::sin(p[3]), ci = std::cos(p[2]), si = std::sin(p[2]);
  double x1 = cw * xo - sw * yo, y1 = sw * xo + cw * yo;
  return V3{co * x1 - so * ci * y1, so * x1 + co * ci * y1, si * y1};
}

double cost(const double* p, const std::vector<Pt>& pts) {
  double s = 0;
  for (auto& q : pts) {
    V3 d = kepler(p, q.t) - q.r;
    s += d.x * d.x + d.y * d.y + d.z * d.z;
  }
  return s;
}

// Levenberg-Marquardt with a numerical Jacobian
double lm(double* p, const std::vector<Pt>& pts) {
  const int P = 6;
  size_t m = pts.size() * 3;
  std::vector<double> r(m), J(m * P);
  auto resid = [&](const double* q, double* out) {
    for (size_t k = 0; k < pts.size(); k++) {
      V3 d = kepler(q, pts[k].t) - pts[k].r;
      out[3 * k] = d.x; out[3 * k + 1] = d.y; out[3 * k + 2] = d.z;
    }
  };
  double lambda = 1e-3, c = cost(p, pts);
  for (int it = 0; it < 60; it++) {
    resid(p, r.data());
    for (int j = 0; j < P; j++) {
      double q[6];
      std::copy(p, p + P, q);
      double h = j == 0 ? 1e-6 : (j == 1 ? 1e-7 : 1e-7);
      q[j] += h;
      std::vector<double> r2(m);
      resid(q, r2.data());
      for (size_t k = 0; k < m; k++) J[k * P + j] = (r2[k] - r[k]) / h;
    }
    double A[6][6] = {}, g[6] = {};
    for (size_t k = 0; k < m; k++)
      for (int a = 0; a < P; a++) {
        g[a] += J[k * P + a] * r[k];
        for (int b = 0; b < P; b++) A[a][b] += J[k * P + a] * J[k * P + b];
      }
    bool improved = false;
    for (int tries = 0; tries < 8 && !improved; tries++) {
      double M[6][7];
      for (int a = 0; a < P; a++) {
        for (int b = 0; b < P; b++) M[a][b] = A[a][b] + (a == b ? lambda * (A[a][a] + 1e-12) : 0);
        M[a][6] = -g[a];
      }
      for (int col = 0; col < P; col++) {  // Gauss elimination with partial pivoting
        int piv = col;
        for (int rr = col + 1; rr < P; rr++) if (std::fabs(M[rr][col]) > std::fabs(M[piv][col])) piv = rr;
        for (int k = 0; k < 7; k++) std::swap(M[col][k], M[piv][k]);
        if (std::fabs(M[col][col]) < 1e-300) break;
        for (int rr = 0; rr < P; rr++) {
          if (rr == col) continue;
          double f = M[rr][col] / M[col][col];
          for (int k = col; k < 7; k++) M[rr][k] -= f * M[col][k];
        }
      }
      double q[6];
      for (int a = 0; a < P; a++) q[a] = p[a] + (std::fabs(M[a][a]) > 1e-300 ? M[a][6] / M[a][a] : 0);
      q[1] = std::clamp(q[1], 0.0, 0.97);
      q[0] = std::max(q[0], 0.1);
      double c2 = cost(q, pts);
      if (c2 < c) {
        std::copy(q, q + P, p);
        improved = c - c2 > 1e-14 * c;
        c = c2;
        lambda = std::max(lambda * 0.3, 1e-9);
        if (!improved) return c;
      } else {
        lambda *= 10;
      }
    }
    if (!improved) break;
  }
  return c;
}
}  // namespace

OrbitFit fitOrbit(const InputData& in) {
  OrbitFit f;
  if (!in.ok || in.obs.size() < 6) return f;
  // up to 300 positions spread over the time span
  std::vector<Pt> all;
  for (auto& o : in.obs) all.push_back({o.jd, V3{} - o.sun});
  std::sort(all.begin(), all.end(), [](const Pt& x, const Pt& y) { return x.t < y.t; });
  std::vector<Pt> pts;
  size_t step = std::max<size_t>(1, all.size() / 300);
  for (size_t k = 0; k < all.size(); k += step) pts.push_back(all[k]);
  if (pts.back().t - pts.front().t < 1) return f;  // one night: no orbit
  double epoch = 0;
  for (auto& q : pts) epoch += q.t;
  epoch = std::round(epoch / pts.size());
  for (auto& q : pts) q.t -= epoch;

  // starting orbital plane from the angular momentum direction (sum of r_k x r_k+1)
  V3 h{0, 0, 0};
  double rMean = 0;
  for (size_t k = 0; k + 1 < pts.size(); k++) h = h + cross(pts[k].r, pts[k + 1].r);
  for (auto& q : pts) rMean += norm(q.r);
  rMean /= pts.size();
  if (norm(h) < 1e-12) h = V3{0, 0, 1};
  h = normalize(h);
  double inc = std::acos(std::clamp(h.z, -1.0, 1.0)), node = std::atan2(h.x, -h.y);

  double best[6] = {}, bestC = 1e300;
  for (double a0 : {rMean * 0.9, rMean, rMean * 1.15})
    for (double e0 : {0.05, 0.2})
      for (int wi = 0; wi < 6; wi++)
        for (int mi = 0; mi < 6; mi++) {
          double p[6] = {a0, e0, inc, node, wi * M_PI / 3, mi * M_PI / 3};
          // quick screening before the full fit
          if (cost(p, pts) > 50 * bestC && bestC < 1e300) continue;
          double c = lm(p, pts);
          if (c < bestC) { bestC = c; std::copy(p, p + 6, best); }
        }
  f.rms = std::sqrt(bestC / pts.size());
  f.points = (int)pts.size();
  f.a = best[0];
  f.e = best[1];
  f.i = best[2] / D2R;
  f.om = std::fmod(best[3] / D2R + 720, 360);
  f.w = std::fmod(best[4] / D2R + 720, 360);
  f.M0 = std::fmod(best[5] / D2R + 720, 360);
  f.n = kGauss / std::pow(f.a, 1.5) / D2R;
  f.epoch = epoch;
  f.ok = f.rms < 0.02 * f.a;  // two-body orbit fits the observations (perturbations, light time: ~1e-3 AU)
  return f;
}
