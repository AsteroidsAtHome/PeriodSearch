// Solves the Minkowski problem by minimising  sum(A_i h_i) / V(h)^(1/3)  over the support
// distances h_i (dV/dh_i = actual facet area a_i, so at the minimum a_i is proportional to A_i).
// The polytope {x : n_i.x <= h_i} is built through its polar: the convex hull of the points
// n_i / h_i. Each hull triangle gives one polytope vertex, each hull point one facet.
#include "minkowski.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace {

struct Hull {
  struct Face { int v[3]; V3 n; double d; bool alive; };
  std::vector<Face> faces;
};

inline long long ekey(int a, int b) { return (long long)a * 1000003LL + b; }

// incremental 3D convex hull; false if the points are degenerate
bool convexHull(const std::vector<V3>& p, Hull& H) {
  int n = (int)p.size();
  H.faces.clear();
  if (n < 4) return false;
  double scale = 0;
  for (auto& q : p) scale = std::max(scale, norm(q));
  double eps = 1e-12 * scale;
  // initial tetrahedron
  int i0 = 0, i1 = -1, i2 = -1, i3 = -1;
  double best = 0;
  for (int i = 0; i < n; i++) { double d = norm(p[i] - p[i0]); if (d > best) { best = d; i1 = i; } }
  if (i1 < 0) return false;
  best = 0;
  V3 u = normalize(p[i1] - p[i0]);
  for (int i = 0; i < n; i++) { double d = norm(cross(p[i] - p[i0], u)); if (d > best) { best = d; i2 = i; } }
  if (i2 < 0 || best < eps) return false;
  V3 pn = normalize(cross(p[i1] - p[i0], p[i2] - p[i0]));
  best = 0;
  for (int i = 0; i < n; i++) { double d = std::fabs(dot(p[i] - p[i0], pn)); if (d > best) { best = d; i3 = i; } }
  if (i3 < 0 || best < eps) return false;
  V3 inside = (p[i0] + p[i1] + p[i2] + p[i3]) * 0.25;
  std::unordered_map<long long, int> edge;  // directed edge -> face
  auto addFace = [&](int a, int b, int c) {
    Hull::Face f;
    V3 nn = cross(p[b] - p[a], p[c] - p[a]);
    if (dot(nn, p[a] - inside) < 0) { std::swap(b, c); nn = nn * -1.0; }
    f.v[0] = a; f.v[1] = b; f.v[2] = c;
    f.n = normalize(nn);
    f.d = dot(f.n, p[a]);
    f.alive = true;
    int id = (int)H.faces.size();
    H.faces.push_back(f);
    edge[ekey(a, b)] = id; edge[ekey(b, c)] = id; edge[ekey(c, a)] = id;
  };
  addFace(i0, i1, i2); addFace(i0, i1, i3); addFace(i0, i2, i3); addFace(i1, i2, i3);
  std::vector<char> visible;
  for (int q = 0; q < n; q++) {
    if (q == i0 || q == i1 || q == i2 || q == i3) continue;
    visible.assign(H.faces.size(), 0);
    bool any = false;
    for (size_t f = 0; f < H.faces.size(); f++)
      if (H.faces[f].alive && dot(H.faces[f].n, p[q]) - H.faces[f].d > eps) { visible[f] = 1; any = true; }
    if (!any) continue;
    std::vector<std::pair<int, int>> horizon;
    for (size_t f = 0; f < H.faces.size(); f++) {
      if (!visible[f]) continue;
      for (int k = 0; k < 3; k++) {
        int a = H.faces[f].v[k], b = H.faces[f].v[(k + 1) % 3];
        auto it = edge.find(ekey(b, a));
        if (it == edge.end()) return false;
        if (!visible[it->second]) horizon.push_back({a, b});
      }
    }
    for (size_t f = 0; f < visible.size(); f++)
      if (visible[f]) {
        H.faces[f].alive = false;
        for (int k = 0; k < 3; k++) {
          auto it = edge.find(ekey(H.faces[f].v[k], H.faces[f].v[(k + 1) % 3]));
          if (it != edge.end() && it->second == (int)f) edge.erase(it);
        }
      }
    for (auto& e : horizon) {
      Hull::Face f;
      int a = e.first, b = e.second;
      V3 nn = cross(p[b] - p[a], p[q] - p[a]);
      f.v[0] = a; f.v[1] = b; f.v[2] = q;
      f.n = normalize(nn);
      f.d = dot(f.n, p[a]);
      f.alive = true;
      int id = (int)H.faces.size();
      H.faces.push_back(f);
      edge[ekey(a, b)] = id; edge[ekey(b, q)] = id; edge[ekey(q, a)] = id;
    }
  }
  return true;
}

struct Poly {
  std::vector<V3> vert;                 // one per hull face (may repeat when >3 facets meet)
  std::vector<std::vector<int>> facet;  // per facet: vertex indices, counter-clockwise about n_i
  std::vector<double> area;
  double volume = 0;
};

bool polytope(const std::vector<V3>& N, const std::vector<double>& h, Poly& P) {
  int n = (int)N.size();
  std::vector<V3> dual(n);
  for (int i = 0; i < n; i++) dual[i] = N[i] * (1.0 / h[i]);
  Hull H;
  if (!convexHull(dual, H)) return false;
  P = Poly();
  P.facet.assign(n, {});
  P.area.assign(n, 0);
  for (auto& f : H.faces) {
    if (!f.alive) continue;
    if (f.d <= 0) return false;  // origin not inside
    int id = (int)P.vert.size();
    P.vert.push_back(f.n * (1.0 / f.d));
    for (int k = 0; k < 3; k++) P.facet[f.v[k]].push_back(id);
  }
  for (int i = 0; i < n; i++) {
    auto& ids = P.facet[i];
    if (ids.size() < 3) { ids.clear(); continue; }
    V3 e1 = normalize(std::fabs(N[i].x) < 0.9 ? cross(N[i], V3{1, 0, 0}) : cross(N[i], V3{0, 1, 0}));
    V3 e2 = cross(N[i], e1);
    V3 c{0, 0, 0};
    for (int k : ids) c = c + P.vert[k];
    c = c * (1.0 / ids.size());
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
      V3 da = P.vert[a] - c, db = P.vert[b] - c;
      return std::atan2(dot(da, e2), dot(da, e1)) < std::atan2(dot(db, e2), dot(db, e1));
    });
    double A = 0;
    for (size_t k = 0; k < ids.size(); k++)
      A += dot(cross(P.vert[ids[k]] - c, P.vert[ids[(k + 1) % ids.size()]] - c), N[i]);
    P.area[i] = 0.5 * A;
    P.volume += h[i] * P.area[i] / 3;
  }
  return P.volume > 0;
}

}  // namespace

MinkowskiResult minkowski(const std::vector<double>& areaIn, const std::vector<V3>& normalIn,
                          std::function<void(int, double)> progress, std::atomic<bool>* cancel,
                          const MinkowskiOptions& opt) {
  MinkowskiResult R;
  std::vector<double> A = areaIn;
  std::vector<V3> N;
  for (auto& v : normalIn) N.push_back(normalize(v));
  double sum = 0;
  V3 c{0, 0, 0};
  for (size_t i = 0; i < A.size(); i++) { sum += A[i]; c = c + N[i] * A[i]; }
  if (A.size() < 4 || sum <= 0) { R.message = "empty Gaussian image"; return R; }
  double dark = norm(c);
  R.darkFacet = dark / sum;
  if (opt.smoothClosure && dark > 0) {
    // solve (sum A n n^T) w = -c, then A_i *= 1 + w.n_i  ->  sum A n = 0
    double M[3][3] = {{0}}, b[3] = {-c.x, -c.y, -c.z};
    for (size_t i = 0; i < A.size(); i++) {
      double v[3] = {N[i].x, N[i].y, N[i].z};
      for (int r = 0; r < 3; r++)
        for (int q = 0; q < 3; q++) M[r][q] += A[i] * v[r] * v[q];
    }
    double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                 M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    if (std::fabs(det) > 0) {
      V3 w;
      double* wp[3] = {&w.x, &w.y, &w.z};
      for (int k = 0; k < 3; k++) {  // Cramer's rule
        double T[3][3];
        for (int r = 0; r < 3; r++)
          for (int q = 0; q < 3; q++) T[r][q] = q == k ? b[r] : M[r][q];
        *wp[k] = (T[0][0] * (T[1][1] * T[2][2] - T[1][2] * T[2][1]) - T[0][1] * (T[1][0] * T[2][2] - T[1][2] * T[2][0]) +
                  T[0][2] * (T[1][0] * T[2][1] - T[1][1] * T[2][0])) / det;
      }
      sum = 0;
      for (size_t i = 0; i < A.size(); i++) { A[i] *= std::max(1e-6, 1 + dot(w, N[i])); sum += A[i]; }
    }
  } else if (dark > 0) {
    A.push_back(dark); N.push_back(c * (-1.0 / dark)); sum += dark;  // close the Gaussian image with a dark facet
  }
  for (auto& a : A) a /= sum;
  int n = (int)A.size();

  // objective in x = log h (keeps h > 0)
  Poly P;
  auto eval = [&](const std::vector<double>& x, std::vector<double>& g) -> double {
    std::vector<double> h(n);
    for (int i = 0; i < n; i++) h[i] = std::exp(x[i]);
    if (!polytope(N, h, P)) return std::numeric_limits<double>::infinity();
    double S = 0;
    for (int i = 0; i < n; i++) S += A[i] * h[i];
    double V = P.volume, V13 = std::cbrt(V);
    g.resize(n);
    for (int i = 0; i < n; i++) g[i] = (A[i] / V13 - S / 3 * P.area[i] / (V * V13)) * h[i];
    return S / V13;
  };

  // L-BFGS
  const int M = 12;
  std::vector<double> x(n, 0.0), g, xn(n), gn;
  if (opt.warmStart && opt.warmStart->ok) {  // h_i = support function of the given mesh
    double mean = 0;
    for (int i = 0; i < n; i++) {
      double hmax = -1e300;
      for (auto& v : opt.warmStart->v) hmax = std::max(hmax, dot(N[i], v));
      x[i] = std::log(std::max(1e-6, hmax));
      mean += x[i];
    }
  }
  double F = eval(x, g);
  if (!std::isfinite(F)) { R.message = "cannot build the initial polytope"; return R; }
  std::vector<std::vector<double>> S, Y;
  std::vector<double> rho;
  int it = 0;
  for (; it < 5000; it++) {
    if (cancel && cancel->load()) { R.message = "cancelled"; return R; }
    // two-loop recursion -> direction d = -H g
    std::vector<double> q = g, alpha(S.size());
    for (int k = (int)S.size() - 1; k >= 0; k--) {
      double a = 0;
      for (int i = 0; i < n; i++) a += S[k][i] * q[i];
      a *= rho[k];
      alpha[k] = a;
      for (int i = 0; i < n; i++) q[i] -= a * Y[k][i];
    }
    double gamma = 1;
    if (!S.empty()) {
      double sy = 0, yy = 0;
      for (int i = 0; i < n; i++) { sy += S.back()[i] * Y.back()[i]; yy += Y.back()[i] * Y.back()[i]; }
      gamma = yy > 0 ? sy / yy : 1;
    } else {
      double gg = 0;
      for (double v : g) gg += v * v;
      gamma = gg > 0 ? 0.1 / std::sqrt(gg) : 1;
    }
    for (int i = 0; i < n; i++) q[i] *= gamma;
    for (int k = 0; k < (int)S.size(); k++) {
      double b = 0;
      for (int i = 0; i < n; i++) b += Y[k][i] * q[i];
      b *= rho[k];
      for (int i = 0; i < n; i++) q[i] += S[k][i] * (alpha[k] - b);
    }
    double gd = 0;
    for (int i = 0; i < n; i++) gd -= g[i] * q[i];
    if (gd >= 0) {  // not a descent direction: restart with steepest descent
      S.clear(); Y.clear(); rho.clear();
      for (int i = 0; i < n; i++) q[i] = g[i] * gamma;
      gd = 0;
      for (int i = 0; i < n; i++) gd -= g[i] * q[i];
    }
    // backtracking line search (Armijo)
    double step = 1, Fn = 0;
    bool ok = false;
    for (int ls = 0; ls < 40; ls++) {
      for (int i = 0; i < n; i++) xn[i] = x[i] - step * q[i];
      Fn = eval(xn, gn);
      if (std::isfinite(Fn) && Fn <= F + 1e-4 * step * gd) { ok = true; break; }
      step *= 0.5;
    }
    if (!ok) break;
    std::vector<double> s(n), y(n);
    double sy = 0;
    for (int i = 0; i < n; i++) { s[i] = xn[i] - x[i]; y[i] = gn[i] - g[i]; sy += s[i] * y[i]; }
    if (sy > 1e-20) {
      S.push_back(s); Y.push_back(y); rho.push_back(1 / sy);
      if ((int)S.size() > M) { S.erase(S.begin()); Y.erase(Y.begin()); rho.erase(rho.begin()); }
    }
    double rel = (F - Fn) / std::max(1e-300, std::fabs(F));
    x = xn; g = gn; F = Fn;
    if (it % 10 == 0 && (progress || opt.mismatchTol > 0)) {
      double asum = 0, mm = 0;
      for (double a : P.area) asum += a;
      for (int i = 0; i < n; i++) mm += std::fabs(P.area[i] / asum - A[i]);
      if (progress && it % 25 == 0) progress(it, mm / 2);
      if (opt.mismatchTol > 0 && mm / 2 < opt.mismatchTol) break;
    }
    if (rel < 1e-15 && it > 50) break;
  }
  R.iterations = it;
  std::vector<double> h(n);
  for (int i = 0; i < n; i++) h[i] = std::exp(x[i]);
  if (!polytope(N, h, P)) { R.message = "final polytope failed"; return R; }
  double asum = 0, mm = 0;
  for (double a : P.area) asum += a;
  for (int i = 0; i < n; i++) mm += std::fabs(P.area[i] / asum - A[i]);
  R.areaMismatch = mm / 2;

  // mesh: merge coincident vertices, fan-triangulate facets, centre and scale
  Mesh& m = R.mesh;
  double scale = 0;
  for (auto& v : P.vert) scale = std::max(scale, norm(v));
  std::vector<int> remap(P.vert.size(), -1);
  for (size_t i = 0; i < P.vert.size(); i++) {
    for (size_t j = 0; j < m.v.size() && remap[i] < 0; j++)
      if (norm(P.vert[i] - m.v[j]) < 1e-9 * scale) remap[i] = (int)j;
    if (remap[i] < 0) { remap[i] = (int)m.v.size(); m.v.push_back(P.vert[i]); }
  }
  for (int i = 0; i < n; i++) {
    std::vector<int> poly;
    for (int k : P.facet[i]) {
      int r = remap[k];
      if (poly.empty() || (poly.back() != r && poly.front() != r)) poly.push_back(r);
    }
    for (size_t k = 1; k + 1 < poly.size(); k++) {
      m.tri.push_back(poly[0]); m.tri.push_back(poly[k]); m.tri.push_back(poly[k + 1]);
    }
  }
  // centre of mass and volume (tetrahedra from an interior point)
  V3 o{0, 0, 0};
  for (auto& v : m.v) o = o + v;
  o = o * (1.0 / m.v.size());
  V3 cm{0, 0, 0};
  double vol = 0;
  for (size_t t = 0; t + 2 < m.tri.size(); t += 3) {
    V3 a = m.v[m.tri[t]] - o, b = m.v[m.tri[t + 1]] - o, cc = m.v[m.tri[t + 2]] - o;
    double v6 = dot(a, cross(b, cc)) / 6;
    vol += v6;
    cm = cm + (a + b + cc) * (v6 / 4);
  }
  cm = cm * (1.0 / vol) + o;
  double r = std::cbrt(3 * vol / (4 * M_PI));
  m.radius = 0;
  for (auto& v : m.v) { v = (v - cm) * (1.0 / r); m.radius = std::max(m.radius, norm(v)); }
  m.ok = !m.tri.empty();
  R.ok = m.ok;
  if (!R.ok) R.message = "no facets";
  return R;
}
