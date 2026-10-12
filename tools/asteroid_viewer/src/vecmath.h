#pragma once
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

struct V3 { double x = 0, y = 0, z = 0; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 normalize(V3 a) { double n = norm(a); return n > 0 ? a * (1.0 / n) : a; }

// 3x3 matrix, row-major: m[r][c]
struct M3 {
  double m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  V3 operator*(V3 v) const {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
  }
  M3 operator*(const M3& b) const {
    M3 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        r.m[i][j] = 0;
        for (int k = 0; k < 3; k++) r.m[i][j] += m[i][k] * b.m[k][j];
      }
    return r;
  }
  M3 T() const {
    M3 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) r.m[i][j] = m[j][i];
    return r;
  }
};

// 4x4 column-major float matrix for OpenGL
struct M4 {
  float a[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  float& operator()(int r, int c) { return a[c * 4 + r]; }
  float operator()(int r, int c) const { return a[c * 4 + r]; }
  M4 operator*(const M4& b) const {
    M4 r;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        float s = 0;
        for (int k = 0; k < 4; k++) s += (*this)(i, k) * b(k, j);
        r(i, j) = s;
      }
    return r;
  }
};

inline M4 fromM3(const M3& m) {
  M4 r;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) r(i, j) = (float)m.m[i][j];
  return r;
}

inline M4 perspective(float fovy, float aspect, float zn, float zf) {
  M4 r;
  float f = 1.0f / std::tan(fovy / 2);
  r(0, 0) = f / aspect; r(1, 1) = f; r(2, 2) = (zf + zn) / (zn - zf);
  r(2, 3) = 2 * zf * zn / (zn - zf); r(3, 2) = -1; r(3, 3) = 0;
  return r;
}

inline M4 lookAt(V3 eye, V3 target, V3 up) {
  V3 f = normalize(target - eye);
  if (std::fabs(dot(f, normalize(up))) > 0.999) up = std::fabs(f.z) < 0.9 ? V3{0, 0, 1} : V3{1, 0, 0};
  V3 s = normalize(cross(f, up));
  V3 u = cross(s, f);
  M4 r;
  r(0, 0) = (float)s.x; r(0, 1) = (float)s.y; r(0, 2) = (float)s.z; r(0, 3) = (float)-dot(s, eye);
  r(1, 0) = (float)u.x; r(1, 1) = (float)u.y; r(1, 2) = (float)u.z; r(1, 3) = (float)-dot(u, eye);
  r(2, 0) = (float)-f.x; r(2, 1) = (float)-f.y; r(2, 2) = (float)-f.z; r(2, 3) = (float)dot(f, eye);
  return r;
}
