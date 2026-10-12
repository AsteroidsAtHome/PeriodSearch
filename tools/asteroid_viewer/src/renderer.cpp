#include "renderer.h"
#include <algorithm>
#include <map>
#include <vector>

static const char* kVS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aBary;
uniform mat4 uModel;
uniform mat4 uVP;
out vec3 vNormal;
out vec3 vBary;
void main() {
  vNormal = mat3(uModel) * aNormal;   // rotation only
  vBary = aBary;
  gl_Position = uVP * uModel * vec4(aPos, 1.0);
}
)";

static const char* kFS = R"(#version 330 core
in vec3 vNormal;
in vec3 vBary;
uniform vec3 uLight;
uniform vec3 uView;
uniform int uShading;
uniform float uCl;
uniform float uCls;
uniform float uExposure;
uniform int uEdges;
out vec4 fragColor;
void main() {
  vec3 n = normalize(vNormal);
  float mu0 = dot(n, uLight);
  float mu = max(dot(n, uView), 1e-4);
  vec3 c;
  if (uShading == 1) {
    // radiance of the period_search scattering law: mu0 * (cl + cls / (mu + mu0))
    float r = mu0 > 0.0 ? mu0 * (uCl + uCls / (mu + mu0)) : 0.0;
    c = vec3(r * uExposure);
    c = pow(clamp(c, 0.0, 1.0), vec3(2.2));   // keep photometric shading linear on screen
  } else {
    float key = max(mu0, 0.0);
    float fill = 0.5 + 0.5 * dot(n, normalize(vec3(-0.3, 0.2, 1.0)));
    vec3 hlf = normalize(uLight + uView);
    float spec = mu0 > 0.0 ? pow(max(dot(n, hlf), 0.0), 40.0) : 0.0;
    c = vec3(0.72, 0.69, 0.64) * (0.03 + 0.62 * key) + vec3(0.05, 0.06, 0.08) * fill + vec3(0.10) * spec;
  }
  if (uEdges == 1) {
    vec3 d = fwidth(vBary);
    vec3 a = smoothstep(vec3(0.0), d * 1.2, vBary);
    float e = min(min(a.x, a.y), a.z);
    c = mix(c * 0.55 + vec3(0.02), c, e);
  }
  fragColor = vec4(pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
)";

static GLuint compile(GLenum type, const char* src, std::string& err) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(s, sizeof log, nullptr, log);
    err = log;
    glDeleteShader(s);
    return 0;
  }
  return s;
}

bool MeshRenderer::init(std::string& err) {
  GLuint vs = compile(GL_VERTEX_SHADER, kVS, err);
  if (!vs) return false;
  GLuint fs = compile(GL_FRAGMENT_SHADER, kFS, err);
  if (!fs) return false;
  prog_ = glCreateProgram();
  glAttachShader(prog_, vs);
  glAttachShader(prog_, fs);
  glLinkProgram(prog_);
  glDeleteShader(vs);
  glDeleteShader(fs);
  GLint ok = 0;
  glGetProgramiv(prog_, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(prog_, sizeof log, nullptr, log);
    err = log;
    return false;
  }
  uModel_ = glGetUniformLocation(prog_, "uModel");
  uVP_ = glGetUniformLocation(prog_, "uVP");
  uLight_ = glGetUniformLocation(prog_, "uLight");
  uView_ = glGetUniformLocation(prog_, "uView");
  uShading_ = glGetUniformLocation(prog_, "uShading");
  uCl_ = glGetUniformLocation(prog_, "uCl");
  uCls_ = glGetUniformLocation(prog_, "uCls");
  uExposure_ = glGetUniformLocation(prog_, "uExposure");
  uEdges_ = glGetUniformLocation(prog_, "uEdges");
  glGenVertexArrays(1, &vao_);
  glGenBuffers(1, &vbo_);
  return true;
}

void MeshRenderer::upload(const Mesh& m, bool smooth) {
  // flat shading: 3 unique vertices per triangle (pos, normal, barycentric).
  // Edges between coplanar triangles (pieces of one polygonal facet) are hidden by
  // lifting that barycentric component so it never reaches 0.
  size_t nt = m.tri.size() / 3;
  std::vector<V3> tn(nt);
  for (size_t t = 0; t < nt; t++) {
    V3 p0 = m.v[m.tri[3 * t]], p1 = m.v[m.tri[3 * t + 1]], p2 = m.v[m.tri[3 * t + 2]];
    tn[t] = normalize(cross(p1 - p0, p2 - p0));
  }
  std::map<std::pair<int, int>, std::vector<size_t>> edgeTris;
  for (size_t t = 0; t < nt; t++)
    for (int k = 0; k < 3; k++) {
      int a = m.tri[3 * t + (k + 1) % 3], b = m.tri[3 * t + (k + 2) % 3];
      edgeTris[{std::min(a, b), std::max(a, b)}].push_back(t);
    }
  std::vector<V3> vn(m.v.size(), V3{0, 0, 0});
  if (smooth)
    for (size_t t = 0; t < nt; t++)
      for (int k = 0; k < 3; k++) {
        V3 p = m.v[m.tri[3 * t + k]], a = m.v[m.tri[3 * t + (k + 1) % 3]], b = m.v[m.tri[3 * t + (k + 2) % 3]];
        double ang = std::acos(std::max(-1.0, std::min(1.0, dot(normalize(a - p), normalize(b - p)))));
        vn[m.tri[3 * t + k]] = vn[m.tri[3 * t + k]] + tn[t] * ang;
      }
  std::vector<float> buf;
  buf.reserve(m.tri.size() * 9);
  for (size_t t = 0; t < nt; t++) {
    float lift[3] = {0, 0, 0};
    for (int k = 0; k < 3; k++) {  // edge opposite vertex k
      int a = m.tri[3 * t + (k + 1) % 3], b = m.tri[3 * t + (k + 2) % 3];
      auto& ts = edgeTris[{std::min(a, b), std::max(a, b)}];
      bool flat = ts.size() == 2 && dot(tn[ts[0]], tn[ts[1]]) > 0.99999;
      if (flat) lift[k] = 1;
    }
    for (int k = 0; k < 3; k++) {
      V3 p = m.v[m.tri[3 * t + k]], n = smooth ? normalize(vn[m.tri[3 * t + k]]) : tn[t];
      float bary[3] = {lift[0], lift[1], lift[2]};
      bary[k] += 1;
      float v[9] = {(float)p.x, (float)p.y, (float)p.z, (float)n.x, (float)n.y, (float)n.z, bary[0], bary[1], bary[2]};
      buf.insert(buf.end(), v, v + 9);
    }
  }
  count_ = (GLsizei)(buf.size() / 9);
  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(buf.size() * sizeof(float)), buf.data(), GL_STATIC_DRAW);
  for (int a = 0; a < 3; a++) {
    glVertexAttribPointer(a, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(size_t)(a * 3 * sizeof(float)));
    glEnableVertexAttribArray(a);
  }
  glBindVertexArray(0);
}

void MeshRenderer::draw(const M4& model, const M4& vp, V3 light, V3 view, int shading, float cl, float cls,
                        float exposure, bool edges) {
  if (!count_) return;
  glUseProgram(prog_);
  glUniformMatrix4fv(uModel_, 1, GL_FALSE, model.a);
  glUniformMatrix4fv(uVP_, 1, GL_FALSE, vp.a);
  float l[3] = {(float)light.x, (float)light.y, (float)light.z};
  float v[3] = {(float)view.x, (float)view.y, (float)view.z};
  glUniform3fv(uLight_, 1, l);
  glUniform3fv(uView_, 1, v);
  glUniform1i(uShading_, shading);
  glUniform1f(uCl_, cl);
  glUniform1f(uCls_, cls);
  glUniform1f(uExposure_, exposure);
  glUniform1i(uEdges_, edges ? 1 : 0);
  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLES, 0, count_);
  glBindVertexArray(0);
  glUseProgram(0);
}

void MeshRenderer::destroy() {
  if (vbo_) glDeleteBuffers(1, &vbo_);
  if (vao_) glDeleteVertexArrays(1, &vao_);
  if (prog_) glDeleteProgram(prog_);
  vbo_ = vao_ = prog_ = 0;
}
