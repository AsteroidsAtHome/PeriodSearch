#pragma once
#include "data.h"
#include "gl_loader.h"

class MeshRenderer {
 public:
  bool init(std::string& err);
  // smooth: per-vertex normals (angle-weighted average of the adjacent triangles) instead of flat facets
  void upload(const Mesh& m, bool smooth = false);
  // model: body->world, viewProj: world->clip; light/view dirs are world-space unit vectors
  // shading: 0 = studio (Lambert + ambient), 1 = photometric (Lambert + Lommel-Seeliger, as in the fit)
  void draw(const M4& model, const M4& viewProj, V3 lightDir, V3 viewDir, int shading,
            float cl, float cls, float exposure, bool edges);
  void destroy();

 private:
  GLuint prog_ = 0, vao_ = 0, vbo_ = 0;
  GLsizei count_ = 0;
  GLint uModel_ = -1, uVP_ = -1, uLight_ = -1, uView_ = -1, uShading_ = -1, uCl_ = -1, uCls_ = -1,
        uExposure_ = -1, uEdges_ = -1;
};
