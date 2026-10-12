#include "gl_loader.h"
#define PS_DEFINE(ret, name, ...) PFN_ps_gl##name gl##name = nullptr;
PS_GL_FUNCS(PS_DEFINE)
#undef PS_DEFINE

const char* psLoadGL(void* (*getProc)(const char*)) {
#define PS_LOAD(ret, name, ...) \
  gl##name = (PFN_ps_gl##name)getProc("gl" #name); if (!gl##name) return "gl" #name;
  PS_GL_FUNCS(PS_LOAD)
#undef PS_LOAD
  return nullptr;
}
