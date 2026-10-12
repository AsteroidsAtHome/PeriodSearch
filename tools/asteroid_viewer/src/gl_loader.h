// Minimal OpenGL 3.3 core loader (no GLEW/glad needed). Functions are loaded
// through glfwGetProcAddress after a context has been made current.
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_WIN32) && !defined(__MINGW64__) && !defined(_WIN64)
#define PS_GLAPI __stdcall
#elif defined(_WIN32)
#define PS_GLAPI __stdcall
#else
#define PS_GLAPI
#endif

typedef unsigned int GLenum; typedef unsigned char GLboolean; typedef unsigned int GLbitfield;
typedef int GLint; typedef int GLsizei; typedef unsigned int GLuint; typedef float GLfloat;
typedef char GLchar; typedef std::ptrdiff_t GLsizeiptr; typedef void GLvoid;

#define GL_FALSE 0
#define GL_TRUE 1
#define GL_FLOAT 0x1406
#define GL_TRIANGLES 0x0004
#define GL_LINES 0x0001
#define GL_DEPTH_TEST 0x0B71
#define GL_CULL_FACE 0x0B44
#define GL_SCISSOR_TEST 0x0C11
#define GL_BACK 0x0405
#define GL_LESS 0x0201
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_ARRAY_BUFFER 0x8892
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_MULTISAMPLE 0x809D
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401

#define PS_GL_FUNCS(X) \
  X(void, Viewport, GLint, GLint, GLsizei, GLsizei) \
  X(void, Scissor, GLint, GLint, GLsizei, GLsizei) \
  X(void, ClearColor, GLfloat, GLfloat, GLfloat, GLfloat) \
  X(void, Clear, GLbitfield) \
  X(void, Enable, GLenum) \
  X(void, Disable, GLenum) \
  X(void, DepthFunc, GLenum) \
  X(void, CullFace, GLenum) \
  X(void, ReadPixels, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*) \
  X(GLuint, CreateShader, GLenum) \
  X(void, ShaderSource, GLuint, GLsizei, const GLchar* const*, const GLint*) \
  X(void, CompileShader, GLuint) \
  X(void, GetShaderiv, GLuint, GLenum, GLint*) \
  X(void, GetShaderInfoLog, GLuint, GLsizei, GLsizei*, GLchar*) \
  X(GLuint, CreateProgram, void) \
  X(void, AttachShader, GLuint, GLuint) \
  X(void, LinkProgram, GLuint) \
  X(void, GetProgramiv, GLuint, GLenum, GLint*) \
  X(void, GetProgramInfoLog, GLuint, GLsizei, GLsizei*, GLchar*) \
  X(void, DeleteShader, GLuint) \
  X(void, DeleteProgram, GLuint) \
  X(void, UseProgram, GLuint) \
  X(GLint, GetUniformLocation, GLuint, const GLchar*) \
  X(void, UniformMatrix4fv, GLint, GLsizei, GLboolean, const GLfloat*) \
  X(void, Uniform3fv, GLint, GLsizei, const GLfloat*) \
  X(void, Uniform1f, GLint, GLfloat) \
  X(void, Uniform1i, GLint, GLint) \
  X(void, GenVertexArrays, GLsizei, GLuint*) \
  X(void, DeleteVertexArrays, GLsizei, const GLuint*) \
  X(void, BindVertexArray, GLuint) \
  X(void, GenBuffers, GLsizei, GLuint*) \
  X(void, DeleteBuffers, GLsizei, const GLuint*) \
  X(void, BindBuffer, GLenum, GLuint) \
  X(void, BufferData, GLenum, GLsizeiptr, const void*, GLenum) \
  X(void, VertexAttribPointer, GLuint, GLint, GLenum, GLboolean, GLsizei, const void*) \
  X(void, EnableVertexAttribArray, GLuint) \
  X(void, DrawArrays, GLenum, GLint, GLsizei)

#define PS_DECLARE(ret, name, ...) typedef ret (PS_GLAPI *PFN_ps_gl##name)(__VA_ARGS__); extern PFN_ps_gl##name gl##name;
PS_GL_FUNCS(PS_DECLARE)
#undef PS_DECLARE

// returns name of the first function that failed to load, or nullptr on success
const char* psLoadGL(void* (*getProc)(const char*));
