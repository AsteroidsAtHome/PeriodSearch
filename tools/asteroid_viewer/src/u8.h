// UTF-8 everywhere. libstdc++'s std::filesystem (MinGW) treats narrow strings as UTF-8, while
// the narrow Windows APIs (getenv, argv, fopen, ...) use the ANSI code page. A user name with
// e.g. Czech letters in %APPDATA% then makes std::filesystem throw. These helpers obtain and use
// strings as UTF-8 on Windows (via the wide APIs); elsewhere they are plain pass-throughs.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#ifdef _WIN32
inline std::wstring u8widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
  return w;
}
inline std::string u8narrow(const std::wstring& w) {
  if (w.empty()) return std::string();
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr, nullptr);
  return s;
}
#endif

// environment variable as UTF-8 ("" if not set)
inline std::string envUtf8(const char* name) {
#ifdef _WIN32
  const wchar_t* v = _wgetenv(u8widen(name).c_str());
  return v ? u8narrow(v) : std::string();
#else
  const char* v = std::getenv(name);
  return v ? v : "";
#endif
}

// UTF-8 string -> path (never throws for valid UTF-8)
inline std::filesystem::path u8p(const std::string& s) {
#ifdef _WIN32
  return std::filesystem::path(u8widen(s));
#else
  return std::filesystem::path(s);
#endif
}

// path -> UTF-8 string (never throws)
inline std::string u8s(const std::filesystem::path& p) {
#ifdef _WIN32
  return u8narrow(p.wstring());
#else
  return p.string();
#endif
}

inline FILE* fopenUtf8(const std::string& path, const char* mode) {
#ifdef _WIN32
  return _wfopen(u8widen(path).c_str(), u8widen(mode).c_str());
#else
  return std::fopen(path.c_str(), mode);
#endif
}

// command line arguments as UTF-8
inline std::vector<std::string> argsUtf8(int argc, char** argv) {
  std::vector<std::string> a;
#ifdef _WIN32
  int n = 0;
  LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
  if (w) {
    for (int i = 0; i < n; i++) a.push_back(u8narrow(w[i]));
    LocalFree(w);
    return a;
  }
#endif
  for (int i = 0; i < argc; i++) a.push_back(argv[i]);
  return a;
}
