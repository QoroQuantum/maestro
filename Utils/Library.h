/**
 * @file Library.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The library class.
 *
 * Used to dynamically load a library on linux or windows.
 */

#pragma once

#ifndef _LIBRARY_H
#define _LIBRARY_H

#include <iostream>

#if defined(__linux__) || defined(__APPLE__)

#include <dlfcn.h>

#elif defined(_WIN32)

#include <windows.h>

#endif

namespace Utils {

class Library {
 public:
  Library(const Library &) = delete;
  Library &operator=(const Library &) = delete;
  Library(Library &&) = default;
  Library &operator=(Library &&) = default;

  Library() noexcept {}

  virtual ~Library() {
    if (handle)
#if defined(__linux__) || defined(__APPLE__)
      dlclose(handle);
#elif defined(_WIN32)
      FreeLibrary(handle);
#endif
  }

  virtual bool Init(const char *libName) noexcept {
#if defined(__linux__) || defined(__APPLE__)
    handle = dlopen(libName, RTLD_NOW);

    if (handle == nullptr) {
      const char *dlsym_error = dlerror();
      if (!mute && dlsym_error)
        std::cerr << "Library: Unable to load library, error: " << dlsym_error
                  << std::endl;

      return false;
    }
#elif defined(_WIN32)
    handle = LoadLibraryA(libName);
    if (handle == nullptr) {
      const DWORD error = GetLastError();
      if (!mute)
        std::cerr << "Library: Unable to load library, error code: " << error
                  << std::endl;
      return false;
    }
#endif

    return true;
  }

  void *GetFunction(const char *funcName) noexcept {
#if defined(__linux__) || defined(__APPLE__)
    return dlsym(handle, funcName);
#elif defined(_WIN32)
    return GetProcAddress(handle, funcName);
#endif
  }

  const void *GetHandle() const noexcept { return handle; }

  bool IsMuted() const noexcept { return mute; }

  void SetMute(bool m) noexcept { mute = m; }

 private:
#if defined(__linux__) || defined(__APPLE__)
  void *handle = nullptr;
#elif defined(_WIN32)
  HINSTANCE handle = nullptr;
#endif
  bool mute = false;
};

}  // namespace Utils

#endif
