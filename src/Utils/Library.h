/**
 * @file Library.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The library class.
 *
 * Used to dynamically load a library on linux or windows. This file is part of
 * Maestro's plugin/dynamic loading interface through which independent plugin
 * modules (e.g., GPU simulator libraries) are loaded at runtime.
 *
 * @section LICENSE
 *
 * Copyright (C) 2025 Qoro Quantum Ltd
 *
 * This file is part of Maestro.
 *
 * Maestro is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * Maestro is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Maestro. If not, see <https://www.gnu.org/licenses/>.
 *
 * As an additional permission under GNU GPL version 3 section 7, independent
 * plugin modules that communicate with Maestro solely through the C-language
 * dynamic loading interface defined in this file (dlopen/dlsym) may be
 * distributed under terms of your choice, without requiring those modules to be
 * licensed under the GPL. See the "Maestro Plugin Linking Exception" in the
 * LICENSE file for full details.
 */

#pragma once

#ifndef _LIBRARY_H
#define _LIBRARY_H

#include <iostream>

#if defined(__linux__) || defined(__APPLE__)

#include <dlfcn.h>

#elif defined(_WIN32)

#include <windows.h>
#undef min
#undef max

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
