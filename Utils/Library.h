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

#include <initializer_list>
#include <iostream>

#if defined(__linux__) || defined(__APPLE__)

#include <dlfcn.h>

#elif defined(_WIN32)

#include <windows.h>
#undef min
#undef max

#endif

namespace Utils
{

/**
 * @brief Keeps the OpenMP runtime loaded until the process exits.
 *
 * The runtime's worker threads outlive the parallel regions that started them
 * and keep running in the runtime's own code. On Windows the runtime is
 * unloaded with the last module that uses it, be it a plugin or this library
 * when a host frees it, and the workers then fault in code that is gone. A
 * pinned module is never unloaded, so the workers stay valid; the modules
 * that used the runtime can still be unloaded.
 *
 * Does nothing where no OpenMP runtime is loaded, and on other systems.
 */
inline void KeepOpenMPRuntimeLoaded() noexcept
{
#if defined(_WIN32)
    for (const wchar_t *runtime : {L"vcomp140.dll", L"vcomp140d.dll", L"libomp140.x86_64.dll", L"libomp140d.x86_64.dll", L"libiomp5md.dll"})
    {
        HMODULE module = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, runtime, &module);
    }
#endif
}

class Library
{
  public:
    Library(const Library &) = delete;
    Library &operator=(const Library &) = delete;
    Library(Library &&) = delete;
    Library &operator=(Library &&) = delete;

    Library() noexcept
    {
    }

    virtual ~Library()
    {
        // A library is left alone while the process exits, see ProcessIsExiting().
        if (handle && !ProcessIsExiting())
#if defined(__linux__) || defined(__APPLE__)
            dlclose(handle);
#elif defined(_WIN32)
            FreeLibrary(handle);
#endif
    }

    virtual bool Init(const char *libName) noexcept
    {
        if (handle)
            return true;
#if defined(__linux__) || defined(__APPLE__)
        handle = dlopen(libName, RTLD_NOW);

        if (handle == nullptr)
        {
            const char *dlsym_error = dlerror();
            if (!mute && dlsym_error)
                std::cerr << "Library: Unable to load library, error: " << dlsym_error << std::endl;

            return false;
        }
#elif defined(_WIN32)
        handle = LoadLibraryA(libName);
        if (handle == nullptr)
        {
            const DWORD error = GetLastError();
            if (!mute)
                std::cerr << "Library: Unable to load library, error code: " << error << std::endl;
            return false;
        }
        // Resolved here so the destructor does not query the loader at exit.
        if (const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"))
            shutdownInProgress = reinterpret_cast<ShutdownQuery>(GetProcAddress(ntdll, "RtlDllShutdownInProgress"));
        // The library may have brought the OpenMP runtime with it.
        KeepOpenMPRuntimeLoaded();
#endif

        return true;
    }

    void *GetFunction(const char *funcName) noexcept
    {
#if defined(__linux__) || defined(__APPLE__)
        return dlsym(handle, funcName);
#elif defined(_WIN32)
        return GetProcAddress(handle, funcName);
#endif
    }

    const void *GetHandle() const noexcept
    {
        return handle;
    }

    bool IsMuted() const noexcept
    {
        return mute;
    }

    void SetMute(bool m) noexcept
    {
        mute = m;
    }

  protected:
    /**
     * @brief Whether the process is exiting and the loaded library is already
     * torn down.
     *
     * When a Windows process exits, DLLs are detached in the reverse order of
     * their load, each destroying its static objects. A library loaded at run
     * time is therefore detached before the DLL holding the object that loaded
     * it, so a destructor that runs with that DLL's static objects must not
     * call into the library any more: its state is gone. The operating system
     * releases everything the library held.
     *
     * False when static objects are destroyed before the DLLs are detached
     * (an executable's own, or when only this DLL is unloaded), and on other
     * systems.
     */
    bool ProcessIsExiting() const noexcept
    {
#if defined(_WIN32)
        return shutdownInProgress && shutdownInProgress() != FALSE;
#else
        return false;
#endif
    }

  private:
#if defined(__linux__) || defined(__APPLE__)
    void *handle = nullptr;
#elif defined(_WIN32)
    using ShutdownQuery = BOOLEAN(NTAPI *)();

    HINSTANCE handle = nullptr;
    ShutdownQuery shutdownInProgress = nullptr; /**< RtlDllShutdownInProgress */
#endif
    bool mute = false;
};

} // namespace Utils

#endif
