#pragma once

#include "../maestrolib/Interface.h"
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>

// Check the actual dependency, even when an installed Maestro precedes RUNPATH.
inline void CheckLoadedMaestro()
{
    Dl_info loaded{};
    if (!dladdr(reinterpret_cast<void *>(&GetMaestroObject), &loaded) || !loaded.dli_fname ||
        !std::filesystem::equivalent(loaded.dli_fname, MAESTRO_TEST_LIBRARY_PATH))
        throw std::runtime_error("Pauli test loaded a different Maestro library from the build under test");
}
