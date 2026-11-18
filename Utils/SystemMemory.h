#pragma once

#ifndef _UTILS_SYSTEM_MEMORY_H_
#define _UTILS_SYSTEM_MEMORY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#undef min
#undef max
#endif

namespace Utils
{

/** Physical memory available to the process in bytes, or 0 if unknown. On
 * Linux a cgroup memory limit (v1 or v2) lowers it. */
inline double PhysicalMemoryBytes()
{
#if defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) == 0)
        return static_cast<double>(bytes);
#elif defined(__linux__)
    const long pages = sysconf(_SC_PHYS_PAGES), pageSize = sysconf(_SC_PAGE_SIZE);
    double mem = 0.;
    if (pages > 0 && pageSize > 0)
        mem = static_cast<double>(pages) * static_cast<double>(pageSize);

    auto readLimit = [](const std::string &path) -> double {
        std::ifstream f(path);
        if (!f.is_open())
            return 0.;
        std::string token;
        if (!(f >> token) || token == "max")
            return 0.;
        try
        {
            const auto limit = std::stoull(token);
            // Treat limits >= LONG_MAX (typical cgroup v1 unlimited indicator) as unlimited.
            if (limit > 0 && limit < 9223372036854771712ULL)
                return static_cast<double>(limit);
        }
        catch (...)
        {
        }
        return 0.;
    };

    double limit = readLimit("/sys/fs/cgroup/memory.max"); // cgroup v2
    if (limit <= 0.)
        limit = readLimit("/sys/fs/cgroup/memory/memory.limit_in_bytes"); // cgroup v1
    if (limit <= 0.)
        limit = readLimit("/sys/fs/cgroup/memory.limit_in_bytes");

    std::ifstream cgroups("/proc/self/cgroup");
    if (cgroups.is_open())
    {
        std::string line;
        while (std::getline(cgroups, line))
        {
            if (line.rfind("0::", 0) == 0)
            {
                std::string sub = line.substr(3);
                while (!sub.empty() && sub.front() == '/')
                    sub.erase(sub.begin());
                if (!sub.empty())
                {
                    double l = readLimit("/sys/fs/cgroup/" + sub + "/memory.max");
                    if (l > 0.)
                        limit = (limit > 0.) ? std::min(limit, l) : l;
                }
            }
        }
    }

    if (limit > 0.)
        mem = (mem > 0.) ? std::min(mem, limit) : limit;

    return mem;
#elif defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return static_cast<double>(status.ullTotalPhys);
#endif
    return 0.;
}

/** The default trajectory memory budget in MiB for backends that size their
 * own parallel shots by the machine's memory, as Qiskit Aer does (its
 * max_memory_mb defaults to all of it, one state per shot). Maestro counts two
 * or three copies of a state per realization worker, so it takes half the
 * memory. Never below `floorMB`, the budget used when the size is unknown. */
inline size_t MachineTrajectoryBudgetMB(size_t floorMB = 1024)
{
    const double bytes = PhysicalMemoryBytes();
    if (!(bytes > 0.))
        return floorMB;
    return std::max(floorMB, static_cast<size_t>(bytes / 2. / (1024. * 1024.)));
}

} // namespace Utils

#endif
