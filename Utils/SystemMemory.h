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

    // The tightest memory limit of the process's cgroup and of every group
    // above it: a job's limit is often set on a parent group, with
    // memory.max = max on the process's own.
    double limit = 0.;
    const auto apply = [&limit](double l) {
        if (l > 0.)
            limit = limit > 0. ? std::min(limit, l) : l;
    };
    // Each group from `path` (relative to `root`) up to the root, inclusive.
    const auto applyHierarchy = [&](const std::string &root, std::string path, const char *file) {
        while (!path.empty() && path.front() == '/')
            path.erase(path.begin());
        while (true)
        {
            apply(readLimit(root + (path.empty() ? "" : "/" + path) + "/" + file));
            if (path.empty())
                break;
            const auto slash = path.find_last_of('/');
            path = slash == std::string::npos ? std::string() : path.substr(0, slash);
        }
    };
    std::ifstream cgroups("/proc/self/cgroup");
    if (cgroups.is_open())
    {
        std::string line;
        while (std::getline(cgroups, line))
        {
            // cgroup v2: "0::/path"; cgroup v1: "N:<controllers>:/path" with "memory" among them.
            const auto first = line.find(':'), second = line.find(':', first == std::string::npos ? first : first + 1);
            if (first == std::string::npos || second == std::string::npos)
                continue;
            const std::string controllers = line.substr(first + 1, second - first - 1), path = line.substr(second + 1);
            if (line.compare(0, first, "0") == 0 && controllers.empty())
                applyHierarchy("/sys/fs/cgroup", path, "memory.max");
            else if (("," + controllers + ",").find(",memory,") != std::string::npos)
            {
                // The v1 memory controller is mounted at /sys/fs/cgroup/memory,
                // or on some systems at /sys/fs/cgroup itself.
                applyHierarchy("/sys/fs/cgroup/memory", path, "memory.limit_in_bytes");
                applyHierarchy("/sys/fs/cgroup", path, "memory.limit_in_bytes");
            }
        }
    }
    // The roots of every layout, whatever /proc/self/cgroup listed (a cgroup
    // namespace shows the process's group as the root). The tightest limit wins,
    // so a file that does not apply can only be absent, never loosen it.
    apply(readLimit("/sys/fs/cgroup/memory.max"));                   // cgroup v2
    apply(readLimit("/sys/fs/cgroup/memory/memory.limit_in_bytes")); // cgroup v1
    apply(readLimit("/sys/fs/cgroup/memory.limit_in_bytes"));        // cgroup v1 mounted at the root

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
 * memory, at least 1 MiB. `unknownMB` is the budget when the size is unknown. */
inline size_t MachineTrajectoryBudgetMB(size_t unknownMB = 1024)
{
    const double bytes = PhysicalMemoryBytes();
    if (!(bytes > 0.))
        return unknownMB;
    return std::max<size_t>(1, static_cast<size_t>(bytes / 2. / (1024. * 1024.)));
}

} // namespace Utils

#endif
