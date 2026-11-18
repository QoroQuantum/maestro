#pragma once

#ifndef _UTILS_SYSTEM_MEMORY_H_
#define _UTILS_SYSTEM_MEMORY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

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

namespace SystemMemoryDetail
{

/** The files that hold a memory limit for the process: that of its cgroup and
 * of every group above it, as far as the cgroup filesystem is mounted (a job's
 * limit is often set on a parent group, with memory.max = max on the process's
 * own). `cgroups` is the text of /proc/self/cgroup, `mountinfo` that of
 * /proc/self/mountinfo.
 *
 * A group's directory is its path relative to the root of the mount, under the
 * mount point: with /tenant mounted at /sys/fs/cgroup, the group /tenant/job is
 * /sys/fs/cgroup/job, not /sys/fs/cgroup/tenant/job. Without mount information
 * the conventional mount points are assumed. The conventional root files are
 * always listed: the tightest limit wins, so a file that does not apply can only
 * be absent, never loosen it. */
inline std::vector<std::string> CgroupLimitFiles(std::istream &cgroups, std::istream &mountinfo)
{
    std::vector<std::string> files;
    // The process's group in the unified hierarchy (v2, "0::/path") and in the
    // v1 hierarchy that has the memory controller ("N:...,memory,...:/path").
    std::string unified, memory;
    bool hasUnified = false, hasMemory = false;
    for (std::string line; std::getline(cgroups, line);)
    {
        const auto first = line.find(':'), second = line.find(':', first == std::string::npos ? first : first + 1);
        if (first == std::string::npos || second == std::string::npos)
            continue;
        const std::string controllers = line.substr(first + 1, second - first - 1), path = line.substr(second + 1);
        if (line.compare(0, first, "0") == 0 && controllers.empty())
        {
            unified = path;
            hasUnified = true;
        }
        else if (("," + controllers + ",").find(",memory,") != std::string::npos)
        {
            memory = path;
            hasMemory = true;
        }
    }
    // Each group from `relative` (below the directory `mount`) up to the mount itself.
    const auto addHierarchy = [&files](const std::string &mount, std::string relative, const char *file) {
        while (!relative.empty() && relative.front() == '/')
            relative.erase(relative.begin());
        while (!relative.empty() && relative.back() == '/')
            relative.pop_back();
        while (true)
        {
            files.push_back(mount + (relative.empty() ? "" : "/" + relative) + "/" + file);
            if (relative.empty())
                break;
            const auto slash = relative.find_last_of('/');
            relative = slash == std::string::npos ? std::string() : relative.substr(0, slash);
        }
    };
    // The group's path below the root of a mount, or nothing when the group is
    // not below it (a cgroup namespace): then only the mount itself is read.
    const auto below = [](const std::string &root, const std::string &group) -> std::string {
        if (root == "/" || root.empty())
            return group;
        if (group.compare(0, root.size(), root) == 0 && (group.size() == root.size() || group[root.size()] == '/'))
            return group.substr(root.size());
        return {};
    };
    bool mounted = false;
    // "ID parent major:minor root mount-point options... - fstype source super-options"
    for (std::string line; std::getline(mountinfo, line);)
    {
        const auto separator = line.find(" - ");
        if (separator == std::string::npos)
            continue;
        std::istringstream head(line.substr(0, separator)), tail(line.substr(separator + 3));
        std::string id, parent, device, root, mount, type, source, options;
        if (!(head >> id >> parent >> device >> root >> mount) || !(tail >> type))
            continue;
        tail >> source >> options;
        if (type == "cgroup2")
        {
            addHierarchy(mount, hasUnified ? below(root, unified) : std::string(), "memory.max");
            mounted = true;
        }
        else if (type == "cgroup" && ("," + options + ",").find(",memory,") != std::string::npos)
        {
            addHierarchy(mount, hasMemory ? below(root, memory) : std::string(), "memory.limit_in_bytes");
            mounted = true;
        }
    }
    if (!mounted)
    {
        if (hasUnified)
            addHierarchy("/sys/fs/cgroup", unified, "memory.max");
        if (hasMemory)
        {
            // The v1 memory controller is mounted at /sys/fs/cgroup/memory, or on some systems at /sys/fs/cgroup itself.
            addHierarchy("/sys/fs/cgroup/memory", memory, "memory.limit_in_bytes");
            addHierarchy("/sys/fs/cgroup", memory, "memory.limit_in_bytes");
        }
    }
    files.push_back("/sys/fs/cgroup/memory.max");                   // cgroup v2
    files.push_back("/sys/fs/cgroup/memory/memory.limit_in_bytes"); // cgroup v1
    files.push_back("/sys/fs/cgroup/memory.limit_in_bytes");        // cgroup v1 mounted at the root
    return files;
}

} // namespace SystemMemoryDetail

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

    // The tightest memory limit of the process's cgroup and of every group above it.
    double limit = 0.;
    std::ifstream cgroups("/proc/self/cgroup"), mountinfo("/proc/self/mountinfo");
    for (const auto &file : SystemMemoryDetail::CgroupLimitFiles(cgroups, mountinfo))
        if (const double l = readLimit(file); l > 0.)
            limit = limit > 0. ? std::min(limit, l) : l;

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
