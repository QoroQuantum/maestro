#include "../Utils/Threads/RealizationWorkers.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef MAESTRO_REALIZATION_NOISE_TESTS
#include "../Circuit/Factory.h"
#include "../Execution/NoisyEstimation.h"
#include "../Execution/PrefixCheckpoint.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Noise/NoiseAdd.h"
#include <cmath>
#include <set>
#endif

namespace
{
void Require(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

struct Payload
{
    explicit Payload(uint32_t value) : value(value)
    {
        const size_t count = ++live;
        size_t previous = maximum.load();
        while (previous < count && !maximum.compare_exchange_weak(previous, count))
        {
        }
    }

    ~Payload()
    {
        --live;
    }

    uint32_t value;
    inline static std::atomic<size_t> live{0}, maximum{0};
};

void OrderedPreparationAndPrivateWorkers()
{
    const size_t total = 127;
    const size_t workers = Utils::RealizationWorkerCount(total, 4);
    Payload::maximum = 0;
    std::mt19937 expectedRng(7), actualRng(7);
    std::vector<uint32_t> expected(total), actual(total);
    for (auto &value : expected)
        value = expectedRng();
    std::vector<std::atomic<bool>> occupied(workers);
    for (auto &value : occupied)
        value = false;
    size_t prepared = 0;
    Utils::RunRealizations(
        total, workers,
        [&](size_t index) {
            Require(index == prepared++, "preparation order changed");
            return std::make_unique<Payload>(actualRng());
        },
        [&](size_t worker, size_t index, std::unique_ptr<Payload> payload) {
            Require(!occupied[worker].exchange(true), "worker state used concurrently");
            // Give the producer an opportunity to fill its bounded queue.
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            actual[index] = payload->value;
            occupied[worker] = false;
#ifdef _OPENMP
            if (workers > 1)
                Require(omp_get_max_threads() == 1, "worker exceeded its internal thread budget");
#endif
        });
    Require(actual == expected, "parallel execution changed the serial noise stream");
    Require(prepared == total, "realizations were skipped");
    Require(Payload::live == 0, "payload remained alive after execution");
    Require(Payload::maximum <= 2 * workers + 1, "producer exceeded its bounded payload budget");
}

void FailuresJoinAndRelease()
{
    for (const bool prepareFailure : {false, true})
    {
#ifdef _OPENMP
        const int previousThreads = omp_get_max_threads();
#endif
        std::atomic<size_t> active{0};
        bool rejected = false;
        try
        {
            Utils::RunRealizations(
                64, 4,
                [&](size_t index) {
                    if (prepareFailure && index == 7)
                        throw std::runtime_error("prepare failed");
                    return std::make_unique<Payload>(static_cast<uint32_t>(index));
                },
                [&](size_t, size_t index, std::unique_ptr<Payload>) {
                    struct Active
                    {
                        std::atomic<size_t> &value;
                        Active(std::atomic<size_t> &value) : value(value)
                        {
                            ++value;
                        }
                        ~Active()
                        {
                            --value;
                        }
                    } guard(active);
                    if (!prepareFailure && index == 3)
                        throw std::runtime_error("execute failed");
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                });
        }
        catch (const std::runtime_error &error)
        {
            rejected = std::string(error.what()) == (prepareFailure ? "prepare failed" : "execute failed");
        }
        Require(rejected, "realization exception was not propagated");
        Require(active == 0, "execution returned before every worker joined");
        Require(Payload::live == 0, "failed execution retained queued payloads");
#ifdef _OPENMP
        Require(omp_get_max_threads() == previousThreads, "failed execution changed the caller's OpenMP limit");
#endif
    }
}

void BudgetsAndEmptyWork()
{
    Require(Utils::RealizationWorkerCount(0, 4) == 0, "empty work acquired workers");
    Require(Utils::RealizationWorkerCount(9, 1) == 1, "requested serial execution was ignored");
    Require(Utils::RealizationWorkerCount(2, 8) <= 2, "more workers than realizations");
    Require(Utils::LimitRealizationWorkersByMemory(8, size_t{1} << 20, 10, 2, 2) == 4, "checkpoint memory was not reserved");
    Require(Utils::LimitRealizationWorkersByMemory(8, size_t{1} << 20, 1, 2, 2) == 1, "memory bound lost the serial fallback");
#ifdef MAESTRO_REALIZATION_NOISE_TESTS
    MaestroExecution::SimulatorConfig config;
    config.simulation_type = Simulators::SimulationType::kStatevector;
    config.max_simulators = 8;
    config.trajectory_max_memory_mb = 80;
    // Twenty qubits need 16 MiB: five register-sized slots permit two direct
    // workers, but only one network worker with its additional prepared source.
    const auto cpuWorkers = Utils::RealizationWorkerCount(8, 8);
    Require(MaestroExecution::SamplingWorkers(config, 8, 20) == std::min<size_t>(2, cpuWorkers), "direct-session memory accounting changed");
    Require(MaestroExecution::SamplingWorkers(config, 8, 20, 3) == 1, "network trajectory's prepared source was not budgeted");
    config.fixed_backend = true;
    Require(MaestroExecution::NetworkRealizationCopies(config, false) == 2 && MaestroExecution::NetworkRealizationCopies(config, true) == 3,
            "dense realization copy accounting changed");
    for (const auto method :
         {Simulators::SimulationType::kStabilizer, Simulators::SimulationType::kMatrixProductState, Simulators::SimulationType::kMatrixProductOperator})
    {
        config.simulation_type = method;
        Require(MaestroExecution::NetworkRealizationCopies(config, false) == 2 && MaestroExecution::NetworkRealizationCopies(config, true) == 4,
                "tensor/tableau realization omitted its cloned checkpoint");
    }
    config.simulation_type = Simulators::SimulationType::kStatevector;
    config.fixed_backend = false;
    config.optimization_candidates = {{Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector},
                                      {Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState}};
    Require(MaestroExecution::NetworkRealizationCopies(config, true) == 4, "candidate realization omitted the largest copy requirement");
    Require(MaestroExecution::NetworkRealizationCopies(config, false) == 4, "one-shot candidate realization omitted its prepared checkpoint");
    config.fixed_backend = true;
    Require(MaestroExecution::NetworkRealizationCopies(config, true) == 3, "fixed backend counted an unselectable tensor candidate");
    config.fixed_backend = false;
    config.simulation_type = Simulators::SimulationType::kMatrixProductState;
    config.optimization_candidates = {{Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector}};
    Require(MaestroExecution::NetworkRealizationCopies(config, true) == 3, "candidate list did not replace the requested tensor method");
    config.max_simulators.reset();
    config.native_options["max_simulators"] = "1";
    Require(MaestroExecution::EffectiveMaxSimulators(config) == 1 && MaestroExecution::SamplingWorkers(config, 8, 2) == 1,
            "native worker limit did not constrain realization scheduling");
    config.max_simulators = 3;
    Require(MaestroExecution::EffectiveMaxSimulators(config) == 3, "typed worker limit did not override the native option");
    // Unset, the trajectory budget is 1 GiB, and on the CPU Aer statevector and
    // MPS half the machine's memory, as Aer fills it with parallel shots.
    MaestroExecution::SimulatorConfig budget;
    budget.simulation_type = Simulators::SimulationType::kStatevector;
    budget.fixed_backend = true;
    Require(MaestroExecution::RealizationBudgetMB(budget) == 1024, "QCSim's default trajectory budget changed");
#ifndef NO_QISKIT_AER
    budget.simulator_type = Simulators::SimulatorType::kQiskitAer;
    const size_t machine = Utils::MachineTrajectoryBudgetMB();
    Require(machine >= 1 && MaestroExecution::RealizationBudgetMB(budget) == machine, "Aer's statevector did not take the machine-sized budget");
    budget.simulation_type = Simulators::SimulationType::kMatrixProductState;
    Require(MaestroExecution::RealizationBudgetMB(budget) == machine, "Aer's MPS did not take the machine-sized budget");
    budget.simulation_type = Simulators::SimulationType::kDensityMatrix;
    Require(MaestroExecution::RealizationBudgetMB(budget) == 1024, "Aer's density matrix took the trajectory budget");
    budget.simulation_type = Simulators::SimulationType::kStatevector;
    budget.native_options["trajectory_max_memory_mb"] = "300";
    Require(MaestroExecution::RealizationBudgetMB(budget) == 300, "a native trajectory budget lost to Aer's default");
    budget.trajectory_max_memory_mb = 200;
    Require(MaestroExecution::RealizationBudgetMB(budget) == 200, "a typed trajectory budget lost to Aer's default");
    // A group's limit file is found below the mount's root: with /tenant mounted at /sys/fs/cgroup, the
    // group /tenant/job is /sys/fs/cgroup/job. Its parents up to the mount are read too.
    {
        std::istringstream cgroups("0::/tenant/job/step\n"), mounts("30 25 0:26 /tenant /sys/fs/cgroup rw,nosuid - cgroup2 cgroup2 rw\n");
        const auto files = Utils::SystemMemoryDetail::CgroupLimitFiles(cgroups, mounts);
        const auto has = [&](const char *file) { return std::find(files.begin(), files.end(), file) != files.end(); };
        Require(has("/sys/fs/cgroup/job/step/memory.max") && has("/sys/fs/cgroup/job/memory.max") && has("/sys/fs/cgroup/memory.max"),
                "a cgroup below the mount's root was not resolved against it");
        Require(!has("/sys/fs/cgroup/tenant/job/step/memory.max"), "a cgroup path was appended to the mount point unchanged");
        // cgroup v1: the hierarchy with the memory controller, wherever it is mounted.
        std::istringstream v1("5:cpu,cpuacct:/a\n9:memory:/docker/abc\n"),
            v1Mounts("40 30 0:35 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n41 30 0:36 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n");
        const auto v1Files = Utils::SystemMemoryDetail::CgroupLimitFiles(v1, v1Mounts);
        const auto v1Has = [&](const char *file) { return std::find(v1Files.begin(), v1Files.end(), file) != v1Files.end(); };
        Require(v1Has("/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes") && v1Has("/sys/fs/cgroup/memory/docker/memory.limit_in_bytes") &&
                    v1Has("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                "the cgroup v1 memory hierarchy was not walked");
        Require(!v1Has("/sys/fs/cgroup/cpu/a/memory.limit_in_bytes"), "a cgroup v1 hierarchy without the memory controller was read");
        // Without mount information the conventional mount points are assumed.
        std::istringstream plain("0::/user.slice/job\n"), none("");
        const auto plainFiles = Utils::SystemMemoryDetail::CgroupLimitFiles(plain, none);
        Require(std::find(plainFiles.begin(), plainFiles.end(), "/sys/fs/cgroup/user.slice/job/memory.max") != plainFiles.end(),
                "the conventional cgroup mount point was not assumed without mount information");
    }
    {
        const auto listed = [](const std::vector<std::string> &files, const char *file) { return std::find(files.begin(), files.end(), file) != files.end(); };
        // A mount whose root does not hold the process's group shows another group's limits: it is
        // skipped, and so are the conventional paths once the mounts are known.
        std::istringstream cgroups("0::/a/job\n"), mounts("30 25 0:26 /a /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n"
                                                          "31 25 0:27 /b /mnt/other rw - cgroup2 cgroup2 rw\n"
                                                          "32 25 0:28 /../outside /mnt/outside rw - cgroup2 cgroup2 rw\n");
        const auto files = Utils::SystemMemoryDetail::CgroupLimitFiles(cgroups, mounts);
        Require(listed(files, "/sys/fs/cgroup/job/memory.max") && listed(files, "/sys/fs/cgroup/memory.max"), "the process's own cgroup mount was not read");
        Require(!listed(files, "/mnt/other/memory.max") && !listed(files, "/mnt/outside/memory.max"), "another group's cgroup mount was read");
        Require(files.size() == 2, "paths outside the process's cgroup mount were read");
        // In a cgroup namespace a mount rooted at "/.." (or "/../..") shows an ancestor of the
        // namespace's root: its own limit applies. "/../other" is a sibling of the root.
        std::istringstream inherited("0::/\n"), inheritedMounts("30 25 0:26 /.. /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n"
                                                                "31 25 0:27 /../.. /mnt/grandparent rw - cgroup2 cgroup2 rw\n"
                                                                "32 25 0:28 /../other /mnt/sibling rw - cgroup2 cgroup2 rw\n"
                                                                "33 25 0:29 /..other /mnt/named rw - cgroup2 cgroup2 rw\n");
        const auto inheritedFiles = Utils::SystemMemoryDetail::CgroupLimitFiles(inherited, inheritedMounts);
        Require(listed(inheritedFiles, "/sys/fs/cgroup/memory.max") && listed(inheritedFiles, "/mnt/grandparent/memory.max"),
                "the limit of an ancestor of the cgroup namespace's root was not read");
        Require(!listed(inheritedFiles, "/mnt/sibling/memory.max") && !listed(inheritedFiles, "/mnt/named/memory.max") && inheritedFiles.size() == 2,
                "a sibling of the cgroup namespace's root was read");
        // Paths are matched by whole components first, whatever they are called: a process outside its
        // namespace's root under a mount of that group, and a group whose name starts with "..".
        for (const auto &[group, root] : std::vector<std::pair<std::string, std::string>>{{"/../other/job", "/../other"}, {"/..tenant/job", "/..tenant"}})
        {
            std::istringstream outside("0::" + group + "\n"), outsideMounts("30 25 0:26 " + root + " /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n");
            const auto outsideFiles = Utils::SystemMemoryDetail::CgroupLimitFiles(outside, outsideMounts);
            Require(outsideFiles.size() == 2 && listed(outsideFiles, "/sys/fs/cgroup/job/memory.max") && listed(outsideFiles, "/sys/fs/cgroup/memory.max"),
                    "a cgroup below a mount root starting with /.. was not matched by components");
        }
        // A sibling subtree reached through the parent: the process's path below the mount is known.
        std::istringstream through("0::/../other/job\n"), throughMounts("30 25 0:26 /.. /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n");
        const auto throughFiles = Utils::SystemMemoryDetail::CgroupLimitFiles(through, throughMounts);
        Require(throughFiles.size() == 3 && listed(throughFiles, "/sys/fs/cgroup/other/job/memory.max") && listed(throughFiles, "/sys/fs/cgroup/memory.max"),
                "a cgroup below the parent of the namespace's root was not walked");
        // A process moved above a mount's root is not below it, though its path starts with the root:
        // /../../other/job under a mount rooted at /.. (and under the namespace's own root). Only the
        // mount that holds it is read, for cgroup v2 and v1.
        for (const bool v2 : {true, false})
        {
            const std::string type = v2 ? "cgroup2 cgroup2 rw" : "cgroup cgroup rw,memory", file = v2 ? "memory.max" : "memory.limit_in_bytes";
            std::istringstream moved(std::string(v2 ? "0::" : "9:memory:") + "/../../other/job\n"),
                movedMounts("30 25 0:26 /.. /mnt/parent rw - " + type + "\n31 25 0:27 / /mnt/root rw - " + type + "\n32 25 0:28 /../../other /mnt/job rw - " +
                            type + "\n");
            const auto movedFiles = Utils::SystemMemoryDetail::CgroupLimitFiles(moved, movedMounts);
            Require(movedFiles.size() == 2 && listed(movedFiles, ("/mnt/job/job/" + file).c_str()) && listed(movedFiles, ("/mnt/job/" + file).c_str()),
                    "a mount the process has moved out of was read for its memory limit");
        }
        // The same without mount information: nothing is assumed for a group that leaves its
        // hierarchy's root, neither paths that climb out of the conventional mount point nor that
        // mount point's own limit.
        for (const bool v2 : {true, false})
        {
            std::istringstream moved(std::string(v2 ? "0::" : "9:memory:") + "/../../other/job\n"), none("");
            for (const auto &file : Utils::SystemMemoryDetail::CgroupLimitFiles(moved, none))
            {
                Require(file.find("..") == std::string::npos, "a cgroup limit was looked up outside the conventional mount point");
                Require(file != (v2 ? "/sys/fs/cgroup/memory.max" : "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                        "the limit of a hierarchy root the process has left was read");
            }
        }
        // mountinfo escapes a space in a path as \040 (and a backslash as \134).
        std::istringstream spaced("0::/job\n"), spacedMounts("30 25 0:26 / /mnt/my\\040cgroup\\134x rw - cgroup2 cgroup2 rw\n");
        const auto spacedFiles = Utils::SystemMemoryDetail::CgroupLimitFiles(spaced, spacedMounts);
        Require(listed(spacedFiles, "/mnt/my cgroup\\x/job/memory.max"), "an escaped mount point was not decoded");
        // A cgroup namespace: the process's group is the root of its mount.
        std::istringstream rooted("0::/\n"), rootedMounts("30 25 0:26 / /sys/fs/cgroup ro - cgroup2 cgroup rw\n");
        const auto rootedFiles = Utils::SystemMemoryDetail::CgroupLimitFiles(rooted, rootedMounts);
        Require(rootedFiles.size() == 1 && rootedFiles[0] == "/sys/fs/cgroup/memory.max", "a namespaced cgroup root was not read");
    }
    // 22 qubits on Aer: 1 GiB held them to five workers; with 16 GiB the CPUs decide.
    budget.trajectory_max_memory_mb.reset();
    budget.native_options.clear();
    budget.max_simulators = 16;
    if (machine >= 16 * 1024)
        Require(MaestroExecution::SamplingWorkers(budget, 64, 22) == Utils::RealizationWorkerCount(64, 16),
                "Aer's 22-qubit trajectories were still held by a 1 GiB budget");
#endif
#endif
    Utils::RunRealizations(
        0, 4,
        [](size_t) {
            throw std::runtime_error("unexpected preparation");
            return 0;
        },
        [](size_t, size_t, int) { throw std::runtime_error("unexpected execution"); });
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(1);
    Require(Utils::RealizationWorkerCount(100, 8) == 1, "caller OpenMP budget was ignored");
    omp_set_num_threads(previous);
    std::atomic<bool> nestedSerial{true};
#pragma omp parallel num_threads(2)
    {
        if (Utils::RealizationWorkerCount(100, 8) != 1)
            nestedSerial = false;
    }
    Require(nestedSerial, "nested OpenMP call acquired outer workers");
#endif
}

void NestedRealizationsStaySerial()
{
    if (Utils::RealizationWorkerCount(2, 2) < 2)
        return;
    std::atomic<size_t> nested{0};
    Utils::RunRealizations(
        8, 2, [](size_t index) { return index; },
        [&](size_t, size_t, size_t) {
            const auto caller = std::this_thread::get_id();
            Require(Utils::RealizationWorkerCount(8, 4) == 1, "nested realization acquired another outer team");
            Utils::RunRealizations(
                3, 4, [](size_t index) { return index; },
                [&](size_t worker, size_t, size_t) {
                    Require(worker == 0 && caller == std::this_thread::get_id(), "nested worker did not execute inline");
                    ++nested;
                });
        });
    Require(nested == 24, "nested realization work was lost");
}

#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
void OverlappingOpenMPBudgetLeases()
{
    struct Restore
    {
        int previous = omp_get_max_threads();

        ~Restore()
        {
            omp_set_num_threads(previous);
        }
    } restore;

    omp_set_num_threads(4);
    if (Utils::RealizationWorkerCount(2, 2) < 2)
        return;

    struct Signal
    {
        std::mutex mutex;
        std::condition_variable condition;
        bool ready = false;

        void Set()
        {
            {
                const std::lock_guard lock(mutex);
                ready = true;
            }
            condition.notify_all();
        }

        void Wait()
        {
            std::unique_lock lock(mutex);
            if (!condition.wait_for(lock, std::chrono::seconds(5), [&] { return ready; }))
                throw std::runtime_error("overlapping budget test timed out");
        }
    } enteredA, enteredB, enteredSerial, releaseA, releaseB, releaseSerial;

    std::exception_ptr errorA, errorB, errorSerial, errorMain;
    std::thread a([&] {
        try
        {
            Utils::RunRealizations(
                2, 2, [](size_t index) { return index; },
                [&](size_t, size_t index, size_t) {
                    if (!index)
                    {
                        enteredA.Set();
                        releaseA.Wait();
                    }
                });
        }
        catch (...)
        {
            errorA = std::current_exception();
        }
    });
    std::thread b([&] {
        try
        {
            enteredA.Wait();
            Require(omp_get_max_threads() == 1, "outer realization did not limit the MSVC process");
            Require(Utils::RealizationWorkerCount(2, 2) == 2, "overlapping call lost the saved caller budget");
            Utils::RunRealizations(
                2, 2, [](size_t index) { return index; },
                [&](size_t, size_t index, size_t) {
                    if (!index)
                    {
                        enteredB.Set();
                        releaseB.Wait();
                        throw std::runtime_error("overlap execution failed");
                    }
                });
            throw std::runtime_error("overlap exception was swallowed");
        }
        catch (const std::runtime_error &error)
        {
            if (std::string(error.what()) != "overlap execution failed")
                errorB = std::current_exception();
        }
        catch (...)
        {
            errorB = std::current_exception();
        }
    });
    std::thread serial([&] {
        try
        {
            enteredB.Wait();
            Utils::RunRealizations(
                1, 1, [](size_t) { return 0; },
                [&](size_t, size_t, int) {
                    enteredSerial.Set();
                    releaseSerial.Wait();
                });
        }
        catch (...)
        {
            errorSerial = std::current_exception();
        }
    });
    try
    {
        enteredSerial.Wait();
        releaseA.Set();
        a.join();
        Require(omp_get_max_threads() == 1, "first overlapping exit restored the process limit early");
        releaseB.Set();
        b.join();
        Require(omp_get_max_threads() == 1, "exception exit ignored a concurrent serial fallback");
    }
    catch (...)
    {
        errorMain = std::current_exception();
    }
    releaseA.Set();
    releaseB.Set();
    releaseSerial.Set();
    if (a.joinable())
        a.join();
    if (b.joinable())
        b.join();
    serial.join();
    for (const auto &error : {errorMain, errorA, errorB, errorSerial})
        if (error)
            std::rethrow_exception(error);
    Require(omp_get_max_threads() == 4, "final overlapping exit did not restore the original caller limit");
}
#endif

#ifdef MAESTRO_REALIZATION_NOISE_TESTS
void RequireSameOperation(const std::shared_ptr<Circuits::IOperation<>> &a, const std::shared_ptr<Circuits::IOperation<>> &b)
{
    using Type = Circuits::OperationType;
    Require(a->GetType() == b->GetType(), "planned injection changed an operation type");
    Require(a->AffectedQubits() == b->AffectedQubits() && a->AffectedBits() == b->AffectedBits(), "planned injection changed operation targets");
    switch (a->GetType())
    {
    case Type::kGate: {
        const auto first = std::static_pointer_cast<Circuits::IQuantumGate<>>(a), second = std::static_pointer_cast<Circuits::IQuantumGate<>>(b);
        Require(first->GetGateType() == second->GetGateType() && first->GetParams() == second->GetParams(), "planned injection changed a gate");
        break;
    }
    case Type::kQuantumChannel: {
        const auto &first = std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(a)->GetChannel().GetKrausOperators();
        const auto &second = std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(b)->GetChannel().GetKrausOperators();
        Require(first.size() == second.size(), "planned injection changed a channel");
        for (size_t k = 0; k < first.size(); ++k)
            Require(first[k].rows() == second[k].rows() && first[k].cols() == second[k].cols() && first[k] == second[k], "planned injection changed a channel");
        break;
    }
    case Type::kDelay:
        Require(std::static_pointer_cast<Circuits::Delay<>>(a)->GetDuration() == std::static_pointer_cast<Circuits::Delay<>>(b)->GetDuration(),
                "planned injection changed a delay");
        break;
    case Type::kReset:
        Require(std::static_pointer_cast<Circuits::Reset<>>(a)->GetResetTargets() == std::static_pointer_cast<Circuits::Reset<>>(b)->GetResetTargets(),
                "planned injection changed a reset");
        break;
    case Type::kMeasurement: {
        const auto &first = std::static_pointer_cast<Circuits::MeasurementOperation<>>(a)->GetReadout();
        const auto &second = std::static_pointer_cast<Circuits::MeasurementOperation<>>(b)->GetReadout();
        Require(first.size() == second.size(), "planned injection changed readout");
        for (size_t k = 0; k < first.size(); ++k)
            Require(first[k].p_meas1_prep0 == second[k].p_meas1_prep0 && first[k].p_meas0_prep1 == second[k].p_meas0_prep1,
                    "planned injection changed readout");
        break;
    }
    default:
        break;
    }
}

// A NoiseInjectionPlan must reproduce each free injector exactly: the same
// operations in the same order, and the same noise-RNG state afterwards, for
// every layer, channel mode and successive realization.
void NoiseInjectionPlanMatchesInjectors()
{
    using Q = Types::qubits_vector;
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::TGate<>>(1));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    circuit->AddOperation(std::make_shared<Circuits::Delay<>>(2, 2e-6));
    std::vector<double> pair(16, 0.01);
    pair[0] = 0.85;
    circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Q{1, 2}, Simulators::QuantumChannel::Pauli(pair)));
    circuit->AddOperation(std::make_shared<Circuits::CCXGate<>>(0, 1, 2));
    circuit->AddOperation(std::make_shared<Circuits::RzGate<>>(2, 0.3));
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
    circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Q{0}));
    const auto condition = std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{0}, std::vector<bool>{true});
    circuit->AddOperation(std::make_shared<Circuits::ConditionalGate<>>(std::make_shared<Circuits::XGate<>>(1), condition));
    circuit->AddOperation(std::make_shared<Circuits::Random<>>(std::vector<size_t>{3}, 5));
    // Qubit 4 appears only in a channel, so a realization may need the width delay.
    circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Q{4}, Simulators::QuantumChannel::Pauli(0.3, 0.1, 0.1)));
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{1, 1}, {2, 2}}));

    noise::NoiseModel sampled;
    sampled.set_all_depolarizing(5, 0.05);
    sampled.set_bit_flip(3, 0.2);
    sampled.set_1q_gate_depolarizing(0, 0.04);
    sampled.set_2q_gate_depolarizing(1, 0.06);
    sampled.set_2q_depolarizing(0, 1, 0.08);
    sampled.set_t1(0, 0.05);
    sampled.set_t1_2q(1, 0.07);
    sampled.set_thermal_relaxation(2, 50e-9, 2e-6, 1.5e-6, 0.05);
    sampled.set_thermal_relaxation_2q(2, 300e-9, 2e-6, 1.5e-6);
    sampled.set_phase_damping(1, 0.05);
    sampled.set_idle_noise(2, 3e-6, 2e-6, 0.1, 1e4);
    sampled.set_readout_error(0, 0.05, 0.1);
    sampled.set_readout_error(2, 0.02, 0.03);
    sampled.set_coherent_rotation(0, 0.01, 0.02, 0.03);
    sampled.set_coherent_depolarizing(1, 0.02);
    sampled.set_correlated_ou(2, 0.01, 1e5, 50e-9);
    sampled.set_crosstalk(0, 3, 0.02);
    sampled.set_crosstalk(1, 9, 0.03); // A spectator outside the circuit's register.

    noise::NoiseModel exact = sampled;
    exact.set_generalized_amplitude_damping(3, 0.05, 0.1);
    exact.set_correlated_phase_flip(0, 1, 0.02);
    Simulators::QuantumChannel::KrausOperators damping(2, Simulators::QuantumChannel::Matrix::Zero(2, 2));
    damping[0](0, 0) = 1.0;
    damping[0](1, 1) = std::sqrt(0.9);
    damping[1](0, 1) = std::sqrt(0.1);
    exact.set_kraus_channel(Q{1}, damping);

    for (const auto *model : {&sampled, &exact})
    {
        const noise::NoiseInjectionPlan plan(circuit, *model);
        for (const bool exactChannels : {false, true})
            for (int layer = 0; layer < 3; ++layer)
            {
                std::mt19937 direct(31), planned(31);
                for (size_t realization = 0; realization < 4; ++realization)
                {
                    std::shared_ptr<Circuits::Circuit<>> expected, actual;
                    bool expectedThrew = false, actualThrew = false;
                    try
                    {
                        if (layer == 0)
                            expected = exactChannels ? noise::inject_exact_noise(circuit, *model) : noise::inject_noise(circuit, *model, direct);
                        else if (layer == 1)
                            expected = exactChannels ? noise::inject_coherent_noise_exact(circuit, *model, direct)
                                                     : noise::inject_coherent_noise(circuit, *model, direct);
                        else
                            expected = exactChannels ? noise::inject_combined_noise_exact(circuit, *model, direct)
                                                     : noise::inject_combined_noise(circuit, *model, direct);
                    }
                    catch (const std::invalid_argument &)
                    {
                        expectedThrew = true;
                    }
                    try
                    {
                        if (layer == 0)
                            actual = exactChannels ? plan.ExactPauli() : plan.Pauli(planned);
                        else if (layer == 1)
                            actual = plan.Coherent(planned, exactChannels);
                        else
                            actual = plan.Combined(planned, exactChannels);
                    }
                    catch (const std::invalid_argument &)
                    {
                        actualThrew = true;
                    }
                    Require(expectedThrew == actualThrew, "planned injection changed noise-model validation");
                    Require(direct == planned, "planned injection consumed a different noise stream");
                    if (expectedThrew)
                        break;
                    noise::attach_readout_error(expected, *model);
                    plan.AttachReadout(actual);
                    const auto &expectedOps = expected->GetOperations(), &actualOps = actual->GetOperations();
                    Require(expectedOps.size() == actualOps.size(), "planned injection changed the operation count");
                    for (size_t index = 0; index < expectedOps.size(); ++index)
                        RequireSameOperation(expectedOps[index], actualOps[index]);
                }
            }
    }
    // Readout is attached to per-realization copies, never to the caller's circuit.
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() == Circuits::OperationType::kMeasurement)
            Require(std::static_pointer_cast<Circuits::MeasurementOperation<>>(op)->GetReadout().empty(), "injection attached readout to the source circuit");
}

void InjectionCopiesAndRegisterWidth()
{
    using Measure = Circuits::MeasurementOperation<>;
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    const auto ry = std::make_shared<Circuits::RyGate<>>(0, 0.25);
    circuit->AddOperation(ry);
    circuit->AddOperation(std::make_shared<Measure>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
    noise::NoiseModel model;
    model.set_bit_flip(0, 0.2);
    model.set_coherent_rotation(0, 0.01, 0.0, 0.0);
    // The public injectors return independent copies a caller may edit.
    const auto edited = [&](const std::shared_ptr<Circuits::Circuit<>> &noisy) {
        for (const auto &op : noisy->GetOperations())
            if (const auto gate = std::dynamic_pointer_cast<Circuits::RyGate<>>(op); gate && gate->GetQubit() == 0 && gate->GetTheta() == 0.25)
            {
                Require(gate != ry, "public injection shared a source gate");
                gate->SetTheta(1.5);
                return ry->GetTheta() == 0.25;
            }
        Require(false, "injected circuit lost the source gate");
        return false;
    };
    std::mt19937 rng(5);
    Require(edited(noise::inject_noise(circuit, model, rng)), "editing inject_noise output changed the source");
    Require(edited(noise::inject_exact_noise(circuit, model)), "editing inject_exact_noise output changed the source");
    Require(edited(noise::inject_coherent_noise(circuit, model, rng)), "editing inject_coherent_noise output changed the source");
    Require(edited(noise::inject_coherent_noise_exact(circuit, model, rng)), "editing inject_coherent_noise_exact output changed the source");
    Require(edited(noise::inject_combined_noise(circuit, model, rng)), "editing inject_combined_noise output changed the source");
    Require(edited(noise::inject_combined_noise_exact(circuit, model, rng)), "editing inject_combined_noise_exact output changed the source");
    Require(edited(noise::inject_correlated_noise(circuit, model, rng)), "editing inject_correlated_noise output changed the source");
    // Plan realizations are read-only and share plain gates with the source.
    const noise::NoiseInjectionPlan plan(circuit, model);
    Require(plan.Pauli(rng)->GetOperations().front() == ry, "plan realizations no longer share immutable gates");
    Require(plan.RegisterWidth() == 1, "a model without crosstalk widened the register");

    // Crosstalk rotates spectators beyond the circuit's register.
    auto spectators = std::make_shared<Circuits::Circuit<>>();
    spectators->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    spectators->AddOperation(std::make_shared<Circuits::TGate<>>(0));
    spectators->AddOperation(std::make_shared<Measure>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
    noise::NoiseModel crosstalk;
    crosstalk.set_crosstalk(0, 1, 0.2);
    crosstalk.set_crosstalk(0, 5, 0.1);
    const noise::NoiseInjectionPlan reach(spectators, crosstalk);
    Require(reach.RegisterWidth() == 6, "register width ignored crosstalk spectators");
    const auto noisy = reach.Combined(rng, false);
    Require(noisy->GetMaxQubitIndex() == 5, "crosstalk realization did not reach its spectator");
    // Outside a fixed register, the spectators' rotations are a global phase.
    const auto trimmed = noise::drop_outside_spectator_phases(noisy, 1);
    Require(trimmed->GetMaxQubitIndex() == 0, "spectator rotations outside the register were kept");
    size_t kept = 0;
    for (const auto &op : noisy->GetOperations())
        kept += op->AffectedQubits().front() == 0;
    Require(trimmed->GetOperations().size() == kept, "trimming removed an operation inside the register");
    Require(noise::drop_outside_spectator_phases(noisy, 6)->GetOperations().size() == noisy->GetOperations().size(),
            "trimming changed a realization that fits its register");
    auto flipped = std::make_shared<Circuits::Circuit<>>(noisy->GetOperations());
    flipped->AddOperation(std::make_shared<Circuits::XGate<>>(3));
    bool rejected = false;
    try
    {
        noise::drop_outside_spectator_phases(flipped, 1);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "trimming dropped a non-phase operation outside the register");
}

void NoiseAddFrameCountPolicies()
{
    using Method = Simulators::SimulationType;
    const auto makeNetwork = [](const char *policy, Method method = Method::kStabilizer, bool optimize = false) {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{5}, std::vector<size_t>{5});
        network->Configure("counts_policy", policy);
        network->SetOptimizeSimulator(optimize);
        network->GetController()->SetOptimizeCircuit(false);
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
        return network;
    };

    class CountExecutionX : public Circuits::XGate<>
    {
      public:
        explicit CountExecutionX(std::shared_ptr<std::atomic<size_t>> calls) : XGate(0), calls(std::move(calls))
        {
        }

        void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
        {
            ++*calls;
            XGate::Execute(simulator, state);
        }

        std::shared_ptr<Circuits::IOperation<>> Clone() const override
        {
            return std::make_shared<CountExecutionX>(*this);
        }

        std::shared_ptr<Circuits::IOperation<>> Remap(const std::unordered_map<Types::qubit_t, Types::qubit_t> &,
                                                      const std::unordered_map<Types::qubit_t, Types::qubit_t> & = {}) const override
        {
            return Clone();
        }

      private:
        std::shared_ptr<std::atomic<size_t>> calls;
    };

    for (const char *policy : {"auto", "legacy", "frame_v1"})
        for (bool seeded : {false, true})
        {
            auto calls = std::make_shared<std::atomic<size_t>>(0);
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<CountExecutionX>(calls));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
            noise::NoiseModel certain;
            certain.set_bit_flip(0, 1.0);
            noise::NoiseAdd add;
            if (seeded)
                add.seed(0);
            const auto counts = add.noisy_execute(circuit, makeNetwork(policy), 0, certain, 41, 41);
            Require(counts.size() == 1 && counts.at(std::vector<bool>(5, false)) == 41, "count policy changed a certain Pauli error");
            const bool framed = std::string(policy) != "legacy";
            Require(framed ? calls->load() == 0 : calls->load() > 0, "count policy did not select the requested execution path");
        }
    // A requested non-Stabilizer method needs backend optimization. Under auto
    // the ordinary executor must also be able to pick the stabilizer itself:
    // a candidate set that excludes it, or exact channels, which pin the
    // requested mixed-state backend, keep the trajectory path. A restricted
    // candidate set binds an optimizing Stabilizer request too.
    enum class Case
    {
        Fixed,
        Optimized,
        RestrictedCandidates,
        ExactChannels,
        StabilizerOptimized,
        StabilizerRestricted
    };
    for (const char *policy : {"auto", "frame_v1"})
        for (const Case scenario :
             {Case::Fixed, Case::Optimized, Case::RestrictedCandidates, Case::ExactChannels, Case::StabilizerOptimized, Case::StabilizerRestricted})
        {
            auto calls = std::make_shared<std::atomic<size_t>>(0);
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<CountExecutionX>(calls));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
            noise::NoiseModel certain;
            certain.set_bit_flip(0, 1.0);
            noise::NoiseAdd add;
            const bool stabilizer = scenario == Case::StabilizerOptimized || scenario == Case::StabilizerRestricted;
            const Method method = scenario == Case::ExactChannels ? Method::kDensityMatrix : stabilizer ? Method::kStabilizer : Method::kStatevector;
            auto network = makeNetwork(policy, method, scenario != Case::Fixed);
            if (scenario == Case::RestrictedCandidates || scenario == Case::StabilizerRestricted)
                network->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Method::kStatevector);
            add.set_exact_channels(scenario == Case::ExactChannels);
            const auto counts = add.noisy_execute(circuit, network, 0, certain, 41, 41);
            Require(counts.size() == 1 && counts.at(std::vector<bool>(5, false)) == 41, "non-Stabilizer count policy changed a certain Pauli error");
            // A candidate set without the stabilizer binds frame_v1 as well.
            const bool restricted = scenario == Case::RestrictedCandidates || scenario == Case::StabilizerRestricted;
            const bool framed = scenario != Case::Fixed && !restricted &&
                                (std::string(policy) == "frame_v1" || scenario == Case::Optimized || scenario == Case::StabilizerOptimized);
            Require(framed ? calls->load() == 0 : calls->load() > 0, "count policy replaced a method the ordinary executor would honor");
        }
    auto bell = std::make_shared<Circuits::Circuit<>>();
    bell->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    bell->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    bell->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}));
    noise::NoiseModel model;
    model.set_all_depolarizing(2, 0.17);
    model.set_readout_error_symmetric(0, 0.1);
    for (bool configuredSeed : {false, true})
    {
        auto automatic = makeNetwork("auto"), replay = makeNetwork("auto");
        if (configuredSeed)
        {
            automatic->Configure("seed", "0");
            replay->Configure("seed", "0");
        }
        noise::NoiseAdd a, b;
        a.seed(0);
        b.seed(0);
        for (size_t call = 0; call < 2; ++call)
            Require(a.full_noise_execute(bell, automatic, 0, model, 127, 127) == b.full_noise_execute(bell, replay, 0, model, 127, 127),
                    "automatic frame policy did not replay a seeded call sequence");
    }
    auto frame = makeNetwork("frame_v1");
    noise::NoiseAdd framed;
    framed.seed(17);
    std::vector<Circuits::Circuit<>::ExecuteResults> sequence;
    for (size_t call = 0; call < 3; ++call)
        sequence.push_back(framed.full_noise_execute(bell, frame, 0, model, 2049, 2049));
    Require(sequence[0] != sequence[1] || sequence[1] != sequence[2], "frame policy replayed successive calls");
    framed.seed(17);
    for (const auto &expected : sequence)
        Require(framed.full_noise_execute(bell, frame, 0, model, 2049, 2049) == expected, "reseed did not replay frame count calls");

    // Shared-noise realizations and unsupported gates must use the unchanged
    // seeded trajectory executor even when a caller requests frame sampling.
    for (bool nonClifford : {false, true})
    {
        auto circuit = std::static_pointer_cast<Circuits::Circuit<>>(bell->Clone());
        if (nonClifford)
        {
            circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
            circuit->AddOperation(std::make_shared<Circuits::TGate<>>(0));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
        }
        const Method method = nonClifford ? Method::kStatevector : Method::kStabilizer;
        auto requested = makeNetwork("frame_v1", method), legacy = makeNetwork("legacy", method);
        noise::NoiseAdd a, b;
        a.seed(31);
        b.seed(31);
        const int realizations = nonClifford ? 137 : 7;
        Require(a.noisy_execute(circuit, requested, 0, model, 137, realizations) == b.noisy_execute(circuit, legacy, 0, model, 137, realizations),
                "unsupported frame request did not preserve seeded trajectory fallback");
    }
    auto sparse = std::make_shared<Circuits::Circuit<>>();
    sparse->AddOperation(std::make_shared<Circuits::XGate<>>(4));
    sparse->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{4, 0}}));
    noise::NoiseModel certain;
    certain.set_bit_flip(4, 1.0);
    noise::NoiseAdd automatic;
    const auto counts = automatic.noisy_execute(sparse, makeNetwork("auto"), 0, certain, 83, 83);
    noise::NoiseAdd legacySparse;
    const auto baseline = legacySparse.noisy_execute(sparse, makeNetwork("legacy"), 0, certain, 83, 83);
    Require(counts == baseline && counts.size() == 1 && counts.at(std::vector<bool>(5, false)) == 83,
            "frame counts changed sparse classical output width or network padding");

    const auto measured = [](std::initializer_list<std::shared_ptr<Circuits::IOperation<>>> gates) {
        auto circuit = std::make_shared<Circuits::Circuit<>>();
        for (const auto &gate : gates)
            circuit->AddOperation(gate);
        circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
        return circuit;
    };
    // Frames inject exact channels: thermal noise that the trajectories clamp
    // (T1 < T2 <= 2 T1), and warn about, keeps the trajectories.
    const auto twice = measured({std::make_shared<Circuits::HadamardGate<>>(0), std::make_shared<Circuits::HadamardGate<>>(0)});
    for (const double t2 : {2.0, 1.0})
    {
        noise::NoiseModel thermal;
        thermal.set_thermal_relaxation(0, 1.0, 1.0, t2, 0.5);
        noise::NoiseAdd framedAdd, legacyAdd;
        for (auto *add : {&framedAdd, &legacyAdd})
        {
            add->set_warning_handler([](const std::string &) {});
            add->seed(4);
        }
        const bool same = framedAdd.full_noise_execute(twice, makeNetwork("frame_v1"), 0, thermal, 2000, 2000) ==
                          legacyAdd.full_noise_execute(twice, makeNetwork("legacy"), 0, thermal, 2000, 2000);
        Require(same == (t2 > 1.0), t2 > 1.0 ? "clamped thermal noise ran on frames" : "unclamped thermal noise left frames");
    }

    // Frame errors follow the noise seed and outcomes the simulator seed, as
    // on the trajectories, so either can change alone.
    const auto seededFrames = [&](const std::shared_ptr<Circuits::Circuit<>> &circuit, const noise::NoiseModel &model, const char *simulatorSeed,
                                  unsigned noiseSeed) {
        auto network = makeNetwork("frame_v1");
        network->Configure("seed", simulatorSeed);
        noise::NoiseAdd add;
        add.seed(noiseSeed);
        return add.full_noise_execute(circuit, network, 0, model, 2000, 2000);
    };
    const auto flipped = measured({std::make_shared<Circuits::XGate<>>(0)});
    noise::NoiseModel flip;
    flip.set_bit_flip(0, 0.3);
    Require(seededFrames(flipped, flip, "1", 7) == seededFrames(flipped, flip, "2", 7), "the simulator seed changed frame errors");
    Require(seededFrames(flipped, flip, "1", 7) != seededFrames(flipped, flip, "1", 8), "the noise seed did not drive frame errors");
    const auto coin = measured({std::make_shared<Circuits::HadamardGate<>>(0), std::make_shared<Circuits::XGate<>>(1)});
    noise::NoiseModel aside;
    aside.set_bit_flip(1, 0.3);
    Require(seededFrames(coin, aside, "1", 7) == seededFrames(coin, aside, "1", 8), "the noise seed changed frame outcomes");
    Require(seededFrames(coin, aside, "1", 7) != seededFrames(coin, aside, "2", 7), "the simulator seed did not drive frame outcomes");

    // A null circuit injects to null, warning or not.
    noise::NoiseAdd nullInjector;
    Require(!nullInjector.inject(nullptr, flip) && !nullInjector.inject_combined(nullptr, flip), "a null circuit did not inject to null");

    bool rejected = false;
    try
    {
        makeNetwork("invalid");
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "invalid network counts policy was accepted");
}

void NoiseAddBudgetsNetworkAllocation()
{
    class ObserveThreadX : public Circuits::XGate<>
    {
      public:
        ObserveThreadX(std::shared_ptr<std::atomic<bool>> foreign, std::thread::id caller) : XGate(0), foreign(std::move(foreign)), caller(caller)
        {
        }

        void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
        {
            if (std::this_thread::get_id() != caller)
                *foreign = true;
            XGate::Execute(simulator, state);
        }

        std::shared_ptr<Circuits::IOperation<>> Clone() const override
        {
            return std::make_shared<ObserveThreadX>(*this);
        }

        std::shared_ptr<Circuits::IOperation<>> Remap(const std::unordered_map<Types::qubit_t, Types::qubit_t> &,
                                                      const std::unordered_map<Types::qubit_t, Types::qubit_t> & = {}) const override
        {
            return Clone(); // The test uses qubit 0 and classical bit 0.
        }

      private:
        std::shared_ptr<std::atomic<bool>> foreign;
        std::thread::id caller;
    };

    for (size_t hostQubits : {size_t{14}, size_t{16}})
        for (size_t budgetMB : {size_t{1}, size_t{2}, size_t{8}})
        {
            // The caller's register stays allocated while workers run: a
            // whole-network statevector of 15 qubits (512 KiB) on a 14-qubit
            // host, of 17 qubits (2 MiB) on a 16-qubit one. Worker clones hold
            // only backend placeholders and each realization's register is the
            // one qubit the circuit uses, so the 512 KiB caller leaves room for
            // workers within 1 MiB, while the 2 MiB one exhausts 1-2 MiB alone
            // and forces a serial fallback.
            auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{hostQubits}, std::vector<size_t>{hostQubits});
            network->Configure("trajectory_max_memory_mb", std::to_string(budgetMB).c_str());
            network->SetMaxSimulators(4);
            network->SetOptimizeSimulator(false);
            network->GetController()->SetOptimizeCircuit(false);
            network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
            auto foreign = std::make_shared<std::atomic<bool>>(false);
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<ObserveThreadX>(foreign, std::this_thread::get_id()));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
            noise::NoiseModel model;
            model.set_bit_flip(0, 0.1);
            noise::NoiseAdd add;
            add.seed(23);
            const auto counts = add.noisy_execute(circuit, network, 0, model, 41, 41);
            size_t total = 0;
            for (const auto &entry : counts)
                total += entry.second;
            Require(total == 41, "memory-limited noisy execution lost shots");
            if (hostQubits == 16 && budgetMB <= 2)
                Require(!foreign->load(), "the caller's retained register was left out of the worker budget");
            else if (Utils::RealizationWorkerCount(41, network->GetMaxSimulators()) > 1)
                Require(foreign->load(), "placeholder worker clones were budgeted as whole-network registers");
        }
    {
        // Crosstalk adds rotations on eight spectators of the one-qubit
        // circuit: each realization holds a nine-qubit density matrix (4 MiB),
        // which a 1 MiB budget cannot multiply across workers.
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{9}, std::vector<size_t>{1});
        network->Configure("trajectory_max_memory_mb", "1");
        network->SetMaxSimulators(4);
        network->SetOptimizeSimulator(false);
        network->GetController()->SetOptimizeCircuit(false);
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kDensityMatrix, Network::INetwork<>::kBackendPlaceholderQubits);
        auto foreign = std::make_shared<std::atomic<bool>>(false);
        auto circuit = std::make_shared<Circuits::Circuit<>>();
        circuit->AddOperation(std::make_shared<ObserveThreadX>(foreign, std::this_thread::get_id()));
        circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
        noise::NoiseModel model;
        for (int q = 1; q < 9; ++q)
            model.set_crosstalk(0, q, 0.125);
        noise::NoiseAdd add;
        add.seed(11);
        const auto counts = add.full_noise_execute(circuit, network, 0, model, 8, 8);
        size_t total = 0;
        for (const auto &entry : counts)
            total += entry.second;
        Require(total == 8, "spectator noisy execution lost shots");
        Require(!foreign->load(), "crosstalk spectators were left out of the realization worker budget");
    }
    // Z on qubit 0 keeps the instrumented X gate in the observable's cone. A
    // bit flip is diagonal in the Pauli basis, so it no longer mixes the axes
    // of an X observable, which commutes with that gate and would drop it.
    for (const std::string observable : {"Z", "ZZZZZZZZZ"})
    {
        // An expectation run allocates every position its observables name,
        // idle ones included: a nine-qubit Pauli string makes each realization
        // a nine-qubit density matrix (4 MiB), which a 1 MiB budget cannot
        // multiply across workers, while a one-qubit observable leaves room.
        const bool wide = observable.size() > 1;
        noise::NoiseModel model;
        model.set_bit_flip(0, 0.125);
        const auto observed = [](const std::shared_ptr<std::atomic<bool>> &foreign) {
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<ObserveThreadX>(foreign, std::this_thread::get_id()));
            return circuit;
        };
        const auto check = [&](const std::shared_ptr<std::atomic<bool>> &foreign, const char *serialMessage, const char *parallelMessage) {
            if (wide)
                Require(!foreign->load(), serialMessage);
            else if (Utils::RealizationWorkerCount(8, 4) > 1)
                Require(foreign->load(), parallelMessage);
        };
        {
            auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{9}, std::vector<size_t>{1});
            network->Configure("trajectory_max_memory_mb", "1");
            network->SetMaxSimulators(4);
            network->SetOptimizeSimulator(false);
            network->GetController()->SetOptimizeCircuit(false);
            network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kDensityMatrix,
                                     Network::INetwork<>::kBackendPlaceholderQubits);
            auto foreign = std::make_shared<std::atomic<bool>>(false);
            noise::NoiseAdd add;
            add.seed(11);
            const auto values = add.full_noise_estimate(observed(foreign), network, 0, {observable}, model, 8);
            Require(values.size() == 1 && std::isfinite(values[0]), "noisy estimate lost its value");
            check(foreign, "observable qubits were left out of the noisy estimate worker budget",
                  "a one-qubit observable serialized the noisy estimate workers");
        }
        {
            MaestroExecution::SimulatorConfig config;
            config.simulator_type = Simulators::SimulatorType::kQCSim;
            config.simulation_type = Simulators::SimulationType::kDensityMatrix;
            config.fixed_backend = true;
            config.max_simulators = 4;
            config.trajectory_max_memory_mb = 1;
            auto foreign = std::make_shared<std::atomic<bool>>(false);
            const auto estimate =
                MaestroExecution::EstimateNoisy(observed(foreign), {observable}, model, 8, config, 11, MaestroExecution::NoiseLayer::kCombined);
            Require(estimate.values.size() == 1 && estimate.realizations == 8, "library noisy estimate lost realizations");
            check(foreign, "observable qubits were left out of the library noisy estimate worker budget",
                  "a one-qubit observable serialized the library noisy estimate workers");
        }
    }
    for (bool pinned : {false, true})
    {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{2}, std::vector<size_t>{2});
        network->Configure("counts_policy", "legacy");
        network->Configure("trajectory_max_memory_mb", "8");
        network->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
        if (!pinned)
            network->AddOptimizationSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
        network->SetOptimizeSimulator(true);
        network->SetMaxSimulators(4);
        network->GetController()->SetOptimizeCircuit(false);
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
        auto foreign = std::make_shared<std::atomic<bool>>(false);
        auto circuit = std::make_shared<Circuits::Circuit<>>();
        circuit->AddOperation(std::make_shared<ObserveThreadX>(foreign, std::this_thread::get_id()));
        circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
        noise::NoiseModel model;
        model.set_coherent_rotation(0, 0.0, 0.0, 0.1);
        noise::NoiseAdd add; // Keep backend selection unpinned by a public seed.
        const auto counts = add.full_noise_execute(circuit, network, 0, model, 31, 31);
        size_t total = 0;
        for (const auto &entry : counts)
            total += entry.second;
        Require(total == 31, "automatic backend fallback lost noisy shots");
        if (!pinned)
            Require(!foreign->load(), "mixed/unbounded backend candidates escaped the source-method memory estimate");
        else if (Utils::RealizationWorkerCount(31, network->GetMaxSimulators()) > 1)
            Require(foreign->load(), "a known singleton backend unnecessarily disabled realization workers");
    }

    for (const auto method :
         {Simulators::SimulationType::kStabilizer, Simulators::SimulationType::kMatrixProductState, Simulators::SimulationType::kMatrixProductOperator})
        for (const bool automatic : {false, true})
        {
            // With a bond cap of four, each two-qubit tensor/tableau estimate
            // is just over 1 MiB. Reserve the caller, execution clone and spare
            // first. Six MiB then admit only one two-copy fixed one-shot worker;
            // eight MiB admit only one four-copy worker with a prepared source
            // and cloned checkpoints. Thirty-two MiB permit parallel workers.
            const bool mpo = method == Simulators::SimulationType::kMatrixProductOperator;
            const int shots = mpo ? 129 : 8;
            const size_t lowBudget = automatic || mpo ? 8 : 6;
            auto foreign = std::make_shared<std::atomic<bool>>(false);
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<ObserveThreadX>(foreign, std::this_thread::get_id()));
            circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
            circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}));
            noise::NoiseModel model;
            model.set_bit_flip(1, 0.125);
            const auto makeNetwork = [&](size_t budgetMB) {
                auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{2}, std::vector<size_t>{2});
                network->Configure("counts_policy", "legacy");
                network->Configure("trajectory_max_memory_mb", std::to_string(budgetMB).c_str());
                network->Configure("matrix_product_state_max_bond_dimension", "4");
                network->SetMaxSimulators(4);
                network->SetOptimizeSimulator(automatic);
                if (automatic)
                    network->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, method);
                network->GetController()->SetOptimizeCircuit(false);
                network->CreateSimulator(Simulators::SimulatorType::kQCSim, method, Network::INetwork<>::kBackendPlaceholderQubits);
                return network;
            };
            Circuits::Circuit<>::ExecuteResults serialCounts;
            for (const size_t budgetMB : {lowBudget, size_t{32}})
            {
                foreign->store(false);
                noise::NoiseAdd add;
                add.seed(23);
                add.set_exact_channels(false);
                const auto counts = add.noisy_execute(circuit, makeNetwork(budgetMB), 0, model, shots, 8);
                size_t total = 0;
                for (const auto &entry : counts)
                    total += entry.second;
                Require(total == static_cast<size_t>(shots), "tensor/tableau memory budget lost noisy shots");
                if (budgetMB == lowBudget)
                {
                    Require(!foreign->load(), "tensor/tableau realization budget omitted retained checkpoints and clones");
                    serialCounts = counts;
                }
                else
                {
                    if (Utils::RealizationWorkerCount(8, 4) > 1)
                        Require(foreign->load(), "tensor/tableau realization workers did not use the larger memory budget");
                    Require(counts == serialCounts, "tensor/tableau memory budget changed seeded noisy counts");
                }
                noise::NoiseAdd replay;
                replay.seed(23);
                replay.set_exact_channels(false);
                Require(replay.noisy_execute(circuit, makeNetwork(budgetMB), 0, model, shots, 8) == counts,
                        "tensor/tableau memory-limited realizations did not replay");
            }
        }
}

#ifndef NO_QISKIT_AER
// The network's shot workers on the Aer statevector and matrix product state follow the trajectory
// memory budget. A circuit with a gate after a measurement runs shot by shot on those workers; the
// gate records whether a thread other than the caller executed it.
void AerShotWorkersFollowTheBudget()
{
    class ObserveThreadX : public Circuits::XGate<>
    {
      public:
        ObserveThreadX(std::shared_ptr<std::atomic<bool>> foreign, std::thread::id caller) : XGate(0), foreign(std::move(foreign)), caller(caller)
        {
        }

        void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
        {
            if (std::this_thread::get_id() != caller)
                *foreign = true;
            XGate::Execute(simulator, state);
        }

        std::shared_ptr<Circuits::IOperation<>> Clone() const override
        {
            return std::make_shared<ObserveThreadX>(*this);
        }

        std::shared_ptr<Circuits::IOperation<>> Remap(const std::unordered_map<Types::qubit_t, Types::qubit_t> &,
                                                      const std::unordered_map<Types::qubit_t, Types::qubit_t> & = {}) const override
        {
            return Clone(); // The test uses qubit 0.
        }

      private:
        std::shared_ptr<std::atomic<bool>> foreign;
        std::thread::id caller;
    };

    struct Case
    {
        Simulators::SimulationType method;
        size_t qubits;
        const char *budget; // trajectory_max_memory_mb, or null for the default
        const char *bond;   // matrix_product_state_max_bond_dimension, or null
        bool workers;       // whether shot workers are expected
        const char *what;
    };

    const std::vector<Case> cases{
        // A 17-qubit statevector is 2 MiB, and a worker holds two.
        {Simulators::SimulationType::kStatevector, 17, "1", nullptr, false, "a 1 MiB budget did not hold 17-qubit Aer statevector shots to one worker"},
        {Simulators::SimulationType::kStatevector, 17, "64", nullptr, true, "a 64 MiB budget kept 17-qubit Aer statevector shots off the workers"},
        {Simulators::SimulationType::kStatevector, 17, nullptr, nullptr, true, "the default budget kept 17-qubit Aer statevector shots off the workers"},
        // A matrix product state without a bond limit has no known size: one worker under an
        // explicit budget, not limited without one.
        {Simulators::SimulationType::kMatrixProductState, 17, "1", nullptr, false,
         "an explicit budget did not hold Aer MPS shots of unknown size to one worker"},
        {Simulators::SimulationType::kMatrixProductState, 17, nullptr, nullptr, true,
         "Aer MPS shots without a budget or a bond limit were kept off the workers"},
        {Simulators::SimulationType::kMatrixProductState, 17, "1", "32", false, "a 1 MiB budget did not hold bounded Aer MPS shots to one worker"},
        {Simulators::SimulationType::kMatrixProductState, 17, "64", "4", true, "a 64 MiB budget kept small bounded Aer MPS shots off the workers"},
        // A bond limit beyond what 17 qubits can reach (2^8) counts as 256: 105 MiB a state.
        {Simulators::SimulationType::kMatrixProductState, 17, "1", "2147483647", false,
         "a 1 MiB budget did not hold Aer MPS shots with a huge bond limit to one worker"},
        {Simulators::SimulationType::kMatrixProductState, 17, "2048", "2147483647", true,
         "a 2 GiB budget kept Aer MPS shots with a huge bond limit off the workers"},
    };
    for (const auto &c : cases)
    {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{c.qubits}, std::vector<size_t>{2});
        if (c.budget)
            network->Configure("trajectory_max_memory_mb", c.budget);
        if (c.bond)
            network->Configure("matrix_product_state_max_bond_dimension", c.bond);
        network->SetMaxSimulators(4);
        network->SetOptimizeSimulator(false);
        network->GetController()->SetOptimizeCircuit(false);
        network->CreateSimulator(Simulators::SimulatorType::kQiskitAer, c.method);
        auto foreign = std::make_shared<std::atomic<bool>>(false);
        auto circuit = std::make_shared<Circuits::Circuit<>>();
        // The register holds the qubits the circuit uses: a gate on each makes it the whole host.
        for (size_t q = 0; q < c.qubits; ++q)
            circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(q));
        circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
        circuit->AddOperation(std::make_shared<ObserveThreadX>(foreign, std::this_thread::get_id()));
        circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 1}}));
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 64);
        size_t total = 0;
        for (const auto &entry : counts)
        {
            total += entry.second;
            // The X after the first measurement flips the qubit: the two bits always differ.
            Require(entry.first.size() >= 2 && entry.first[0] != entry.first[1], "Aer shot workers returned a wrong outcome");
        }
        Require(total == 64, "Aer shot workers lost shots");
        Require(foreign->load() == c.workers, c.what);
    }
}
#endif

void NoiseAddPreservesRealizationStreams()
{
    using Method = Simulators::SimulationType;
    using Backend = Simulators::SimulatorType;

    struct ExecutionThreads
    {
        std::mutex mutex;
        std::set<std::thread::id> values;
    };

    class ObserveX : public Circuits::XGate<>
    {
      public:
        explicit ObserveX(std::shared_ptr<ExecutionThreads> threads) : XGate(0), threads(std::move(threads))
        {
        }

        void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
        {
            {
                const std::lock_guard lock(threads->mutex);
                threads->values.insert(std::this_thread::get_id());
            }
            XGate::Execute(simulator, state);
        }

        std::shared_ptr<Circuits::IOperation<>> Clone() const override
        {
            return std::make_shared<ObserveX>(*this);
        }

        std::shared_ptr<Circuits::IOperation<>> Remap(const std::unordered_map<Types::qubit_t, Types::qubit_t> &,
                                                      const std::unordered_map<Types::qubit_t, Types::qubit_t> & = {}) const override
        {
            return Clone(); // Test uses contiguous identity qubit/classical maps.
        }

      private:
        std::shared_ptr<ExecutionThreads> threads;
    };

    for (const auto method : {Method::kStabilizer, Method::kStatevector, Method::kMatrixProductState, Method::kDensityMatrix, Method::kMatrixProductOperator})
        for (const std::string configured : {"", "0", "18446744073709551615"})
        {
            const auto observed = std::make_shared<ExecutionThreads>();
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<ObserveX>(observed));
            circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
            circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 2}}));
            circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(1));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}));
            const auto makeNetwork = [&](size_t workers) {
                auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{2}, std::vector<size_t>{3});
                network->SetMaxSimulators(workers);
                network->SetOptimizeSimulator(false);
                network->GetController()->SetOptimizeCircuit(false);
                network->Configure("matrix_product_state_max_bond_dimension", "8");
                if (!configured.empty())
                    network->Configure("seed", configured.c_str());
                network->CreateSimulator(Backend::kQCSim, method);
                return network;
            };
            auto serial = makeNetwork(1), parallel = makeNetwork(4);
            noise::NoiseModel model;
            model.set_all_depolarizing(2, 0.2);
            model.set_readout_error_symmetric(0, 0.15);
            if (method != Method::kStabilizer)
                model.set_all_coherent_depolarizing(2, 0.02);
            noise::NoiseAdd serialNoise, parallelNoise;
            serialNoise.seed(17);
            parallelNoise.seed(17);
            const bool exact = method == Method::kDensityMatrix || method == Method::kMatrixProductOperator;
            serialNoise.set_exact_channels(exact);
            parallelNoise.set_exact_channels(exact);
            Circuits::Circuit<>::ExecuteResults first;
            for (size_t call = 0; call < 3; ++call)
            {
                const auto expected = serialNoise.full_noise_execute(circuit, serial, 0, model, 127, 127);
                observed->values.clear();
                const auto actual = parallelNoise.full_noise_execute(circuit, parallel, 0, model, 127, 127);
                Require(actual == expected, "parallel noisy counts or advancing realization streams changed");
                if (call == 0)
                    first = actual;
                if (Utils::RealizationWorkerCount(127, parallel->GetMaxSimulators()) > 1)
                    Require(!observed->values.count(std::this_thread::get_id()), "noise realizations did not dispatch to worker threads");
                Require(parallel->GetSimulator()->GetConfiguration("seed") == configured, "noisy execution changed the caller's seed");
            }
            parallelNoise.seed(17);
            Require(parallelNoise.full_noise_execute(circuit, parallel, 0, model, 127, 127) == first, "reseed did not reproduce parallel counts");
        }
}

// Monte Carlo estimates run on realization workers with the serial loop's
// values, and multi-shot batches of terminal circuits on a dense
// method run there too, keeping every shot.
void NoiseAddParallelEstimatesAndBatches()
{
    using Method = Simulators::SimulationType;

    struct Threads
    {
        std::mutex mutex;
        std::set<std::thread::id> values;
    };

    class ObserveH : public Circuits::HadamardGate<>
    {
      public:
        explicit ObserveH(std::shared_ptr<Threads> threads) : HadamardGate(0), threads(std::move(threads))
        {
        }

        void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
        {
            {
                const std::lock_guard lock(threads->mutex);
                threads->values.insert(std::this_thread::get_id());
            }
            HadamardGate::Execute(simulator, state);
        }

        std::shared_ptr<Circuits::IOperation<>> Clone() const override
        {
            return std::make_shared<ObserveH>(*this);
        }

        std::shared_ptr<Circuits::IOperation<>> Remap(const std::unordered_map<Types::qubit_t, Types::qubit_t> &,
                                                      const std::unordered_map<Types::qubit_t, Types::qubit_t> & = {}) const override
        {
            return Clone(); // Contiguous identity maps only.
        }

      private:
        std::shared_ptr<Threads> threads;
    };

    const auto makeNetwork = [](size_t workers, Method method, const char *seed) {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{2}, std::vector<size_t>{2});
        network->SetMaxSimulators(workers);
        network->SetOptimizeSimulator(false);
        network->GetController()->SetOptimizeCircuit(false);
        network->Configure("matrix_product_state_max_bond_dimension", "8");
        if (seed)
            network->Configure("seed", seed);
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
        return network;
    };
    noise::NoiseModel model;
    model.set_all_depolarizing(2, 0.1);
    model.set_all_coherent_depolarizing(2, 0.05);
    const std::vector<std::string> paulis{"ZZ", "XX", "ZI", "IX"};
    const bool parallelMachine = Utils::RealizationWorkerCount(48, 4) > 1;
    for (const auto method : {Method::kStatevector, Method::kMatrixProductState})
        for (const char *seed : {"5", static_cast<const char *>(nullptr)})
        {
            const auto observed = std::make_shared<Threads>();
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<ObserveH>(observed));
            circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
            circuit->AddOperation(std::make_shared<Circuits::RxGate<>>(1, 0.4));
            // Without a network seed, NoiseAdd's own seed sets the streams.
            auto serial = makeNetwork(1, method, seed), parallel = makeNetwork(4, method, seed);
            noise::NoiseAdd serialNoise, parallelNoise;
            serialNoise.seed(23);
            parallelNoise.seed(23);
            for (size_t call = 0; call < 2; ++call)
            {
                const auto expected = serialNoise.full_noise_estimate(circuit, serial, 0, paulis, model, 48);
                observed->values.clear();
                const auto actual = parallelNoise.full_noise_estimate(circuit, parallel, 0, paulis, model, 48);
                Require(actual.size() == expected.size(), "parallel estimates changed their shape");
                for (size_t i = 0; i < actual.size(); ++i)
                    Require(std::abs(actual[i] - expected[i]) < 1e-10, "parallel Monte Carlo estimates differ from the serial loop");
                if (parallelMachine)
                    Require(!observed->values.count(std::this_thread::get_id()), "estimate realizations did not dispatch to worker threads");
            }
        }

    // Eight realizations of 250 shots each, sampled from one prepared state.
    // Each realization draws its Pauli errors once for all 250 of its shots, so
    // bit-flip noise would move the correlated count in steps of 250 (with
    // depolarizing p = 0.05, two broken realizations happen ~9% of the time).
    // Dephasing never changes ZZ parity: every correct shot stays correlated.
    const auto observed = std::make_shared<Threads>();
    auto bell = std::make_shared<Circuits::Circuit<>>();
    bell->AddOperation(std::make_shared<ObserveH>(observed));
    bell->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    bell->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}));
    noise::NoiseModel dephasing;
    dephasing.set_all_dephasing(2, 0.05);
    for (const auto method : {Method::kStatevector, Method::kDensityMatrix})
        for (const char *seed : {"0", static_cast<const char *>(nullptr)})
        {
            observed->values.clear();
            noise::NoiseAdd injected;
            if (seed)
                injected.seed(0);
            injected.set_exact_channels(method == Method::kDensityMatrix);
            const auto counts = injected.noisy_execute(bell, makeNetwork(4, method, seed), 0, dephasing, 2000, 8);
            size_t total = 0, correlated = 0;
            for (const auto &[bits, count] : counts)
            {
                total += count;
                correlated += bits.size() >= 2 && bits[0] == bits[1] ? count : 0;
            }
            Require(total == 2000, "parallel multi-shot batches lost shots");
            Require(correlated == total, "parallel multi-shot batches lost Bell correlations");
            if (parallelMachine)
                Require(!observed->values.count(std::this_thread::get_id()), "multi-shot batches did not dispatch to worker threads");
            if (seed)
            {
                noise::NoiseAdd replay;
                replay.seed(0);
                replay.set_exact_channels(method == Method::kDensityMatrix);
                Require(replay.noisy_execute(bell, makeNetwork(4, method, seed), 0, dephasing, 2000, 8) == counts,
                        "seeded parallel prepared-state batches did not replay");

                MaestroExecution::SimulatorConfig config;
                config.simulation_type = method;
                config.fixed_backend = true;
                config.optimize_circuit = false;
                config.seed = 0;
                config.max_simulators = 4;
                observed->values.clear();
                const auto records = MaestroExecution::ExecuteNoisy(bell, dephasing, config, 2000, 8, 0, {});
                if (parallelMachine)
                    Require(!observed->values.count(std::this_thread::get_id()), "seeded library batches did not dispatch to realization workers");
                const auto replayed = MaestroExecution::ExecuteNoisy(bell, dephasing, config, 2000, 8, 0, {}).counts;
                Require(std::is_permutation(records.counts.begin(), records.counts.end(), replayed.begin(), replayed.end()),
                        "seeded library prepared-state batches did not replay");
            }
        }
}

// libmaestro's Monte Carlo estimates, fidelities and prefix checkpoints give
// the same seeded results on one worker or several.
void LibraryRealizationsMatchSerial()
{
    using Method = Simulators::SimulationType;
    using CF = Circuits::CircuitFactory<>;
    const auto makeConfig = [](Method method, size_t workers) {
        MaestroExecution::SimulatorConfig config;
        config.simulator_type = Simulators::SimulatorType::kQCSim;
        config.simulation_type = method;
        config.fixed_backend = true;
        config.seed = 41;
        config.max_simulators = workers;
        config.max_bond_dimension = 8;
        return config;
    };
    const bool parallelMachine = Utils::RealizationWorkerCount(40, 4) > 1;
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    circuit->AddOperation(std::make_shared<Circuits::RxGate<>>(1, 0.4));
    noise::NoiseModel model;
    model.set_all_depolarizing(2, 0.1);
    model.set_all_coherent_depolarizing(2, 0.05);
    const std::vector<std::string> paulis{"ZZ", "XX", "ZI", "IY"};
    const auto equal = [](double a, double b) { return std::abs(a - b) < 1e-12; };
    for (const auto method : {Method::kStatevector, Method::kMatrixProductState})
    {
        const auto serial = makeConfig(method, 1), parallel = makeConfig(method, 4);
        if (parallelMachine)
            Require(MaestroExecution::SamplingWorkers(parallel, 40, 2) > 1, "the parallel configuration has one realization worker");
        const auto expected = MaestroExecution::EstimateNoisy(circuit, paulis, model, 40, serial, 9, MaestroExecution::NoiseLayer::kCombined);
        const auto actual = MaestroExecution::EstimateNoisy(circuit, paulis, model, 40, parallel, 9, MaestroExecution::NoiseLayer::kCombined);
        Require(actual.values.size() == paulis.size() && expected.values.size() == paulis.size(), "noisy estimates changed their shape");
        for (size_t i = 0; i < paulis.size(); ++i)
            Require(equal(actual.values[i], expected.values[i]) && equal(actual.ideal_values[i], expected.ideal_values[i]),
                    "parallel noisy estimates differ from one worker");
        Require(actual.realizations == 40, "noisy estimates lost realizations");
        // Ideal <ZZ> of the Bell pair after Rx(0.4) on qubit 1 is cos(0.4).
        Require(std::abs(expected.ideal_values[0] - std::cos(0.4)) < 1e-9 && expected.values[0] < expected.ideal_values[0] - 0.01,
                "noisy estimates did not apply noise");

        const auto fidelity = MaestroExecution::NoisyFidelity(circuit, model, 40, serial, 9);
        const auto parallelFidelity = MaestroExecution::NoisyFidelity(circuit, model, 40, parallel, 9);
        Require(equal(fidelity.fidelity, parallelFidelity.fidelity) && equal(fidelity.std_error, parallelFidelity.std_error),
                "parallel noisy fidelity differs from one worker");
        Require(fidelity.fidelity > 0.3 && fidelity.fidelity < 0.999 && fidelity.std_error > 0, "noisy fidelity is implausible");
    }

    // Prefix X q1, H q0; the suffix measures q0, then q1 only when c0 is 1.
    // c1 must equal c0 on every row: an unwritten bit reads 0 on every worker.
    // Noise on the idle qubit 2 selects the realization path.
    auto prefix = std::make_shared<Circuits::Circuit<>>();
    prefix->AddOperation(std::make_shared<Circuits::XGate<>>(1));
    prefix->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    auto suffix = std::make_shared<Circuits::Circuit<>>();
    suffix->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
    suffix->AddOperation(CF::CreateConditionalMeasurement(
        std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{1, 1}}), CF::CreateEqualCondition({0}, {true})));
    noise::NoiseModel idle;
    idle.set_bit_flip(2, 0.5);
    Require(MaestroExecution::PrefixCheckpoint::HasNoise(suffix, &idle) && !MaestroExecution::PrefixCheckpoint::HasNoise(suffix, nullptr),
            "checkpoint noise detection is wrong");
    for (const auto method : {Method::kStatevector, Method::kDensityMatrix})
    {
        MaestroExecution::PrefixCheckpoint serial(prefix, 3, makeConfig(method, 1)), parallel(prefix, 3, makeConfig(method, 4));
        std::vector<uint8_t> first;
        for (size_t call = 0; call < 2; ++call)
        {
            const auto expected = serial.ExecuteSuffix(suffix, 2000, &idle, 64, std::nullopt, 0, true);
            const auto actual = parallel.ExecuteSuffix(suffix, 2000, &idle, 64, std::nullopt, 0, true);
            Require(actual.rows == expected.rows && actual.width == 3 && actual.realizations == 64, "parallel checkpoint rows differ from one worker");
            size_t ones = 0;
            for (size_t shot = 0; shot < 2000; ++shot)
            {
                const uint8_t *row = actual.rows.data() + 3 * shot;
                Require(row[1] == row[0] && row[2] == 0, "a checkpoint row kept a classical bit from an earlier shot");
                ones += row[0];
            }
            Require(ones > 850 && ones < 1150, "checkpoint rows are not balanced");
            if (call == 0)
                first = actual.rows;
            else
                Require(actual.rows != first, "seeded checkpoint calls repeated a batch");
        }
        const auto counts = serial.ExecuteSuffix(suffix, 500, nullptr, 1, 3, 0, false);
        size_t total = 0;
        for (const auto &[bits, count] : counts.counts)
        {
            Require(bits.size() == 3 && bits[0] == bits[1] && !bits[2], "noiseless checkpoint counts are wrong");
            total += count;
        }
        Require(total == 500 && counts.realizations == 0, "noiseless checkpoint counts lost shots");
    }
}
#endif
} // namespace

int main()
{
    try
    {
        BudgetsAndEmptyWork();
        OrderedPreparationAndPrivateWorkers();
        FailuresJoinAndRelease();
        NestedRealizationsStaySerial();
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
        OverlappingOpenMPBudgetLeases();
#endif
#ifdef MAESTRO_REALIZATION_NOISE_TESTS
        NoiseAddPreservesRealizationStreams();
        NoiseInjectionPlanMatchesInjectors();
        InjectionCopiesAndRegisterWidth();
        NoiseAddBudgetsNetworkAllocation();
#ifndef NO_QISKIT_AER
        AerShotWorkersFollowTheBudget();
#endif
        NoiseAddFrameCountPolicies();
        NoiseAddParallelEstimatesAndBatches();
        LibraryRealizationsMatchSerial();
#endif
        std::cout << "Realization worker tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
