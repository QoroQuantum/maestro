#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../Execution/Transforms/Export.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Utils
{
namespace RealizationDetail
{
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
// The accessor lives in libmaestro: header-local state would create unrelated
// leases in the DLL, Python extension and native callers of this header.
struct ProcessOpenMPBudgetState
{
    std::mutex mutex;
    size_t participants = 0;
    int savedThreads = 0;
    int limit = 0;
    std::unordered_map<std::thread::id, size_t> workerDepth;
};

MAESTRO_TRANSFORM_API ProcessOpenMPBudgetState &ProcessOpenMPBudget();

// A serial fallback joins the same lease, so another call cannot restore the
// process limit while this call is still using it. The mutex is never held
// during preparation, execution or a join; a worker can safely call us again.
class ScopedProcessOpenMPBudget
{
  public:
    explicit ScopedProcessOpenMPBudget(int requestedLimit)
    {
        auto &state = ProcessOpenMPBudget();
        const std::lock_guard lock(state.mutex);
        if (!state.participants)
            state.savedThreads = std::max(1, omp_get_max_threads());
        ++state.participants;
        LimitUnlocked(state, requestedLimit);
    }

    void Limit(int requestedLimit)
    {
        auto &state = ProcessOpenMPBudget();
        const std::lock_guard lock(state.mutex);
        LimitUnlocked(state, requestedLimit);
    }

    ~ScopedProcessOpenMPBudget()
    {
        auto &state = ProcessOpenMPBudget();
        const std::lock_guard lock(state.mutex);
        if (--state.participants == 0)
        {
            if (state.limit)
                omp_set_num_threads(state.savedThreads);
            state.savedThreads = 0;
            state.limit = 0;
        }
    }

    ScopedProcessOpenMPBudget(const ScopedProcessOpenMPBudget &) = delete;
    ScopedProcessOpenMPBudget &operator=(const ScopedProcessOpenMPBudget &) = delete;

  private:
    static void LimitUnlocked(ProcessOpenMPBudgetState &state, int requestedLimit)
    {
        if (requestedLimit > 0)
        {
            requestedLimit = std::min(requestedLimit, state.savedThreads);
            if (!state.limit || requestedLimit < state.limit)
            {
                state.limit = requestedLimit;
                omp_set_num_threads(requestedLimit);
            }
        }
    }
};

inline bool IsRealizationWorker()
{
    auto &state = ProcessOpenMPBudget();
    const std::lock_guard lock(state.mutex);
    return state.workerDepth.count(std::this_thread::get_id()) != 0;
}

inline int CallerOpenMPBudget()
{
    auto &state = ProcessOpenMPBudget();
    const std::lock_guard lock(state.mutex);
    return state.participants ? state.savedThreads : omp_get_max_threads();
}

class ScopedRealizationWorker
{
  public:
    ScopedRealizationWorker()
    {
        auto &state = ProcessOpenMPBudget();
        const std::lock_guard lock(state.mutex);
        ++state.workerDepth[std::this_thread::get_id()];
    }

    ~ScopedRealizationWorker()
    {
        auto &state = ProcessOpenMPBudget();
        const std::lock_guard lock(state.mutex);
        const auto current = state.workerDepth.find(std::this_thread::get_id());
        if (--current->second == 0)
            state.workerDepth.erase(current);
    }
};
#else
inline thread_local size_t workerDepth = 0;

inline bool IsRealizationWorker()
{
    return workerDepth != 0;
}

#ifdef _OPENMP
inline int CallerOpenMPBudget()
{
    return omp_get_max_threads();
}
#endif

class ScopedRealizationWorker
{
  public:
    ScopedRealizationWorker()
    {
        ++workerDepth;
    }

    ~ScopedRealizationWorker()
    {
        --workerDepth;
    }
};
#endif
} // namespace RealizationDetail

// The simulator state one realization holds, for the realization executors'
// memory budgets (Python noisy execution, prefix checkpoints, C++ NoiseAdd).
enum class RealizationState
{
    kStatevector,
    kDensityMatrix,
    kStabilizer,
    kMatrixProductState,
    kMatrixProductOperator
};

// Bytes of one realization's state plus gate workspace, or 0 when no bound is
// known (callers then run serially). Bond-capped MPS and MPO terms were
// calibrated on QCSim: net of setup, measured peaks were about 35-40 bytes per
// qubit and squared bond for MPS and about 250 for MPO, so these keep roughly
// a twofold margin, plus a two-site SVD workspace and 1 MiB of scratch.
inline long double RealizationStateBytes(RealizationState state, size_t qubits, size_t bond = 0)
{
    const long double n = static_cast<long double>(qubits);
    const long double scratch = 1 << 20;
    switch (state)
    {
    case RealizationState::kStatevector:
    case RealizationState::kDensityMatrix: {
        const size_t bitsPerQubit = state == RealizationState::kDensityMatrix ? 2 : 1;
        if (qubits > (std::numeric_limits<size_t>::digits - 5) / bitsPerQubit)
            return std::numeric_limits<long double>::infinity();
        return static_cast<long double>(size_t{1} << (qubits * bitsPerQubit)) * 16;
    }
    case RealizationState::kStabilizer:
        return 64 * n * n + scratch;
    case RealizationState::kMatrixProductState:
    case RealizationState::kMatrixProductOperator: {
        if (!bond)
            return 0;
        const long double chi2 = static_cast<long double>(bond) * static_cast<long double>(bond);
        return state == RealizationState::kMatrixProductOperator ? (512 * n + 2048) * chi2 + scratch : (64 * n + 512) * chi2 + scratch;
    }
    }
    return 0;
}

// Realizations share the caller's CPU budget with any inner simulator work.
inline size_t RealizationWorkerCount(size_t total, size_t requested = 0)
{
    if (!total)
        return 0;
    if (RealizationDetail::IsRealizationWorker())
        return 1;
    size_t budget = std::max(1u, std::thread::hardware_concurrency());
#ifdef _OPENMP
    if (omp_in_parallel())
        return 1;
    budget = std::min(budget, static_cast<size_t>(std::max(1, RealizationDetail::CallerOpenMPBudget())));
#endif
    return std::min(total, std::min(budget, requested ? requested : budget));
}

// Keep at least one worker: callers already require enough memory for one shot.
// Include checkpoint copies and backend workspaces in stateBytes/copiesPerWorker.
inline size_t LimitRealizationWorkersByMemory(size_t workers, size_t stateBytes, size_t budgetMB, size_t reservedCopies = 0, size_t copiesPerWorker = 1)
{
    if (!workers || !stateBytes)
        return workers;
    const size_t maximum = std::numeric_limits<size_t>::max();
    const size_t bytes = budgetMB > (maximum >> 20) ? maximum : budgetMB << 20;
    const size_t slots = bytes / stateBytes;
    const size_t available = slots > reservedCopies ? slots - reservedCopies : 0;
    return std::min(workers, std::max<size_t>(1, available / std::max<size_t>(1, copiesPerWorker)));
}

namespace RealizationDetail
{
#if defined(_OPENMP) && (!defined(_MSC_VER) || defined(__clang__) || defined(_OPENMP_LLVM_RUNTIME))
class ScopedSerialOpenMP
{
  public:
    ScopedSerialOpenMP() : previous(omp_get_max_threads())
    {
        omp_set_num_threads(1);
    }

    ~ScopedSerialOpenMP()
    {
        omp_set_num_threads(previous);
    }

  private:
    int previous;
};
#endif
} // namespace RealizationDetail

// Preparation remains ordered on the calling thread, preserving an existing
// serial noise RNG exactly. Only native execution is dispatched to workers.
// At most workers queued payloads, workers executing payloads and one payload
// under preparation are live. Worker indices own private execution state.
// On failure, stop accepting work, join every thread and rethrow the exception.
template <typename Prepare, typename Execute> void RunRealizations(size_t total, size_t workers, Prepare &&prepare, Execute &&execute)
{
    if (!total)
        return;
    workers = RealizationWorkerCount(total, workers);
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
    // Participating calls share one root budget, including serial fallbacks.
    RealizationDetail::ScopedProcessOpenMPBudget processBudget(workers > 1 ? 1 : 0);
#endif
    if (workers <= 1)
    {
        for (size_t index = 0; index < total; ++index)
            execute(size_t{0}, index, prepare(index));
        return;
    }

    using Payload = std::decay_t<std::invoke_result_t<Prepare &, size_t>>;
    std::deque<std::pair<size_t, Payload>> queue;
    std::mutex mutex;
    std::condition_variable available, space;
    bool finished = false, failed = false;
    std::exception_ptr error;
    const auto fail = [&] {
        {
            const std::lock_guard lock(mutex);
            if (!error)
                error = std::current_exception();
            failed = true;
        }
        available.notify_all();
        space.notify_all();
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    try
    {
        for (size_t worker = 0; worker < workers; ++worker)
            threads.emplace_back([&, worker] {
#if defined(_OPENMP) && (!defined(_MSC_VER) || defined(__clang__) || defined(_OPENMP_LLVM_RUNTIME))
                RealizationDetail::ScopedSerialOpenMP serialOpenMP;
#endif
                try
                {
                    RealizationDetail::ScopedRealizationWorker workerScope;
                    for (;;)
                    {
                        std::unique_lock lock(mutex);
                        available.wait(lock, [&] { return failed || finished || !queue.empty(); });
                        if (failed || queue.empty())
                            return;
                        auto work = std::move(queue.front());
                        queue.pop_front();
                        lock.unlock();
                        space.notify_one();
                        execute(worker, work.first, std::move(work.second));
                    }
                }
                catch (...)
                {
                    fail();
                }
            });

        for (size_t index = 0; index < total; ++index)
        {
            {
                std::unique_lock lock(mutex);
                space.wait(lock, [&] { return failed || queue.size() < workers; });
                if (failed)
                    break;
            }
            auto payload = prepare(index);
            {
                const std::lock_guard lock(mutex);
                if (failed)
                    break;
                queue.emplace_back(index, std::move(payload));
            }
            available.notify_one();
        }
    }
    catch (...)
    {
        fail();
    }
    {
        const std::lock_guard lock(mutex);
        finished = true;
    }
    available.notify_all();
    for (auto &thread : threads)
        thread.join();
    if (error)
        std::rethrow_exception(error);
}

} // namespace Utils
