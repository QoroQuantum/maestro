/**
 * @file ForestContractor.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The Forest Tensor Contractor.
 * A forest circuit (in particular, a tree) is a quantum circuit that can be
 * simulated efficiently on a classical computer. The contraction of the tensors
 * in the network can be done efficiently without increasing the tensors' rank.
 * Just start with contracting the leaves of the circuit tree with the leaves of
 * the 'super' tree, getting a rank-2 tensor (that is, a matrix - equivalent
 * with a one qubit gate if a projector is not involved) which can be contracted
 * with the new leaf... and so on until the root is also contracted out.
 *
 * Tensor contractions using the Forest contraction method.
 */

#pragma once

#ifndef __FOREST_CONTRACTOR_H_
#define __FOREST_CONTRACTOR_H_ 1

#include "BaseContractor.h"

#include <cmath>
#include <list>
#include <queue>

namespace TensorNetworks {

/** Greedy, memory-first contraction with incremental candidates and plan reuse.
 */
class ForestContractor : public BaseContractor {
 public:
  virtual ~ForestContractor() = default;

  double Contract(const TensorNetwork &network, Types::qubit_t qubit) override {
    CachedPlan plan;
    const auto &original = network.GetTensors();
    const auto &group = network.GetQubitGroup(qubit);
    for (size_t i = 0; i < original.size(); ++i)
      if (original[i] && !original[i]->qubits.empty() &&
          group.count(original[i]->qubits[0]))
        plan.inputs.push_back(i);
    if (plan.inputs.empty())
      throw std::invalid_argument("Cannot contract an empty tensor network");

    bool cacheable = MakeSignature(network, qubit, plan);
    if (cacheable) {
      for (auto it = plans.begin(); it != plans.end(); ++it) {
        if (it->fingerprint == plan.fingerprint &&
            it->signature == plan.signature) {
          plans.splice(plans.begin(), plans, it);
          ++planCacheHits;
          if (!plans.front().preparationAttempted)
            PreparePlan(network, plans.front());
          return ExecutePlan(network, plans.front());
        }
      }
    }

    TensorsMap tensors;
    std::unordered_map<Eigen::Index, size_t> slots;
    std::unordered_map<Eigen::Index, size_t> versions;
    tensors.reserve(plan.inputs.size());
    if (cacheable) slots.reserve(plan.inputs.size());
    versions.reserve(plan.inputs.size());
    maxTensorRank = 0;
    for (size_t slot = 0; slot < plan.inputs.size(); ++slot) {
      const auto &node = original[plan.inputs[slot]];
      tensors.emplace(node->GetId(), node->CloneWithoutTensorCopy());
      if (cacheable) slots.emplace(node->GetId(), slot);
      versions.emplace(node->GetId(), 0);
      maxTensorRank = std::max(maxTensorRank, node->GetRank());
    }
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateLater>
        candidates;
    for (const auto &entry : tensors)
      AddCandidates(entry.first, tensors, versions, candidates, true);

    Eigen::Index resultId = tensors.begin()->first;
    while (tensors.size() > 1) {
      while (!candidates.empty()) {
        const auto &candidate = candidates.top();
        const auto a = versions.find(candidate.a),
                   b = versions.find(candidate.b);
        if (a != versions.end() && b != versions.end() &&
            a->second == candidate.versionA && b->second == candidate.versionB)
          break;
        candidates.pop();
      }
      if (candidates.empty())
        throw std::invalid_argument(
            "Tensor network has no connected contraction pair");
      const Candidate candidate = candidates.top();
      candidates.pop();

      if (cacheable) {
        Step step;
        step.left = slots.at(candidate.a);
        step.right = slots.at(candidate.b);
        const auto &node = tensors.at(candidate.a);
        for (size_t i = 0; i < node->connections.size(); ++i)
          if (node->connections[i] == candidate.b)
            step.axes.emplace_back(i, node->connectionsIndices[i]);
        plan.axisBytes += step.axes.capacity() * sizeof(step.axes[0]);
        plan.steps.push_back(std::move(step));
        if (PlanBytes(plan) > planCacheByteLimit) {
          cacheable = false;
          plan.signature.clear();
          plan.steps.clear();
        }
      }

      resultId = ContractNodes(qubit, tensors, candidate.a, candidate.b,
                               candidate.resultRank);
      versions.erase(candidate.b);
      ++versions.at(candidate.a);
      if (candidate.resultRank == 0) {
        if (tensors.size() == 1 ||
            tensors.at(resultId)->contractsTheNeededQubit)
          break;
        tensors.erase(resultId);
        versions.erase(resultId);
      } else {
        AddCandidates(resultId, tensors, versions, candidates, false);
      }
    }

    const double result = std::real(tensors.at(resultId)->tensor->atOffset(0));
    if (cacheable) {
      plan.resultSlot = slots.at(resultId);
      plan.maxRank = maxTensorRank;
      plan.bytes = PlanBytes(plan);
      if (plan.bytes <= planCacheByteLimit) {
        while (!plans.empty() && cachedBytes > planCacheByteLimit - plan.bytes)
          if (!DropLastPreparedPlan()) EvictLastPlan();
        cachedBytes += plan.bytes;
        plans.push_front(std::move(plan));
      }
    }
    return result;
  }

  // The cache holds metadata; the separate workspace holds reusable storage.
  // Tensor values are recomputed for every evaluation, including cache hits.
  void SetPlanCacheByteLimit(size_t bytes) {
    planCacheByteLimit = bytes;
    while (!plans.empty() && cachedBytes > bytes)
      if (!DropLastPreparedPlan()) EvictLastPlan();
    for (auto &plan : plans)
      if (!plan.prepared) plan.preparationAttempted = false;
    if (plans.empty()) ClearWorkspace();
  }
  size_t GetPlanCacheByteLimit() const { return planCacheByteLimit; }
  size_t GetCachedPlanBytes() const { return cachedBytes; }
  size_t GetCachedPlanCount() const { return plans.size(); }
  size_t GetPlanCacheHits() const { return planCacheHits; }
  size_t GetPreparedPlanCount() const {
    return std::count_if(plans.begin(), plans.end(),
                         [](const auto &plan) { return bool(plan.prepared); });
  }
  // Per-contractor retained output storage. Zero disables buffer retention,
  // while still allowing prepared execution with fresh result allocations.
  void SetWorkspaceByteLimit(size_t bytes) {
    workspaceByteLimit = bytes;
    if (GetWorkspaceBytes() > bytes) ClearWorkspace();
  }
  size_t GetWorkspaceByteLimit() const { return workspaceByteLimit; }
  size_t GetWorkspaceBytes() const {
    return workspace.capacity() * sizeof(Complex);
  }
  void ClearPlanCache() {
    plans.clear();
    cachedBytes = 0;
    planCacheHits = 0;
    ClearWorkspace();
  }

  std::shared_ptr<ITensorContractor> Clone() const override {
    auto cloned = std::make_shared<ForestContractor>();
    cloned->maxTensorRank = maxTensorRank;
    cloned->enableMultithreading = enableMultithreading;
    cloned->planCacheByteLimit = planCacheByteLimit;
    cloned->workspaceByteLimit = workspaceByteLimit;
    // Clones start with an independent empty cache, avoiding metadata copies
    // for each simulator/trajectory clone.
    return cloned;
  }

 private:
  using Complex = std::complex<double>;
  struct PreparedStep {
    Utils::detail::TensorContractionPlan contraction;
    size_t outputOffset = 0;
  };
  struct PreparedPlan {
    std::vector<PreparedStep> steps;
    size_t bytes = 0, workspaceSize = 0;
  };
  struct Step {
    size_t left = 0, right = 0;
    std::vector<std::pair<size_t, size_t>> axes;
  };
  struct CachedPlan {
    std::vector<size_t> signature, inputs;
    std::vector<Step> steps;
    size_t fingerprint = 0, resultSlot = 0, maxRank = 0;
    size_t axisBytes = 0, bytes = 0;
    // Immutable after construction, including when the contractor is copied.
    std::shared_ptr<const PreparedPlan> prepared;
    bool preparationAttempted = false;
  };
  struct Candidate {
    Eigen::Index a, b;
    size_t versionA, versionB, resultRank, edge;
    double outputSize, work;
  };
  struct CandidateLater {
    bool operator()(const Candidate &a, const Candidate &b) const {
      if (a.outputSize != b.outputSize) return a.outputSize > b.outputSize;
      if (a.work != b.work) return a.work > b.work;
      // Prefer the circuit's later tensors on exact ties, as the forest
      // contractor traditionally does. Make ordering independent of hash maps.
      if (a.a != b.a) return a.a < b.a;
      if (a.edge != b.edge) return a.edge > b.edge;
      return a.b < b.b;
    }
  };
  using Candidates =
      std::priority_queue<Candidate, std::vector<Candidate>, CandidateLater>;

  static void AddCandidates(
      Eigen::Index id, const TensorsMap &tensors,
      const std::unordered_map<Eigen::Index, size_t> &versions,
      Candidates &candidates, bool initial) {
    const auto &node = tensors.at(id);
    for (size_t i = 0; i < node->connections.size(); ++i) {
      const auto next = node->connections[i];
      if (next == TensorNode::NotConnected || next == id ||
          (initial && id < next))
        continue;
      // Parallel edges describe one contraction over all shared axes.
      if (std::find(node->connections.begin(), node->connections.begin() + i,
                    next) != node->connections.begin() + i)
        continue;
      const auto aId = std::max(id, next), bId = std::min(id, next);
      const auto &a = tensors.at(aId), &b = tensors.at(bId);
      Candidate candidate{aId, bId, versions.at(aId), versions.at(bId), 0, 0,
                          0,   0};
      double reductionSize = 0;
      bool first = true;
      for (size_t axis = 0; axis < a->connections.size(); ++axis) {
        const double dimension = std::log2(double(a->tensor->GetDim(axis)));
        if (a->connections[axis] == bId) {
          reductionSize += dimension;
          if (first) {
            candidate.edge = axis;
            first = false;
          }
        } else {
          ++candidate.resultRank;
          candidate.outputSize += dimension;
        }
      }
      for (size_t axis = 0; axis < b->connections.size(); ++axis)
        if (b->connections[axis] != aId) {
          ++candidate.resultRank;
          candidate.outputSize += std::log2(double(b->tensor->GetDim(axis)));
        }
      // Logarithmic sizes avoid overflow for plans containing high-rank nodes.
      candidate.work = candidate.outputSize + reductionSize;
      candidates.push(candidate);
    }
  }

  bool MakeSignature(const TensorNetwork &network, Types::qubit_t qubit,
                     CachedPlan &plan) const {
    if (!planCacheByteLimit) return false;
    auto append = [&](size_t value) {
      plan.signature.push_back(value);
      plan.fingerprint ^= value + size_t(0x9e3779b97f4a7c15ULL) +
                          (plan.fingerprint << 6) + (plan.fingerprint >> 2);
    };
    append(static_cast<size_t>(qubit));
    for (auto index : plan.inputs) {
      const auto &node = network.GetTensors()[index];
      append(index);
      append(static_cast<size_t>(node->GetId()));
      append(node->GetRank());
      append(node->tensor->IsDummy());
      for (auto dimension : node->tensor->GetDims()) append(dimension);
      append(node->connections.size());
      for (auto connection : node->connections)
        append(static_cast<size_t>(connection));
      append(node->connectionsIndices.size());
      for (auto axis : node->connectionsIndices)
        append(static_cast<size_t>(axis));
      append(node->qubits.size());
      for (auto q : node->qubits) append(static_cast<size_t>(q));
      if (PlanBytes(plan) > planCacheByteLimit) return false;
    }
    return true;
  }

  double ExecutePlan(const TensorNetwork &network, const CachedPlan &plan) {
    if (plan.prepared) return ExecutePrepared(network, plan);
    std::vector<std::shared_ptr<const Utils::Tensor<>>> values;
    values.reserve(plan.inputs.size());
    for (auto index : plan.inputs)
      values.push_back(network.GetTensors()[index]->tensor);
    for (const auto &step : plan.steps) {
      values[step.left] =
          std::make_shared<Utils::Tensor<>>(values[step.left]->Contract(
              *values[step.right], step.axes, enableMultithreading));
      values[step.right].reset();
    }
    maxTensorRank = plan.maxRank;
    return std::real((*values[plan.resultSlot])[size_t(0)]);
  }

  // Build only on reuse. Keep the lightweight order cache if prepared metadata
  // would exceed the budget, and check offset-table sizes before allocation.
  void PreparePlan(const TensorNetwork &network, CachedPlan &plan) {
    plan.preparationAttempted = true;
    // Prepared metadata may replace older prepared metadata, but must not
    // evict contraction orders. This protects multi-qubit query workloads.
    size_t limit = planCacheByteLimit - cachedBytes;
    for (const auto &cached : plans)
      if (cached.prepared) limit += cached.prepared->bytes;
    size_t bytes = sizeof(PreparedPlan) + 2 * sizeof(void *);
    if (!Utils::detail::TensorAddBytes(bytes, plan.steps.size(),
                                       sizeof(PreparedStep), limit))
      return;
    auto prepared = std::make_shared<PreparedPlan>();
    prepared->steps.reserve(plan.steps.size());
    bytes = sizeof(PreparedPlan) + 2 * sizeof(void *);
    if (!Utils::detail::TensorAddBytes(bytes, prepared->steps.capacity(),
                                       sizeof(PreparedStep), limit))
      return;

    std::vector<const std::vector<size_t> *> dimensions;
    dimensions.reserve(plan.inputs.size());
    for (auto index : plan.inputs) {
      const auto &tensor = network.GetTensors()[index]->tensor;
      if (tensor->IsDummy()) return;
      dimensions.push_back(&tensor->GetDims());
    }
    const size_t none = std::numeric_limits<size_t>::max();
    std::vector<size_t> active(plan.inputs.size(), none), free, capacities;
    for (const auto &step : plan.steps) {
      prepared->steps.emplace_back();
      auto &item = prepared->steps.back();
      if (!item.contraction.Prepare(*dimensions[step.left],
                                    *dimensions[step.right], step.axes,
                                    limit - bytes))
        return;
      bytes += item.contraction.ExtraBytes();
      dimensions[step.left] = &item.contraction.GetDims();
      dimensions[step.right] = nullptr;
      const size_t size = item.contraction.GetSize();
      size_t buffer;
      if (free.empty()) {
        buffer = capacities.size();
        capacities.push_back(size);
      } else {
        const auto best =
            std::min_element(free.begin(), free.end(), [&](size_t a, size_t b) {
              const bool fitA = capacities[a] >= size,
                         fitB = capacities[b] >= size;
              if (fitA != fitB) return fitA;
              return fitA ? capacities[a] < capacities[b]
                          : capacities[a] > capacities[b];
            });
        buffer = *best;
        free.erase(best);
        capacities[buffer] = std::max(capacities[buffer], size);
      }
      item.outputOffset = buffer;
      // Choose the destination before releasing either live input.
      if (active[step.left] != none) free.push_back(active[step.left]);
      if (active[step.right] != none) free.push_back(active[step.right]);
      active[step.left] = buffer;
      active[step.right] = none;
    }
    // Place non-overlapping buffers in one arena, retaining only the offsets.
    size_t arenaSize = 0;
    bool fits = true;
    for (auto &capacity : capacities) {
      const size_t size = capacity;
      capacity = arenaSize;
      if (!Utils::detail::TensorAddBytes(
              arenaSize, size, 1,
              std::numeric_limits<size_t>::max() / sizeof(Complex))) {
        fits = false;
        break;
      }
    }
    if (fits) {
      prepared->workspaceSize = arenaSize;
      for (auto &step : prepared->steps)
        step.outputOffset = capacities[step.outputOffset];
    }
    prepared->bytes = bytes;
    while (cachedBytes > planCacheByteLimit - bytes)
      if (!DropLastPreparedPlan()) return;
    plan.prepared = std::move(prepared);
    plan.bytes += bytes;
    cachedBytes += bytes;
  }

  bool PrepareWorkspace(size_t size) {
    if (!size || size > workspaceByteLimit / sizeof(Complex)) return false;
    if (workspace.size() < size) {
      // Exact construction avoids vector growth retaining more than requested.
      std::vector<Complex> replacement(size);
      if (replacement.capacity() > workspaceByteLimit / sizeof(Complex))
        return false;
      workspace.swap(replacement);
    }
    return true;
  }
  void ClearWorkspace() { std::vector<Complex>().swap(workspace); }

  double ExecutePrepared(const TensorNetwork &network, const CachedPlan &plan) {
    const auto &prepared = *plan.prepared;
    const bool reuse = PrepareWorkspace(prepared.workspaceSize);
    std::vector<const Complex *> values;
    values.reserve(plan.inputs.size());
    for (auto index : plan.inputs)
      values.push_back(&(*network.GetTensors()[index]->tensor)[size_t(0)]);
    std::vector<std::unique_ptr<Complex[]>> owned(reuse ? 0
                                                        : plan.inputs.size());
    for (size_t i = 0; i < plan.steps.size(); ++i) {
      const auto &step = plan.steps[i];
      const auto &item = prepared.steps[i];
      std::unique_ptr<Complex[]> fresh;
      Complex *dest;
      if (reuse)
        dest = workspace.data() + item.outputOffset;
      else {
        fresh.reset(new Complex[item.contraction.GetSize()]);
        dest = fresh.get();
      }
      item.contraction.Execute(values[step.left], values[step.right], dest,
                               enableMultithreading);
      if (!reuse) {
        owned[step.left] = std::move(fresh);
        owned[step.right].reset();
      }
      values[step.left] = dest;
      values[step.right] = nullptr;
    }
    maxTensorRank = plan.maxRank;
    return values[plan.resultSlot][0].real();
  }

  static size_t PlanBytes(const CachedPlan &plan) {
    return sizeof(CachedPlan) + 2 * sizeof(void *) + plan.axisBytes +
           (plan.signature.capacity() + plan.inputs.capacity()) *
               sizeof(size_t) +
           plan.steps.capacity() * sizeof(Step) +
           (plan.prepared ? plan.prepared->bytes : 0);
  }
  void EvictLastPlan() {
    cachedBytes -= plans.back().bytes;
    plans.pop_back();
  }
  bool DropLastPreparedPlan() {
    for (auto it = plans.rbegin(); it != plans.rend(); ++it)
      if (it->prepared) {
        cachedBytes -= it->prepared->bytes;
        it->bytes -= it->prepared->bytes;
        it->prepared.reset();
        // Do not repeatedly promote/demote the same plan at a tight budget.
        // A budget change or a new order-cache entry allows another attempt.
        return true;
      }
    return false;
  }
  size_t planCacheByteLimit = 8 * 1024 * 1024;
  size_t cachedBytes = 0, planCacheHits = 0;
  size_t workspaceByteLimit = 16 * 1024 * 1024;
  std::vector<Complex> workspace;
  std::list<CachedPlan> plans;
};

}  // namespace TensorNetworks

#endif  // __FOREST_CONTRACTOR_H_
