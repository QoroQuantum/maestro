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
          EvictLastPlan();
        cachedBytes += plan.bytes;
        plans.push_front(std::move(plan));
      }
    }
    return result;
  }

  // Cache metadata only. Tensor values belong to the network and are always
  // recomputed, so the same plan can serve new amplitudes and measurement data.
  void SetPlanCacheByteLimit(size_t bytes) {
    planCacheByteLimit = bytes;
    while (!plans.empty() && cachedBytes > bytes) EvictLastPlan();
  }
  size_t GetPlanCacheByteLimit() const { return planCacheByteLimit; }
  size_t GetCachedPlanCount() const { return plans.size(); }
  size_t GetPlanCacheHits() const { return planCacheHits; }
  void ClearPlanCache() {
    plans.clear();
    cachedBytes = 0;
    planCacheHits = 0;
  }

  std::shared_ptr<ITensorContractor> Clone() const override {
    auto cloned = std::make_shared<ForestContractor>();
    cloned->maxTensorRank = maxTensorRank;
    cloned->enableMultithreading = enableMultithreading;
    cloned->planCacheByteLimit = planCacheByteLimit;
    // Clones start with an independent empty cache, avoiding metadata copies
    // for each simulator/trajectory clone.
    return cloned;
  }

 private:
  struct Step {
    size_t left = 0, right = 0;
    std::vector<std::pair<size_t, size_t>> axes;
  };
  struct CachedPlan {
    std::vector<size_t> signature, inputs;
    std::vector<Step> steps;
    size_t fingerprint = 0, resultSlot = 0, maxRank = 0;
    size_t axisBytes = 0, bytes = 0;
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

  static size_t PlanBytes(const CachedPlan &plan) {
    return sizeof(CachedPlan) + 2 * sizeof(void *) + plan.axisBytes +
           (plan.signature.capacity() + plan.inputs.capacity()) *
               sizeof(size_t) +
           plan.steps.capacity() * sizeof(Step);
  }
  void EvictLastPlan() {
    cachedBytes -= plans.back().bytes;
    plans.pop_back();
  }
  size_t planCacheByteLimit = 8 * 1024 * 1024;
  size_t cachedBytes = 0, planCacheHits = 0;
  std::list<CachedPlan> plans;
};

}  // namespace TensorNetworks

#endif  // __FOREST_CONTRACTOR_H_
