#pragma once

#include "Circuit/Circuit.h"
#include "Export.h"
#include <map>
#include <optional>

namespace MaestroExecution
{
struct BoundedExpectation
{
    double value = 0;
    double error = 0;
};

// Request-local transfer cache. The source circuit must outlive this workspace.
// A deterministic work budget bounds failed attempts across all observables.
struct BoundedPauliWorkspace
{
    std::map<const Circuits::IOperation<double> *, std::map<size_t, std::vector<double>>> transfers;
    size_t work = 0;
    size_t work_limit = 262144;
};

// Bounded work, with fallback if the Pauli expansion grows too large. Unlike a
// depth cutoff, the discarded operator has an explicit coefficient-norm bound.
MAESTRO_TRANSFORM_API std::optional<BoundedExpectation> EvaluateBoundedPauli(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                             const std::string &observable, double tolerance,
                                                                             BoundedPauliWorkspace *workspace = nullptr);
} // namespace MaestroExecution
