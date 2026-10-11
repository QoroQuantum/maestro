#include "TransformPipeline.h"

#include "CausalConeTransform.h"
#include "ParityReductionTransform.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace MaestroExecution
{
namespace
{
size_t RegisterWidth(const TransformContext &ctx)
{
    size_t width = ctx.circuit->GetMaxQubitIndex() + 1;
    for (const auto &observable : ctx.observables)
        width = std::max(width, observable.size());
    return std::max(size_t{1}, width);
}
} // namespace

void TransformPipeline::Run(TransformContext &ctx)
{
    if (!ctx.circuit)
        throw std::invalid_argument("TransformPipeline requires a non-null circuit.");

    ctx.qubits_before = RegisterWidth(ctx);
    ctx.qubits_after = ctx.qubits_before;
    ctx.applied_transforms.clear();
    ctx.auto_reduced = false;
    ctx.expectation_factors.assign(ctx.observables.size(), 1.0);

    // Per-run instances avoid shared mutable pass state across concurrent calls.
    ParityReductionTransform parity;
    CausalConeTransform causal_cone;
    const std::array<ITransform *, 2> transforms{&parity, &causal_cone};
    for (auto *transform : transforms)
    {
        if (!transform->IsApplicable(ctx))
            continue;
        const auto prev_circuit = ctx.circuit;
        const auto prev_observables = ctx.observables;
        transform->Apply(ctx);
        if (ctx.circuit != prev_circuit || ctx.observables != prev_observables)
        {
            ctx.applied_transforms.emplace_back(transform->Name());
            ctx.qubits_after = RegisterWidth(ctx);
        }
    }
}
} // namespace MaestroExecution
