#include "TransformPipeline.h"

#include "CausalConeTransform.h"
#include "ParityReductionTransform.h"

#include <algorithm>
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

void TransformPipeline::ApplyParity(TransformContext &ctx)
{
    ParityReductionTransform parity;
    if (!parity.IsApplicable(ctx))
        return;
    parity.Apply(ctx);
    if (ctx.auto_reduced)
        ctx.applied_transforms.emplace_back(parity.Name());
    ctx.qubits_after = RegisterWidth(ctx);
}

void TransformPipeline::Tighten(TransformContext &ctx, size_t original_width, bool retry_parity)
{
    CausalConeTransform cone;
    const auto previous = ctx.circuit;
    const auto apply_cone = [&] {
        if (!cone.IsApplicable(ctx))
            return;
        const auto before = ctx.circuit;
        const auto observables = ctx.observables;
        cone.Apply(ctx);
        if (before != ctx.circuit || observables != ctx.observables)
            ctx.applied_transforms.emplace_back(cone.Name());
    };
    apply_cone();
    if (retry_parity && (previous != ctx.circuit || RegisterWidth(ctx) < original_width))
    {
        ApplyParity(ctx);
        if (ctx.auto_reduced)
            apply_cone();
    }
    ctx.qubits_after = RegisterWidth(ctx);
}

void TransformPipeline::Run(TransformContext &ctx)
{
    if (!ctx.circuit)
        throw std::invalid_argument("TransformPipeline requires a non-null circuit.");
    ctx.qubits_before = RegisterWidth(ctx);
    ctx.applied_transforms.clear();
    ctx.auto_reduced = false;
    ctx.expectation_factors.assign(ctx.observables.size(), 1.0);
    ApplyParity(ctx);
    Tighten(ctx, ctx.qubits_before, !ctx.auto_reduced);
}
} // namespace MaestroExecution
