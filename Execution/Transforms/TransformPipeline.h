#pragma once

#include "Export.h"
#include "TransformContext.h"

namespace MaestroExecution
{
// Single-group parity/cone composition, before simulator allocation. Batched
// execution uses PrepareEstimation for global parity followed by local groups.
class MAESTRO_TRANSFORM_API TransformPipeline
{
  public:
    // Production preparation and standalone callers share these phases.
    static void ApplyParity(TransformContext &ctx);
    static void Tighten(TransformContext &ctx, size_t original_width, bool retry_parity);
    static void Run(TransformContext &ctx);
};
} // namespace MaestroExecution
