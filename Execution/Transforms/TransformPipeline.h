#pragma once

#include "Export.h"
#include "TransformContext.h"

namespace MaestroExecution
{
// Run before simulator allocation or network configuration. New passes belong
// in the ordered registry in TransformPipeline.cpp, not in callers or backends.
class MAESTRO_TRANSFORM_API TransformPipeline
{
  public:
    static void Run(TransformContext &ctx);
};
} // namespace MaestroExecution
