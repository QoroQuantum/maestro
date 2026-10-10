#pragma once

#include "Export.h"
#include "TransformContext.h"

#include <string_view>

namespace MaestroExecution
{
class MAESTRO_TRANSFORM_API ITransform
{
  public:
    virtual ~ITransform() = default;
    virtual std::string_view Name() const = 0;
    virtual bool IsApplicable(const TransformContext &ctx) const = 0;
    virtual void Apply(TransformContext &ctx) = 0;
};
} // namespace MaestroExecution
