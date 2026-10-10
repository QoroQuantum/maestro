#pragma once

#include "Export.h"
#include "ITransform.h"

namespace MaestroExecution
{
// One shared cone preserves batching and backend selection for all observables.
// Unsupported operations leave the circuit and observables unchanged.
class MAESTRO_TRANSFORM_API CausalConeTransform : public ITransform
{
  public:
    std::string_view Name() const override;
    bool IsApplicable(const TransformContext &ctx) const override;
    void Apply(TransformContext &ctx) override;
};
} // namespace MaestroExecution
