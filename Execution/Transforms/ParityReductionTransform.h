#pragma once

#include "ITransform.h"

namespace MaestroExecution
{
// Exact check-generated subspaces, not general Clifford symmetry discovery.
class MAESTRO_TRANSFORM_API ParityReductionTransform : public ITransform
{
  public:
    std::string_view Name() const override
    {
        return "ParityReduction";
    }

    bool IsApplicable(const TransformContext &ctx) const override;
    void Apply(TransformContext &ctx) override;
};
} // namespace MaestroExecution
