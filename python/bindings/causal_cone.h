#pragma once

#include "Circuit/Circuit.h"

namespace maestro_bindings {

// One shared cone preserves batching and backend selection for all observables.
// Unsupported operations leave both inputs unchanged. The source is never
// edited.
void ReduceCausalCone(std::shared_ptr<Circuits::Circuit<double>>& circuit,
                      std::vector<std::string>& paulis);

}  // namespace maestro_bindings
