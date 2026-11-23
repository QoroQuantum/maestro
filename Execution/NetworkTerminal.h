#pragma once

#include "../Circuit/Circuit.h"
#include "Transforms/Export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace MaestroExecution
{
// Counts on the mapped network circuit. Call prepare only after establishing that the sampled
// state can be replaced by one deterministic prefix. A rejected plan leaves the register untouched.
MAESTRO_TRANSFORM_API bool SampleNetworkTerminalChannels(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                         const std::shared_ptr<Simulators::ISimulator> &simulator, size_t shots, size_t classical_width,
                                                         size_t result_width, std::optional<uint64_t> seed, const std::function<void()> &prepare,
                                                         Circuits::Circuit<double>::ExecuteResults &counts);
} // namespace MaestroExecution
