#pragma once
#include "Transforms/Export.h"
#include <boost/json.hpp>
#include <cstdint>
#include <vector>

namespace Simulators
{
class QuantumChannel;
}

namespace MaestroExecution
{
constexpr unsigned SchemaVersion = 2;
constexpr size_t MaxRequestBytes = 16 * 1024 * 1024;
constexpr size_t MaxResultBytes = 64 * 1024 * 1024;

// Row-major shot records of an execute request with return_samples: shots rows
// of width bytes (0 or 1), classical bit 0 first. Run fills it instead of
// encoding "samples" in its result, so the rows escape the JSON size bound.
struct ShotRows
{
    std::vector<uint8_t> data;
    size_t shots = 0;
    size_t width = 0;
};

boost::json::object Capabilities();
// Shared by versioned requests and the legacy SimpleExecute/SimpleEstimate instruction arrays.
MAESTRO_TRANSFORM_API Simulators::QuantumChannel ParseInstructionChannel(const boost::json::object &instruction, size_t width);
boost::json::object Run(const boost::json::object &request, bool validate = false, unsigned depth = 0, ShotRows *rows = nullptr);
} // namespace MaestroExecution
