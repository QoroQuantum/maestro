#pragma once
#include "QuantumGate.h"

namespace Simulators
{
inline bool IsControlledRoutingGate(Circuits::QuantumGateType type)
{
    switch (type)
    {
    case Circuits::QuantumGateType::kCXGateType:
    case Circuits::QuantumGateType::kCYGateType:
    case Circuits::QuantumGateType::kCZGateType:
    case Circuits::QuantumGateType::kCPGateType:
    case Circuits::QuantumGateType::kCRxGateType:
    case Circuits::QuantumGateType::kCRyGateType:
    case Circuits::QuantumGateType::kCRzGateType:
    case Circuits::QuantumGateType::kCHGateType:
    case Circuits::QuantumGateType::kCSxGateType:
    case Circuits::QuantumGateType::kCSxDagGateType:
    case Circuits::QuantumGateType::kCUGateType:
        return true;
    default:
        return false;
    }
}

// Metadata for non-executable routing operations emitted by gate fusion.
struct RoutingGateInfo
{
    virtual ~RoutingGateInfo() = default;
    int schmidtRank = 4; // State-vector rank; the MPO model squares it.
    bool isSwap = false;
};
} // namespace Simulators
