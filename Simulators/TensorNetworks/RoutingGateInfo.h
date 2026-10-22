#pragma once
#include "QuantumGate.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace Simulators
{
// Effective operator Schmidt rank of a two-qubit unitary: the participation
// ratio (sum s^2)^2 / sum s^4 of its operator Schmidt coefficients s. It is 1
// for a product of one-qubit gates, 2 for CX or CZ and 4 for SWAP, and it falls
// smoothly to 1 as a rotation angle goes to 0. Routing uses it as the factor by
// which a gate can grow the bond it acts on. Local one-qubit gates on either
// side leave it unchanged.
inline double EffectiveSchmidtRank(const Eigen::MatrixXcd &u)
{
    if (u.rows() != 4 || u.cols() != 4 || !u.allFinite())
        return 4.;
    // Realign: rows pair one qubit's output and input index, columns the other's.
    Eigen::Matrix4cd r;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c)
                for (int d = 0; d < 2; ++d)
                    r(2 * a + c, 2 * b + d) = u(2 * a + b, 2 * c + d);
    const Eigen::Vector4d s = Eigen::JacobiSVD<Eigen::Matrix4cd>(r).singularValues();
    const double s2 = s.squaredNorm(), s4 = s.array().pow(4).sum();
    return s4 > 0. ? std::clamp(s2 * s2 / s4, 1., 4.) : 4.;
}

// A controlled-U gate has operator Schmidt coefficients s^2 = 2 +- |tr U|.
inline double ControlledSchmidtRank(double traceAbs)
{
    return 8. / (4. + traceAbs * traceAbs);
}

// Closed forms for named gates; matrix() is only called for the others.
// angle is the gate's first parameter (ignored by unparameterized gates).
template <class MatrixFn> double RoutingSchmidtRank(Circuits::QuantumGateType type, double angle, MatrixFn &&matrix)
{
    switch (type)
    {
    case Circuits::QuantumGateType::kSwapGateType:
        return 4.;
    case Circuits::QuantumGateType::kCXGateType:
    case Circuits::QuantumGateType::kCYGateType:
    case Circuits::QuantumGateType::kCZGateType:
    case Circuits::QuantumGateType::kCHGateType:
        return 2.;
    case Circuits::QuantumGateType::kCSxGateType:
    case Circuits::QuantumGateType::kCSxDagGateType:
        return ControlledSchmidtRank(std::sqrt(2.)); // |tr SX| = |1 + i|
    case Circuits::QuantumGateType::kCPGateType:     // |1 + exp(i angle)|
    case Circuits::QuantumGateType::kCRxGateType:    // |tr Rx(angle)|, likewise Ry, Rz
    case Circuits::QuantumGateType::kCRyGateType:
    case Circuits::QuantumGateType::kCRzGateType:
        return ControlledSchmidtRank(2. * std::abs(std::cos(angle / 2.)));
    default:
        return EffectiveSchmidtRank(matrix());
    }
}

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
    double schmidtRank = 4; // Effective state-vector rank; the MPO model squares it.
    bool isSwap = false;
};
} // namespace Simulators
