#include "ParityReductionTransform.h"

#include "Circuit/Factory.h"
#include "Execution/SimulatorConfig.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>

namespace MaestroExecution
{
namespace
{
using Mask = uint64_t;
using Gate = Circuits::QuantumGateType;
using Factory = Circuits::CircuitFactory<double>;
using Circuit = Circuits::Circuit<double>;

// Keep the detector bounded and exact allocations modest (64 MiB at rank 22).
constexpr size_t MaxPhysicalQubits = 64;
constexpr size_t MaxEffectiveQubits = 22;

// Portable C++17 bit operations; inputs to FirstBit/LastBit are nonzero.
int Popcount(Mask value)
{
    int count = 0;
    for (; value; value &= value - 1)
        ++count;
    return count;
}

size_t FirstBit(Mask value)
{
    size_t bit = 0;
    while (!(value & 1))
    {
        value >>= 1;
        ++bit;
    }
    return bit;
}

size_t LastBit(Mask value)
{
    size_t bit = 0;
    while (value >>= 1)
        ++bit;
    return bit;
}

struct Basis
{
    std::array<Mask, MaxPhysicalQubits> rows{}, coordinates{};
    std::vector<Mask> generators;

    bool Encode(Mask value, Mask &coordinate) const
    {
        coordinate = 0;
        while (value)
        {
            const auto pivot = LastBit(value);
            if (!rows[pivot])
                return false;
            value ^= rows[pivot];
            coordinate ^= coordinates[pivot];
        }
        return true;
    }

    bool Insert(Mask value)
    {
        const auto original = value;
        Mask coordinate = 0;
        while (value)
        {
            const auto pivot = LastBit(value);
            if (!rows[pivot])
            {
                if (generators.size() == MaxEffectiveQubits)
                    return false;
                rows[pivot] = value;
                coordinates[pivot] = coordinate ^ (Mask{1} << generators.size());
                generators.push_back(original);
                return true;
            }
            value ^= rows[pivot];
            coordinate ^= coordinates[pivot];
        }
        return true;
    }

    Mask Dual(Mask value) const
    {
        Mask result = 0;
        for (size_t j = 0; j < generators.size(); ++j)
            if (Popcount(value & generators[j]) % 2)
                result |= Mask{1} << j;
        return result;
    }
};

struct Rotation
{
    Mask support;
    double angle;
    bool check;
};

void EmitRotation(Circuit &circuit, Mask support, double angle, bool x)
{
    // Empty support contributes only a global phase, irrelevant to expectations.
    if (!support || angle == 0)
        return;
    const auto target = FirstBit(support);
    if (!(support & (support - 1)))
    {
        circuit.AddOperation(Factory::CreateGate(x ? Gate::kRxGateType : Gate::kRzGateType, target, 0, 0, angle));
        return;
    }
    std::vector<size_t> qubits;
    for (auto bits = support; bits; bits &= bits - 1)
        qubits.push_back(FirstBit(bits));
    if (x)
        for (auto q : qubits)
            circuit.AddOperation(Factory::CreateGate(Gate::kHadamardGateType, q));
    for (size_t j = 1; j < qubits.size(); ++j)
        circuit.AddOperation(Factory::CreateGate(Gate::kCXGateType, qubits[j], target));
    circuit.AddOperation(Factory::CreateGate(Gate::kRzGateType, target, 0, 0, angle));
    for (size_t j = qubits.size(); j-- > 1;)
        circuit.AddOperation(Factory::CreateGate(Gate::kCXGateType, qubits[j], target));
    if (x)
        for (auto q : qubits)
            circuit.AddOperation(Factory::CreateGate(Gate::kHadamardGateType, q));
}
} // namespace

bool ParityReductionTransform::IsApplicable(const TransformContext &ctx) const
{
    return ctx.config.auto_reduce && !ctx.observables.empty() && ctx.circuit && !ctx.circuit->GetOperations().empty() &&
           !Simulators::IsDistributedGpuSimulator(ctx.simulator_type);
}

void ParityReductionTransform::Apply(TransformContext &ctx)
{
    size_t width = ctx.circuit->GetMaxQubitIndex() + 1;
    for (const auto &observable : ctx.observables)
    {
        if (observable.find_first_not_of("IXYZixyz") != std::string::npos)
            return;
        const auto last = observable.find_last_not_of("Ii");
        if (last != std::string::npos)
            width = std::max(width, last + 1);
    }
    if (width < 2 || width > MaxPhysicalQubits)
        return;

    // Recognize |+>^N preparation, or its global-H dual starting in |0>^N.
    // Requiring a uniform product eigenstate avoids assuming a symmetry sector.
    Mask prepared = 0;
    bool body = false;
    bool plus = false;
    std::array<Mask, MaxPhysicalQubits> frame{}, x_frame{};
    for (size_t q = 0; q < width; ++q)
        frame[q] = x_frame[q] = Mask{1} << q;
    Basis basis;
    std::vector<Rotation> rotations;
    for (const auto &operation : ctx.circuit->GetOperations())
    {
        if (!operation)
            return;
        if (operation->GetType() == Circuits::OperationType::kNoOp || operation->GetType() == Circuits::OperationType::kDelay)
            continue;
        // Channels, measurements, resets, nested and classically controlled ops
        // must remain on the original execution path.
        if (operation->GetType() != Circuits::OperationType::kGate)
            return;
        const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(operation);
        if (!gate)
            return;
        auto type = gate->GetGateType();
        auto q = gate->GetQubit(0);
        if (q >= width)
            return;
        if (!body && type == Gate::kHadamardGateType)
        {
            if (prepared & (Mask{1} << q))
                return;
            prepared |= Mask{1} << q;
            continue;
        }
        if (!body)
        {
            plus = Popcount(prepared) == static_cast<int>(width);
            if (prepared && !plus)
                return;
            body = true;
        }
        // F maps input bits to the current CNOT frame. Z uses rows of F,
        // while X uses columns of F^-1; both updates are exact over GF(2).
        if (type == Gate::kCXGateType)
        {
            auto control = q;
            auto target = gate->GetQubit(1);
            if (target >= width || control == target)
                return;
            if (!plus)
                std::swap(control, target);
            frame[target] ^= frame[control];
            x_frame[control] ^= x_frame[target];
            continue;
        }
        const auto append = [&](Mask support, double angle, bool check) {
            if (angle == 0)
                return true;
            if (check && !basis.Insert(support))
                return false;
            rotations.push_back({support, angle, check});
            return true;
        };
        constexpr double pi = 3.14159265358979323846;
        if (type == Gate::kCZGateType)
        {
            const auto target = gate->GetQubit(1);
            if (target >= width || target == q)
                return;
            const auto a = plus ? frame[q] : x_frame[q];
            const auto b = plus ? frame[target] : x_frame[target];
            // CZ = phase * Rz(a,pi/2) Rz(b,pi/2) Rzz(a,b,-pi/2).
            // Its global-H dual has the same decomposition about X.
            if (!append(a, pi / 2, plus) || !append(b, pi / 2, plus) || !append(a ^ b, -pi / 2, plus))
                return;
            continue;
        }
        bool z = true;
        double angle = 0;
        switch (type)
        {
        case Gate::kXGateType:
            z = false;
            angle = pi;
            break;
        case Gate::kSxGateType:
            z = false;
            angle = pi / 2;
            break;
        case Gate::kSxDagGateType:
            z = false;
            angle = -pi / 2;
            break;
        case Gate::kZGateType:
            angle = pi;
            break;
        case Gate::kSGateType:
            angle = pi / 2;
            break;
        case Gate::kSdgGateType:
            angle = -pi / 2;
            break;
        case Gate::kTGateType:
            angle = pi / 4;
            break;
        case Gate::kTdgGateType:
            angle = -pi / 4;
            break;
        case Gate::kRxGateType:
        case Gate::kRzGateType:
        case Gate::kPhaseGateType: {
            const auto params = gate->GetParams();
            if (params.size() != 1 || !std::isfinite(params[0]))
                return;
            angle = params[0];
            z = type != Gate::kRxGateType;
            break;
        }
        default:
            return;
        }
        const bool check = plus ? z : !z;
        if (!append(check ? frame[q] : x_frame[q], angle, check))
            return;
    }
    const size_t rank = basis.generators.size();
    if (rank == 0 || rank >= width)
        return;

    auto reduced = std::make_shared<Circuit>();
    for (const auto &rotation : rotations)
    {
        Mask support = 0;
        if (rotation.check)
            basis.Encode(rotation.support, support);
        else
            support = basis.Dual(rotation.support);
        EmitRotation(*reduced, support, rotation.angle, rotation.check);
    }

    std::vector<std::string> observables;
    std::vector<double> factors;
    for (const auto &observable : ctx.observables)
    {
        Mask x = 0, z = 0;
        int physical_y = 0;
        for (size_t q = 0; q < observable.size(); ++q)
        {
            const auto p = std::toupper(static_cast<unsigned char>(observable[q]));
            if (p == 'X' || p == 'Y')
                x |= Mask{1} << q;
            if (p == 'Z' || p == 'Y')
                z |= Mask{1} << q;
            physical_y += p == 'Y';
        }
        if (!plus)
            std::swap(x, z);
        // Conjugate through the remaining CNOT frame in canonical X^x Z^z
        // order. Keep the original i^#Y phase; recomputing it after the frame
        // would lose signs such as CX (Xc Zt) CX = -Yc Yt.
        Mask framed_x = 0, framed_z = 0;
        for (size_t q = 0; q < width; ++q)
        {
            if (x & (Mask{1} << q))
                framed_x ^= x_frame[q];
            if (z & (Mask{1} << q))
                framed_z ^= frame[q];
        }
        x = framed_x;
        z = framed_z;
        Mask effective_x = 0;
        const bool in_span = basis.Encode(z, effective_x);
        const Mask effective_z = basis.Dual(x);
        std::string compact(rank, 'I');
        double factor = 0;
        if (in_span)
        {
            // P = i^#Y X^x Z^z. Acting on Z^(B b)|+>, X supplies
            // (-1)^(x.z) as well as the diagonal (-1)^(B^T x).b.
            const int effective_y = Popcount(effective_x & effective_z);
            const int phase = physical_y - effective_y + 2 * Popcount(x & z) + (!plus ? 2 * physical_y : 0);
            factor = (((phase % 4) + 4) % 4 == 0) ? 1.0 : -1.0;
            for (size_t j = 0; j < rank; ++j)
            {
                const bool a = effective_x & (Mask{1} << j);
                const bool b = effective_z & (Mask{1} << j);
                compact[j] = a ? (b ? 'Y' : 'X') : (b ? 'Z' : 'I');
            }
        }
        // An observable outside the check span takes the state to an
        // orthogonal coset, so its expectation is exactly zero.
        observables.push_back(std::move(compact));
        factors.push_back(factor);
    }
    // Commit only after the whole circuit and every observable were validated.
    ctx.circuit = std::move(reduced);
    ctx.observables = std::move(observables);
    ctx.expectation_factors = std::move(factors);
    ctx.auto_reduced = true;
}
} // namespace MaestroExecution
