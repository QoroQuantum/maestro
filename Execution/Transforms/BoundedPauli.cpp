#include "BoundedPauli.h"
#include "ConeAnalysis.h"

#include <cctype>
#include <cmath>
#include <limits>
#include <map>

namespace MaestroExecution
{
namespace
{
constexpr size_t MaxTerms = 1024;
constexpr size_t MaxObservableWork = 32768;
using Word = std::vector<uint64_t>; // Two bits per cone-local qubit.

size_t At(const Word &p, size_t q)
{
    return (p[q / 32] >> (2 * (q % 32))) & 3;
}

void Set(Word &p, size_t q, size_t axis)
{
    const auto shift = 2 * (q % 32);
    p[q / 32] = (p[q / 32] & ~(uint64_t{3} << shift)) | (static_cast<uint64_t>(axis) << shift);
}

size_t Axis(char p)
{
    return p == 'X' ? 1 : p == 'Y' ? 2 : p == 'Z' ? 3 : 0;
}

Eigen::MatrixXcd PauliMatrix(size_t word, size_t qubits)
{
    const size_t dim = size_t{1} << qubits;
    Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(dim, dim);
    for (size_t col = 0; col < dim; ++col)
    {
        size_t row = col;
        std::complex<double> phase = 1;
        for (size_t q = 0; q < qubits; ++q)
        {
            const auto p = (word >> (2 * q)) & 3;
            const bool bit = (col >> q) & 1;
            if (p == 1 || p == 2)
                row ^= size_t{1} << q;
            if (p == 2)
                phase *= std::complex<double>(0, bit ? -1 : 1);
            if (p == 3 && bit)
                phase = -phase;
        }
        matrix(row, col) = phase;
    }
    return matrix;
}
} // namespace

std::optional<BoundedExpectation> EvaluateBoundedPauli(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::string &observable,
                                                       double tolerance, BoundedPauliWorkspace *workspace)
{
    if (!std::isfinite(tolerance) || tolerance <= 0 || tolerance > 1)
        throw std::invalid_argument("observable_error_tolerance must be finite and in (0, 1]");
    BoundedPauliWorkspace owned;
    if (!workspace)
        workspace = &owned;
    if (workspace->work >= workspace->work_limit || !CanTraceCone(circuit, {observable}, true))
        return std::nullopt;
    const auto cone = TraceStructuralCone(circuit, {observable});
    const auto &ops = circuit->GetOperations();
    if (std::none_of(cone.operations.begin(), cone.operations.end(), [&](size_t i) { return ops[i]->GetType() == Circuits::OperationType::kQuantumChannel; }))
        return std::nullopt;
    const size_t start_work = workspace->work;
    Word word((cone.mapping.size() + 31) / 32);
    for (const auto &[original, compact] : cone.mapping)
        if (original < observable.size())
            Set(word, compact, Axis(static_cast<char>(std::toupper(static_cast<unsigned char>(observable[original])))));
    const Word identity(word.size(), 0);
    Word support = word;
    std::map<Word, double> terms{{word, 1}};
    double constant = 0, error = 0;
    for (auto it = cone.operations.rbegin(); it != cone.operations.rend(); ++it)
    {
        const auto &op = ops[*it];
        if (op->GetType() == Circuits::OperationType::kNoOp || op->GetType() == Circuits::OperationType::kDelay)
            continue;
        auto qs = op->AffectedQubits();
        for (auto &q : qs)
            q = cone.mapping.at(q);
        if (std::none_of(qs.begin(), qs.end(), [&](size_t q) { return At(support, q); }))
            continue;
        if (qs.size() > 3 || workspace->work >= workspace->work_limit || workspace->work - start_work >= MaxObservableWork)
            return std::nullopt;
        std::vector<Eigen::MatrixXcd> kraus;
        if (const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op))
        {
            kraus.push_back(gate->GetMatrix());
            // Named gate matrices order controls most significantly; channels
            // use target[0] as the least-significant local bit.
            if (qs.size() > 1 && gate->GetGateType() != Circuits::QuantumGateType::kSwapGateType && gate->GetGateType() != Circuits::QuantumGateType::kNone)
                std::reverse(qs.begin(), qs.end());
        }
        else if (const auto channel = std::dynamic_pointer_cast<Circuits::QuantumChannelOperation<double>>(op))
            kraus = channel->GetChannel().GetKrausOperators();
        else
            return std::nullopt;
        const size_t dim = size_t{1} << qs.size(), basis_size = dim * dim;
        std::vector<Eigen::MatrixXcd> basis;
        auto &transfers = workspace->transfers[op.get()];
        std::map<Word, double> next;
        for (const auto &[p, coefficient] : terms)
        {
            size_t local = 0;
            for (size_t j = 0; j < qs.size(); ++j)
                local |= At(p, qs[j]) << (2 * j);
            if (!local)
            {
                next[p] += coefficient; // E^dagger(I) = I for every CPTP map.
                continue;
            }
            if (basis_size > workspace->work_limit - workspace->work || workspace->work - start_work + basis_size > MaxObservableWork)
                return std::nullopt;
            workspace->work += basis_size;
            auto found = transfers.find(local);
            if (found == transfers.end())
            {
                if (basis.empty())
                    for (size_t b = 0; b < basis_size; ++b)
                        basis.push_back(PauliMatrix(b, qs.size()));
                Eigen::MatrixXcd evolved = Eigen::MatrixXcd::Zero(dim, dim);
                for (const auto &k : kraus)
                    evolved.noalias() += k.adjoint() * basis[local] * k;
                std::vector<double> transfer(basis_size);
                for (size_t b = 0; b < basis_size; ++b)
                    transfer[b] = (basis[b] * evolved).trace().real() / static_cast<double>(dim);
                found = transfers.emplace(local, std::move(transfer)).first;
            }
            // Reserve a conservative floating-point margin in addition to the
            // mathematical tail bound. No coefficient is silently discarded.
            error += std::abs(coefficient) * 256 * basis_size * dim * kraus.size() * std::numeric_limits<double>::epsilon();
            for (size_t b = 0; b < basis_size; ++b)
            {
                const auto value = coefficient * found->second[b];
                if (std::abs(value) <= 1e-14)
                {
                    error += std::abs(value);
                    continue;
                }
                auto expanded = p;
                for (size_t j = 0; j < qs.size(); ++j)
                    Set(expanded, qs[j], (b >> (2 * j)) & 3);
                next[expanded] += value;
            }
            if (next.size() > MaxTerms || error > tolerance)
                return std::nullopt;
        }
        terms = std::move(next);
        if (const auto found = terms.find(identity); found != terms.end())
        {
            constant += found->second;
            terms.erase(found);
        }
        double tail = 0;
        std::fill(support.begin(), support.end(), uint64_t{0});
        for (const auto &[p, coefficient] : terms)
        {
            tail += std::abs(coefficient);
            for (size_t w = 0; w < p.size(); ++w)
                support[w] |= p[w];
        }
        // ||sum a_P P|| <= sum |a_P|. Every earlier CPTP adjoint is unital
        // and contracts the operator norm, so this bound survives ANY prefix.
        if (tail + error <= tolerance)
            return BoundedExpectation{constant, tail + error};
    }
    // A small expansion can also be evaluated directly on the |0> input.
    for (const auto &[p, coefficient] : terms)
    {
        bool diagonal = true;
        for (size_t q = 0; q < cone.mapping.size(); ++q)
            diagonal = diagonal && (At(p, q) == 0 || At(p, q) == 3);
        if (diagonal)
            constant += coefficient;
    }
    return BoundedExpectation{constant, error};
}
} // namespace MaestroExecution
