#pragma once
#include "MPSSimulator.h"
#include <functional>
#include <type_traits>

namespace Simulators::Private
{
// Keep compatibility with the pinned QCSim dependency while allowing local
// and newer QCSim builds to use the summary and batch APIs.
template <typename T, typename = void> struct HasBondSummary : std::false_type
{
};

template <typename T>
struct HasBondSummary<T, std::void_t<decltype(std::declval<T &>().SetBondDimensionSummaryCallback(std::declval<std::function<void(Eigen::Index)>>()))>>
    : std::true_type
{
};

template <typename T, typename Callback> void InstallBondSummary(T &backend, Callback callback)
{
    if constexpr (HasBondSummary<T>::value)
    {
        backend.SetBondDimensionSummaryCallback(std::move(callback));
    }
    else
    {
        backend.SetBondDimensionCallback([callback = std::move(callback)](const auto &dims) {
            Eigen::Index maximum = 1;
            for (auto dim : dims)
                maximum = std::max(maximum, dim);
            callback(maximum);
        });
    }
}

template <typename T, typename = void> struct HasTensorExpectationBatch : std::false_type
{
};

template <typename T>
struct HasTensorExpectationBatch<T, std::void_t<decltype(std::declval<T &>().ExpectationValues(std::declval<const std::vector<std::string> &>()))>>
    : std::true_type
{
};

template <typename T, typename = void> struct HasNormalizedTensorExpectationBatch : std::false_type
{
};

template <typename T>
struct HasNormalizedTensorExpectationBatch<
    T, std::void_t<decltype(std::declval<const T &>().ExpectationValues(std::declval<const std::vector<std::string> &>(), true))>> : std::true_type
{
};

template <typename T> std::vector<std::complex<double>> ComplexTensorExpectationBatch(const T &backend, const std::vector<std::string> &paulis, bool normalized)
{
    if constexpr (HasNormalizedTensorExpectationBatch<T>::value)
    {
        return backend.ExpectationValues(paulis, normalized);
    }
    else
    {
        std::vector<std::complex<double>> values;
        values.reserve(paulis.size());
        for (const auto &pauli : paulis)
            values.push_back(normalized ? backend.ExpectationValue(pauli) : backend.UnnormalizedExpectationValue(pauli));
        return values;
    }
}

template <typename T> std::vector<std::complex<double>> TensorExpectationBatch(T &backend, const std::vector<std::string> &paulis)
{
    if constexpr (HasTensorExpectationBatch<T>::value)
    {
        return backend.ExpectationValues(paulis);
    }
    else
    {
        std::vector<std::complex<double>> values;
        values.reserve(paulis.size());
        for (const auto &pauli : paulis)
        {
            if constexpr (std::is_same_v<T, QC::TensorNetworks::MPSSimulator>)
            {
                static const QC::Gates::PauliXGate<> x;
                static const QC::Gates::PauliYGate<> y;
                static const QC::Gates::PauliZGate<> z;
                std::vector<QC::Gates::AppliedGate<>> gates;
                for (size_t q = 0; q < pauli.size(); ++q)
                {
                    if (pauli[q] == 'X')
                        gates.emplace_back(x.getRawOperatorMatrix(), q);
                    if (pauli[q] == 'Y')
                        gates.emplace_back(y.getRawOperatorMatrix(), q);
                    if (pauli[q] == 'Z')
                        gates.emplace_back(z.getRawOperatorMatrix(), q);
                }
                values.push_back(backend.ExpectationValue(gates));
            }
            else
            {
                values.push_back(backend.ExpectationValue(pauli));
            }
        }
        return values;
    }
}

} // namespace Simulators::Private
