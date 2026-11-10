/**
 * @file Alias.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Alias sampling for O(1) sampling with a O(N) preprocessing step.
 */

#pragma once

#ifndef _ALIAS_H_
#define _ALIAS_H_

#include "AliasBase.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigen>

#include "PathIntegral.h"
#include <type_traits>
#include <utility>

namespace Utils
{

using PathIntegralAmplitudeMap = std::remove_reference_t<decltype(std::declval<QC::PathIntegral::PathIntegralSimulator &>().GetAmplitudes())>;

// The retained mass of a path-integral state, weighed from amplitudes scaled
// by 2^shift before squaring.
struct PathIntegralMassScale
{
    double total = 0.;
    int shift = 0;
};

inline std::complex<double> ScalePathIntegralAmplitude(const std::complex<double> &amplitude, int shift)
{
    return {std::ldexp(amplitude.real(), shift), std::ldexp(amplitude.imag(), shift)};
}

// Path-integral pruning can leave a non-unit norm. Sample conditionally
// on the finite, positive retained mass, just as direct measurement does.
// As in QCSim's direct measurement, a total below 2^-900, or one that
// overflows, is weighed again from amplitudes scaled by the exact power of
// two that brings the largest component into [0.5, 1): squares below the
// smallest double would vanish, and weights divided by a tiny total lose
// precision. Ordinary states keep their unscaled weights.
inline PathIntegralMassScale PathIntegralMass(const PathIntegralAmplitudeMap &amplitudes)
{
    PathIntegralMassScale mass;
    for (const auto &entry : amplitudes)
        mass.total += std::norm(entry.second);
    if (!(mass.total >= 0x1p-900 && mass.total <= std::numeric_limits<double>::max()))
    {
        double largest = 0.;
        for (const auto &entry : amplitudes)
        {
            const double re = std::abs(entry.second.real()), im = std::abs(entry.second.imag());
            if (re > largest)
                largest = re;
            if (im > largest)
                largest = im;
        }
        if (largest > 0. && std::isfinite(largest))
        {
            int exponent = 0;
            std::frexp(largest, &exponent);
            mass.shift = -exponent;
            mass.total = 0.;
            for (const auto &entry : amplitudes)
                mass.total += std::norm(ScalePathIntegralAmplitude(entry.second, mass.shift));
        }
    }
    if (!(mass.total > 0.) || !std::isfinite(mass.total))
        throw std::domain_error("Path integral sampling requires finite positive probability mass");
    return mass;
}

// Calls build with the weight function of mass: plain squares for an
// ordinary state, so its table is built exactly as before.
template <class Build> void BuildPathIntegralTable(const PathIntegralMassScale &mass, const Build &build)
{
    if (mass.shift)
        build([shift = mass.shift](const std::complex<double> &amplitude) { return std::norm(ScalePathIntegralAmplitude(amplitude, shift)); });
    else
        build([](const std::complex<double> &amplitude) { return std::norm(amplitude); });
}

class Alias : public AliasBase
{
  public:
    Alias() = delete;

    template <class T = Eigen::VectorXcd> Alias(const T &statevector)
    {
        size_t numNonZeroStates = 0;
        bool contiguousStates = true;
        double accum = 0.;
        for (Eigen::Index state = 0; state < static_cast<Eigen::Index>(statevector.size()); ++state)
        {
            const double stateProb = std::norm(statevector[state]);
            if (stateProb < std::numeric_limits<double>::epsilon())
                continue;

            contiguousStates = contiguousStates && static_cast<size_t>(state) == numNonZeroStates;
            ++numNonZeroStates;

            accum += stateProb;
            if (accum > oneMinusEps)
                break;
        }

        // Dense states (and contiguous prefixes) map table positions directly
        // to outcomes. Keep the full-width mapping only for sparse support.
        if (!contiguousStates)
            statesTable.reserve(numNonZeroStates);

        SetAliasTable(numNonZeroStates, [&](const auto &add) {
            size_t row = 0;
            double sum = 0.;
            for (Eigen::Index state = 0; state < static_cast<Eigen::Index>(statevector.size()); ++state)
            {
                const double stateProb = std::norm(statevector[state]);
                if (stateProb < std::numeric_limits<double>::epsilon())
                    continue;

                add(row++, stateProb * numNonZeroStates);
                if (!contiguousStates)
                    statesTable.push_back(state);

                sum += stateProb;
                if (sum > oneMinusEps)
                    break;
            }
        });
    }

    Alias(const PathIntegralAmplitudeMap &amplitudesMap)
    {
        const auto mass = PathIntegralMass(amplitudesMap);
        const double total = mass.total;
        statesTable.reserve(amplitudesMap.size());
        BuildPathIntegralTable(mass, [&](const auto &weight) {
            SetAliasTable(amplitudesMap.size(), [&](const auto &add) {
                size_t row = 0;
                for (const auto &entry : amplitudesMap)
                {
                    add(row++, (weight(entry.second) / total) * amplitudesMap.size());
                    statesTable.push_back(entry.first.getWords()[0]);
                }
            });
        });
    }

    inline size_t Sample(double v) const
    {
        const size_t row = SampleIndex(v);
        return statesTable.empty() ? row : statesTable[row];
    }

  private:
    std::vector<uint64_t> statesTable;
};

class AliasBig : public AliasBase
{
  public:
    AliasBig() = delete;

    AliasBig(const PathIntegralAmplitudeMap &amplitudesMap)
    {
        const auto mass = PathIntegralMass(amplitudesMap);
        const double total = mass.total;
        qubitCount = amplitudesMap.QubitCount();
        wordsPerState = (qubitCount + 63) / 64;
        if (wordsPerState && amplitudesMap.size() > statesTable.max_size() / wordsPerState)
            throw std::length_error("Alias state labels exceed maximum storage size");
        // All keys share a width. Store just their active words instead of a
        // fixed 1024-bit FastVectorBool per entry, without truncating high bits.
        statesTable.reserve(amplitudesMap.size() * wordsPerState);
        BuildPathIntegralTable(mass, [&](const auto &weight) {
            SetAliasTable(amplitudesMap.size(), [&](const auto &add) {
                size_t row = 0;
                for (const auto &entry : amplitudesMap)
                {
                    add(row, (weight(entry.second) / total) * amplitudesMap.size());
                    if (wordsPerState)
                        statesTable.insert(statesTable.end(), entry.first.getWords(), entry.first.getWords() + wordsPerState);
                    ++row;
                }
            });
        });
    }

    inline QC::PathIntegral::FastVectorBool Sample(double v) const
    {
        const size_t row = SampleIndex(v);
        if (!wordsPerState)
            return QC::PathIntegral::FastVectorBool();
        return QC::PathIntegral::FastVectorBool(statesTable.data() + row * wordsPerState, qubitCount);
    }

  private:
    std::vector<uint64_t> statesTable;
    size_t qubitCount = 0;
    size_t wordsPerState = 0;
};

} // namespace Utils

#endif // _ALIAS_H_
