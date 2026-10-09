#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../PathIntegral/PathIntegralSimulator.h"
#include "QCSimGateSimulator.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// path_integral backend. Owns exactly one native QCSim implementation.
class QCSimPathIntegralSimulator : public QCSimGateSimulator
{
  public:
    QCSimPathIntegralSimulator()
    {
        configuration.SetConfiguration("method", "path_integral");
    }

    const char *MethodName() const override
    {
        return "path_integral";
    }

    void Initialize() override
    {
        if (nrQubits != 0)
        {
            InitializeSamplingSeed();
            {
                pathIntegralSimulator = std::make_unique<PathIntegralSimulator>();
                pathIntegralSimulator->SetStartZeroState(nrQubits);
                pathIntegralSimulator->SetSeed(samplingSeed);
            }
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void Reset() override
    {
        if (pathIntegralSimulator)
        {
            pathIntegralSimulator->Reset();
            pathIntegralSimulator->SetStartZeroState(nrQubits);
        }
        upcomingGateIndex = 0;
    }

    void Configure(const char *key, const char *value) override
    {
        if (ConfigureSampling(key, value))
            return;
        if (std::string("method") == key)
        {
            ValidateMethod(value);
            configuration.SetConfiguration(key, value);
            return;
        }
        if (!configuration.WasApplied(key, value))
            configuration.SetConfiguration(key, value);
        if (std::string("seed") == key)
        {
            const uint64_t seed = std::stoull(value);
            SeedAuxiliaryRng(seed);
            SeedSampling(seed);
            if (pathIntegralSimulator)
                pathIntegralSimulator->SetSeed(seed);
            return;
        }
        if (std::string(key) == "pauli_propagator_workers")
        {
            const auto workers = configuration.GetConfigurationAsUnsigned(key);
            if (workers > 1024)
                throw std::invalid_argument("pauli_propagator_workers exceeds 1024");
        }
        if (pathIntegralSimulator)
        {
            if (std::string(key) == "path_integral_threshold")
            {
                pathIntegralSimulator->SetTrimValue(configuration.GetConfigurationAsDouble(key));
            }
        }
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        pathIntegralSimulator = nullptr;
        nrQubits = 0;
        upcomingGateIndex = 0;
        upcomingGates.clear();
    }

    size_t Measure(const Types::qubits_vector &qubits) override
    {
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        size_t res = 0;
        size_t mask = 1ULL;
        DontNotify();
        {
            for (size_t qubit : qubits)
            {
                if (pathIntegralSimulator->MeasureQubit(qubit))
                    res |= mask;
                mask <<= 1;
            }
        }
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override
    {
        std::vector<bool> res(qubits.size(), false);
        DontNotify();
        {
            for (size_t q = 0; q < qubits.size(); ++q)
                if (pathIntegralSimulator->MeasureQubit(qubits[q]))
                    res[q] = true;
        }
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    void ApplyReset(const Types::qubits_vector &qubits) override
    {
        DontNotify();
        {
            for (size_t qubit : qubits)
                if (pathIntegralSimulator->MeasureQubit(qubit))
                {
                    QC::Gates::AppliedGate<> gate(xgate.getRawOperatorMatrix(), qubit);
                    pathIntegralSimulator->PropagateStep(gate, pathIntegralSimulator->Amplitudes());
                }
        }
        Notify();
        NotifyObservers(qubits);
    }

    double Probability(Types::qubit_t outcome) override
    {
        return pathIntegralSimulator->Probability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        return pathIntegralSimulator->AmplitudeForOutcome(outcome);
    }

    std::vector<double> AllProbabilities() override
    {
        const size_t nrBasisStates = 1ULL << GetNumberOfQubits();
        std::vector<double> result(nrBasisStates);
        for (size_t i = 0; i < nrBasisStates; ++i)
            result[i] = pathIntegralSimulator->Probability(i);
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
                result[i] = pathIntegralSimulator->Probability(qubits[i]);
        }
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (!LegacySampling())
            return SamplePrepared<false>(qubits, shots);
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        DontNotify();
        {
            if (nrQubits < 64)
            {
                if (shots > 1)
                {
                    const auto &amplitudes = pathIntegralSimulator->Amplitudes();
                    const Utils::Alias alias(amplitudes);
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const double prob = 1. - uniformZeroOne(rng);
                        const size_t measRaw = alias.Sample(prob);
                        size_t meas = 0;
                        size_t mask = 1ULL;
                        for (auto q : qubits)
                        {
                            const size_t qubitMask = 1ULL << q;
                            if ((measRaw & qubitMask) != 0)
                                meas |= mask;
                            mask <<= 1ULL;
                        }
                        ++result[meas];
                    }
                }
                else
                {
                    const size_t measRaw = MeasureNoCollapse();
                    size_t meas = 0;
                    size_t mask = 1ULL;
                    for (auto q : qubits)
                    {
                        const size_t qubitMask = 1ULL << q;
                        if ((measRaw & qubitMask) != 0)
                            meas |= mask;
                        mask <<= 1ULL;
                    }
                    ++result[meas];
                }
            }
            else
            {
                throw std::runtime_error("QCSimState::SampleCounts: The path integral simulator does not "
                                         "support sampling for more than 63 qubits into 64 bits integers.");
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (!LegacySampling())
            return SamplePrepared<true>(qubits, shots);
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        DontNotify();
        {
            if (nrQubits < 64)
            {
                if (shots > 1)
                {
                    const auto &amplitudes = pathIntegralSimulator->Amplitudes();
                    const Utils::Alias alias(amplitudes);
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const double prob = 1. - uniformZeroOne(rng);
                        const size_t measRaw = alias.Sample(prob);
                        std::vector<bool> meas(qubits.size(), false);
                        for (size_t i = 0; i < qubits.size(); ++i)
                            if (((measRaw >> qubits[i]) & 1) == 1)
                                meas[i] = true;
                        ++result[meas];
                    }
                }
                else
                {
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const auto measRaw = MeasureNoCollapseMany();
                        std::vector<bool> meas(qubits.size(), false);
                        for (size_t i = 0; i < qubits.size(); ++i)
                            if (measRaw[qubits[i]])
                                meas[i] = true;
                        ++result[meas];
                    }
                }
            }
            else
            {
                if (shots > 1)
                {
                    const auto &amplitudes = pathIntegralSimulator->Amplitudes();
                    const Utils::AliasBig alias(amplitudes);
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const double prob = 1. - uniformZeroOne(rng);
                        const auto measRaw = alias.Sample(prob);
                        std::vector<bool> meas(qubits.size(), false);
                        for (size_t i = 0; i < qubits.size(); ++i)
                            if (measRaw.get(qubits[i]))
                                meas[i] = true;
                        ++result[meas];
                    }
                }
                else
                {
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const auto measRaw = MeasureNoCollapseMany();
                        std::vector<bool> meas(qubits.size(), false);
                        for (size_t i = 0; i < qubits.size(); ++i)
                            if (measRaw[qubits[i]])
                                meas[i] = true;
                        ++result[meas];
                    }
                }
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    double ExpectationValue(const std::string &pauliStringOrig) override
    {
        if (pauliStringOrig.empty())
            return 1.0;
        std::string pauliString = pauliStringOrig;
        if (pauliString.size() > GetNumberOfQubits())
        {
            for (size_t i = GetNumberOfQubits(); i < pauliString.size(); ++i)
            {
                const auto pauliOp = toupper(pauliString[i]);
                if (pauliOp != 'I' && pauliOp != 'Z')
                    return 0.0;
            }
            pauliString.resize(GetNumberOfQubits());
        }
        return pathIntegralSimulator->ExpectationValue(pauliString);
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kPathIntegral;
    }

    void SaveState() override
    {
        pathIntegralSimulator->SaveState();
    }

    void RestoreState() override
    {
        pathIntegralSimulator->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (pathIntegralSimulator)
        {
            enableMultithreading = false;
        }
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        {
            if (nrQubits < 64)
            {
                const auto measured = pathIntegralSimulator->MeasureNoCollapse();
                Types::qubit_t result = 0;
                Types::qubit_t mask = 1;
                for (Types::qubit_t q = 0; q < measured.size(); ++q)
                {
                    if (measured.get(q))
                        result |= mask;
                    mask <<= 1;
                }
                return result;
            }
            else
            {
                throw std::runtime_error("QCSimState::MeasureNoCollapse: The path integral simulator does "
                                         "not "
                                         "support measuring more than 63 qubits into 64 bits integers.");
            }
        }
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        const auto measured = pathIntegralSimulator->MeasureNoCollapse();
        std::vector<bool> res(nrQubits);
        for (size_t i = 0; i < nrQubits; ++i)
            res[i] = measured.get(i);
        return res;
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        pgate.SetPhaseShift(lambda);
        {
            QC::Gates::AppliedGate<> agate(pgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(xgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(ygate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(zgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(h.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(sgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(sdggate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(tgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(tdggate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(sxgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(sxdaggate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(k.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        rxgate.SetTheta(theta);
        {
            QC::Gates::AppliedGate<> agate(rxgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        rygate.SetTheta(theta);
        {
            QC::Gates::AppliedGate<> agate(rygate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        rzgate.SetTheta(theta);
        {
            QC::Gates::AppliedGate<> agate(rzgate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        ugate.SetParams(theta, phi, lambda, gamma);
        {
            QC::Gates::AppliedGate<> agate(ugate.getRawOperatorMatrix(), qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(cxgate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(cygate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(czgate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        cpgate.SetPhaseShift(lambda);
        {
            QC::Gates::AppliedGate<> agate(cpgate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crxgate.SetTheta(theta);
        {
            QC::Gates::AppliedGate<> agate(crxgate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crygate.SetTheta(theta);
        {
            QC::Gates::AppliedGate<> agate(crygate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crzgate.SetTheta(theta);
        {
            QC::Gates::AppliedGate<> agate(crzgate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(ch.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(csx.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        {
            QC::Gates::AppliedGate<> agate(csxdag.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        {
            QC::Gates::AppliedGate<> agate(swapgate.getRawOperatorMatrix(), qubit1, qubit0);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        QC::Gates::AppliedGate<> agate(ccxgate.getRawOperatorMatrix(), qubit2, qubit1, qubit0);
        pathIntegralSimulator->ApplyGate(agate);
        NotifyObservers({qubit2, qubit1, qubit0});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        QC::Gates::AppliedGate<> agate(cswapgate.getRawOperatorMatrix(), qubit1, qubit0, ctrl_qubit);
        pathIntegralSimulator->ApplyGate(agate);
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        cugate.SetParams(theta, phi, lambda, gamma);
        {
            QC::Gates::AppliedGate<> agate(cugate.getRawOperatorMatrix(), tgt_qubit, ctrl_qubit);
            pathIntegralSimulator->ApplyGate(agate);
        }
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        return CloneForExecution(NextCloneSeed());
    }

    std::unique_ptr<ISimulator> CloneForExecution(uint64_t seed) override
    {
        auto cloned = std::make_unique<QCSimPathIntegralSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        if (pathIntegralSimulator)
            cloned->pathIntegralSimulator = pathIntegralSimulator->Clone();
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        cloned->SetSeed(seed);
        return cloned;
    }

  protected:
    std::unique_ptr<PathIntegralSimulator> pathIntegralSimulator;

  private:
    template <bool Many> Utils::Sampling::Counts<Many> SamplePrepared(const Types::qubits_vector &qubits, size_t shots)
    {
        if (qubits.empty() || !shots)
            return {};
        Utils::Sampling::ValidateQubits(qubits, nrQubits, Many);
        if (shots == 1)
        {
            // Native direct sampling already normalizes the retained mass.
            // Project from the full label, including packed subsets above bit 63.
            const auto row = pathIntegralSimulator->MeasureNoCollapse();
            auto result = Utils::Sampling::Single<Many>(qubits.size(), [&](size_t bit) { return row.get(qubits[bit]); });
            NotifyObservers(qubits);
            return result;
        }
        const auto &amplitudes = pathIntegralSimulator->Amplitudes();
        auto options = SamplingOptions();
        const size_t count = amplitudes.size();
        if (count > (options.memoryBytes - 4096) / (sizeof(double) + sizeof(const uint64_t *)))
        {
            // AmplitudeMap only exposes sequential iteration. Sweep bounded
            // target batches when a random-access snapshot would exceed the
            // budget, borrowing the original full-width labels as we visit them.
            auto result = Utils::Sampling::CountCdfRange<Many>(
                amplitudes, shots, qubits.size(), [](const auto &row) { return std::norm(row.second); },
                [&](const auto &row, size_t bit) { return row.first.get(qubits[bit]); },
                [&] { return Utils::RandomStream::Uniform(rng); }, options);
            NotifyObservers(qubits);
            return result;
        }
        options.memoryBytes -= count * (sizeof(double) + sizeof(const uint64_t *));
        std::vector<double> probabilities;
        std::vector<const uint64_t *> labels;
        probabilities.reserve(count);
        labels.reserve(count);
        // Iteration follows AmplitudeMap's stored row order, preserved by copy.
        // Labels borrow immutable storage for this batch only, never across a
        // mutation, restore or clone; no wide labels are copied into alias rows.
        for (const auto &entry : amplitudes)
        {
            probabilities.push_back(std::norm(entry.second));
            labels.push_back(entry.first.getWords());
        }
        const Utils::Sampling::Prepared plan(count, shots, [&](size_t row) { return probabilities[row]; }, options);
        auto result = Utils::Sampling::Count<Many>(
            plan, shots, qubits.size(),
            [&](size_t row, size_t bit) { const auto q = qubits[bit]; return ((labels[row][q / 64] >> (q % 64)) & 1) != 0; },
            [&] { return Utils::RandomStream::Uniform(rng); });
        NotifyObservers(qubits);
        return result;
    }
};
} // namespace Simulators::Private
#endif
