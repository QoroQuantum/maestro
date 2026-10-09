#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "ImmediateQCSimSimulator.h"
#include "QcsimPauliPropagator.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// pauli_propagator backend. Owns exactly one native QCSim implementation.
class QCSimPauliPropagatorSimulator : public ImmediateQCSimSimulator
{
  public:
    QCSimPauliPropagatorSimulator()
    {
        configuration.SetConfiguration("method", "pauli_propagator");
    }

    const char *MethodName() const override
    {
        return "pauli_propagator";
    }

    void Initialize() override
    {
        if (nrQubits != 0)
        {
            {
                pp = std::make_unique<Simulators::QcsimPauliPropagator>();
                pp->SetNrQubits(static_cast<int>(nrQubits));
            }
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void Reset() override
    {
        if (pp)
            pp->ClearOperations();
        upcomingGateIndex = 0;
    }

    void Configure(const char *key, const char *value) override
    {
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
            nextSeedStream = 0;
            rng.seed(seed);
            if (pp)
                SeedBackend(pp.get(), seed);
            return;
        }
        if (std::string(key) == "pauli_propagator_workers")
        {
            const auto workers = configuration.GetConfigurationAsUnsigned(key);
            if (workers > 1024)
                throw std::invalid_argument("pauli_propagator_workers exceeds 1024");
            pauliWorkerCount = static_cast<size_t>(workers);
            if (pp && enableMultithreading)
                pp->EnableParallel(pauliWorkerCount);
        }
        if (pp)
        {
            if (std::string(key) == "pauli_propagator_sampling_cache_nodes")
            {
                pp->SetSamplingCacheMaxNodes(configuration.GetConfigurationAsUnsigned(key));
            }
            else if (std::string(key) == "pauli_propagator_coefficient_threshold")
            {
                pp->SetCoefficientThreshold(configuration.GetConfigurationAsDouble(key));
            }
            else if (std::string(key) == "pauli_propagator_pauli_weight_threshold")
            {
                pp->SetPauliWeightThreshold(configuration.GetConfigurationAsUnsigned(key));
            }
            else if (std::string(key) == "pauli_propagator_steps_between_trims")
            {
                pp->SetStepsBetweenTrims(configuration.GetConfigurationAsInt(key));
            }
            else if (std::string(key) == "pauli_propagator_num_gates_between_deduplications")
            {
                pp->SetStepsBetweenDeduplication(configuration.GetConfigurationAsInt(key));
            }
        }
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        if (pp)
            pp->SetNrQubits(static_cast<int>(nrQubits));
        return oldNrQubits;
    }

    void Clear() override
    {
        pp = nullptr;
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
        size_t mask = 1ULL;
        DontNotify();
        {
            std::vector<int> qubitsInt;
            qubitsInt.reserve(qubits.size());
            for (const auto q : qubits)
                qubitsInt.push_back(static_cast<int>(q));
            const auto res = pp->Measure(qubitsInt);
            Types::qubit_t result = 0;
            for (size_t i = 0; i < res.size(); ++i)
            {
                if (res[i])
                    result |= mask;
                mask <<= 1;
            }
            return result;
        }
    }

    std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override
    {
        std::vector<bool> res(qubits.size(), false);
        DontNotify();
        {
            std::vector<int> qubitsInt(qubits.begin(), qubits.end());
            res = pp->Measure(qubitsInt);
        }
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    void ApplyReset(const Types::qubits_vector &qubits) override
    {
        DontNotify();
        {
            std::vector<int> qubitsInt(qubits.begin(), qubits.end());
            const auto res = pp->Measure(qubitsInt);
            for (size_t i = 0; i < res.size(); ++i)
            {
                if (res[i])
                    pp->ApplyX(qubitsInt[i]);
            }
        }
        Notify();
        NotifyObservers(qubits);
    }

    double Probability(Types::qubit_t outcome) override
    {
        return pp->Probability(outcome);
    }

    std::vector<double> AllProbabilities() override
    {
        const size_t nrBasisStates = 1ULL << GetNumberOfQubits();
        std::vector<double> result(nrBasisStates);
        for (size_t i = 0; i < nrBasisStates; ++i)
            result[i] = pp->Probability(i);
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
                result[i] = pp->Probability(qubits[i]);
        }
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        DontNotify();
        {
            std::vector<int> qubitsInt(qubits.begin(), qubits.end());
            if (qubits.size() > 64)
                throw std::invalid_argument("Use SampleCountsMany for more than 64 measured qubits");
            const auto counts = pp->SampleCounts(qubitsInt, shots);
            for (const auto &[bits, count] : counts)
            {
                size_t meas = 0;
                for (size_t i = 0; i < bits.size(); ++i)
                    if (bits[i])
                        meas |= (1ULL << i);
                result[meas] += count;
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        DontNotify();
        {
            std::vector<int> qubitsInt(qubits.begin(), qubits.end());
            const auto counts = pp->SampleCounts(qubitsInt, shots);
            for (const auto &[bits, count] : counts)
                result[bits] += count;
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
        return pp->ExpectationValue(pauliString);
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kPauliPropagator;
    }

    void SaveState() override
    {
        pp->SaveState();
    }

    void RestoreState() override
    {
        pp->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (pp)
        {
            if (multithreading)
                pp->EnableParallel(pauliWorkerCount);
            else
                pp->DisableParallel();
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
            std::vector<int> qubitsInt(GetNumberOfQubits());
            std::iota(qubitsInt.begin(), qubitsInt.end(), 0);
            const auto res = pp->Sample(qubitsInt);
            Types::qubit_t result = 0;
            for (size_t i = 0; i < res.size(); ++i)
            {
                if (res[i])
                    result |= (1ULL << i);
            }
            return result;
        }
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        std::vector<int> qubitsInt(GetNumberOfQubits());
        std::iota(qubitsInt.begin(), qubitsInt.end(), 0);
        return pp->Sample(qubitsInt);
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        pp->ApplyP(static_cast<unsigned int>(qubit), lambda);
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        pp->ApplyX(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        pp->ApplyY(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        pp->ApplyZ(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        pp->ApplyH(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        pp->ApplyS(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        pp->ApplySDG(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        pp->ApplyT(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        pp->ApplyTDG(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        pp->ApplySX(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        pp->ApplySXDG(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        pp->ApplyK(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        pp->ApplyRX(static_cast<unsigned int>(qubit), theta);
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        pp->ApplyRY(static_cast<unsigned int>(qubit), theta);
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        pp->ApplyRZ(static_cast<unsigned int>(qubit), theta);
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        pp->ApplyU(static_cast<unsigned int>(qubit), theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        pp->ApplyCX(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        pp->ApplyCY(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        pp->ApplyCZ(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        pp->ApplyCP(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit), lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        pp->ApplyCRX(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit), theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        pp->ApplyCRY(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit), theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        pp->ApplyCRZ(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit), theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        pp->ApplyCH(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        pp->ApplyCSX(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        pp->ApplyCSXDAG(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        pp->ApplySWAP(static_cast<unsigned int>(qubit0), static_cast<unsigned int>(qubit1));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        pp->ApplyCCX(static_cast<unsigned int>(qubit0), static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit2));
        NotifyObservers({qubit2, qubit1, qubit0});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        pp->ApplyCSwap(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(qubit0), static_cast<unsigned int>(qubit1));
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        pp->ApplyCU(static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit), theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<QCSimPauliPropagatorSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->pauliWorkerCount = pauliWorkerCount;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        if (pp)
            cloned->pp = pp->Clone();
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::unique_ptr<QcsimPauliPropagator> pp;

    size_t pauliWorkerCount = 0;
};
} // namespace Simulators::Private
#endif
