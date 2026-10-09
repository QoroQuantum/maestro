#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "Clifford.h"
#include "ImmediateQCSimSimulator.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// stabilizer backend. Owns exactly one native QCSim implementation.
class QCSimStabilizerSimulator : public ImmediateQCSimSimulator
{
  public:
    QCSimStabilizerSimulator()
    {
        configuration.SetConfiguration("method", "stabilizer");
    }

    const char *MethodName() const override
    {
        return "stabilizer";
    }

    void Initialize() override
    {
        if (nrQubits != 0)
        {
            cliffordSimulator = std::make_unique<QC::Clifford::StabilizerSimulator>(nrQubits);
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void Reset() override
    {
        if (cliffordSimulator)
            cliffordSimulator->Reset();
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
            if (cliffordSimulator)
                SeedBackend(cliffordSimulator.get(), seed);
            return;
        }
        if (std::string(key) == "pauli_propagator_workers")
        {
            const auto workers = configuration.GetConfigurationAsUnsigned(key);
            if (workers > 1024)
                throw std::invalid_argument("pauli_propagator_workers exceeds 1024");
        }
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        if (cliffordSimulator)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        cliffordSimulator = nullptr;
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
                if (cliffordSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
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
                if (cliffordSimulator->MeasureQubit(static_cast<unsigned int>(qubits[q])))
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
                if (cliffordSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
                    cliffordSimulator->ApplyX(static_cast<unsigned int>(qubit));
        }
        Notify();
        NotifyObservers(qubits);
    }

    double Probability(Types::qubit_t outcome) override
    {
        if constexpr (std::numeric_limits<size_t>::digits >= std::numeric_limits<Types::qubit_t>::digits)
        {
            return cliffordSimulator->getBasisStateProbability(static_cast<size_t>(outcome));
        }
        else
        {
            const size_t n = cliffordSimulator->getNrQubits();
            if (n < std::numeric_limits<Types::qubit_t>::digits && (outcome >> n) != 0)
                return 0.0;
            std::vector<bool> bits(n);
            for (size_t q = 0; q < n && q < std::numeric_limits<Types::qubit_t>::digits; ++q)
                bits[q] = ((outcome >> q) & 1) != 0;
            return cliffordSimulator->getBasisStateProbability(bits);
        }
    }

    std::vector<double> AllProbabilities() override
    {
        return cliffordSimulator->AllProbabilities();
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        throw std::runtime_error("QCSimState::Probabilities: Invalid simulation "
                                 "type for obtaining probabilities.");
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        {
            const auto selected = CliffordSamplingQubits(qubits);
            const auto counts = cliffordSimulator->SampleCounts(selected, shots);
            std::unordered_map<Types::qubit_t, Types::qubit_t> result;
            for (const auto &item : counts)
                result.emplace(item.first, item.second);
            NotifyObservers(qubits);
            return result;
        }
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        {
            const auto selected = CliffordSamplingQubits(qubits);
            const auto counts = cliffordSimulator->SampleCountsMany(selected, shots);
            std::unordered_map<std::vector<bool>, Types::qubit_t> result;
            for (const auto &item : counts)
                result.emplace(item.first, item.second);
            NotifyObservers(qubits);
            return result;
        }
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
        return cliffordSimulator->ExpectationValue(pauliString);
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kStabilizer;
    }

    void SaveState() override
    {
        cliffordSimulator->SaveState();
    }

    void RestoreState() override
    {
        cliffordSimulator->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (cliffordSimulator)
            cliffordSimulator->SetMultithreading(multithreading);
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        {
            if (std::abs(lambda - M_PI_2) > 1e-10)
                throw std::runtime_error("QCSimSimulator::ApplyP: Invalid phase shift "
                                         "angle for a Clifford gate.");
            cliffordSimulator->ApplyS(static_cast<unsigned int>(qubit));
        }
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplyX(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplyY(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplyZ(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplyH(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplyS(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplySdg(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyT: The stabilizer simulator does not support "
                                 "non-clifford gates.");
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyTDG: The stabilizer simulator does not support "
                                 "non-clifford gates.");
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplySx(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplySxDag(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        cliffordSimulator->ApplyK(static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyRx: The stabilizer "
                                 "simulator does not support the Rx gate.");
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyRy: The stabilizer "
                                 "simulator does not support the Ry gate.");
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyRz: The stabilizer "
                                 "simulator does not support the Rz gate.");
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyU: The stabilizer "
                                 "simulator does not support the U gate.");
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        cliffordSimulator->ApplyCX(static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        cliffordSimulator->ApplyCY(static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        cliffordSimulator->ApplyCZ(static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCP: The stabilizer "
                                 "simulator does not support the CP gate.");
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCRx: The stabilizer "
                                 "simulator does not support the CRx gate.");
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCRy: The stabilizer "
                                 "simulator does not support the CRy gate.");
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCRz: The stabilizer "
                                 "simulator does not support the CRz gate.");
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCH: The stabilizer "
                                 "simulator does not support the CH gate.");
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCSx: The stabilizer "
                                 "simulator does not support the CSx gate.");
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCSxDAG: The stabilizer "
                                 "simulator does not support the CSxDag gate.");
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        cliffordSimulator->ApplySwap(static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCCX: The stabilizer "
                                 "simulator does not support the CCX gate.");
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCSwap: The stabilizer "
                                 "simulator does not support the CSwap gate.");
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyCU: The stabilizer "
                                 "simulator does not support the CU gate.");
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<QCSimStabilizerSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        if (cliffordSimulator)
            cloned->cliffordSimulator = cliffordSimulator->Clone();
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::vector<size_t> CliffordSamplingQubits(const Types::qubits_vector &qubits) const
    {
        std::vector<size_t> selected;
        selected.reserve(qubits.size());
        for (const auto q : qubits)
        {
            if (q >= cliffordSimulator->getNrQubits())
                throw std::out_of_range("Qubit index out of range");
            selected.push_back(static_cast<size_t>(q));
        }
        return selected;
    }

    std::unique_ptr<QC::Clifford::StabilizerSimulator> cliffordSimulator;
};
} // namespace Simulators::Private
#endif
