#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../../TensorNetworks/Contractors/ForestContractor.h"
#include "../../TensorNetworks/TensorNetwork.h"
#include "QCSimGateSimulator.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// tensor_network backend. Owns exactly one native QCSim implementation.
class QCSimTensorNetworkSimulator : public QCSimGateSimulator
{
  public:
    QCSimTensorNetworkSimulator()
    {
        configuration.SetConfiguration("method", "tensor_network");
    }

    const char *MethodName() const override
    {
        return "tensor_network";
    }

    void Initialize() override
    {
        if (nrQubits != 0)
        {
            {
                tensorNetwork = std::make_unique<TensorNetworks::TensorNetwork>(nrQubits);
                const auto tensorContractor = std::make_shared<TensorNetworks::ForestContractor>();
                tensorNetwork->SetContractor(tensorContractor);
            }
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void Reset() override
    {
        if (tensorNetwork)
            tensorNetwork->Clear();
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
            if (tensorNetwork)
                tensorNetwork->SetSeed(seed);
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
        if (tensorNetwork)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        tensorNetwork = nullptr;
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
                if (tensorNetwork->Measure(static_cast<unsigned int>(qubit)))
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
                if (tensorNetwork->Measure(static_cast<unsigned int>(qubits[q])))
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
                if (tensorNetwork->Measure(static_cast<unsigned int>(qubit)))
                    tensorNetwork->AddGate(xgate, static_cast<unsigned int>(qubit));
        }
        Notify();
        NotifyObservers(qubits);
    }

    double Probability(Types::qubit_t outcome) override
    {
        return tensorNetwork->getBasisStateProbability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        throw std::runtime_error("QCSimState::Amplitude: Not supported for the "
                                 "tensor network simulator.");
    }

    std::vector<double> AllProbabilities() override
    {
        throw std::runtime_error("QCSimState::AllProbabilities: Invalid "
                                 "simulation type for obtaining probabilities.");
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        throw std::runtime_error("QCSimState::Probabilities: Not implemented yet "
                                 "for the tensor network simulator.");
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
            tensorNetwork->SaveState();
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const size_t meas = Measure(qubits);
                ++result[meas];
                tensorNetwork->RestoreState();
            }
            tensorNetwork->ClearSavedState();
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
            tensorNetwork->SaveState();
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const auto meas = MeasureMany(qubits);
                ++result[meas];
                tensorNetwork->RestoreState();
            }
            tensorNetwork->ClearSavedState();
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
        return tensorNetwork->ExpectationValue(pauliString);
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kTensorNetwork;
    }

    void SaveState() override
    {
        tensorNetwork->SaveState();
    }

    void RestoreState() override
    {
        tensorNetwork->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (tensorNetwork)
            tensorNetwork->SetMultithreading(multithreading);
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        const QC::Gates::AppliedGate<> agate(gate, qubit);
        tensorNetwork->AddGate(agate, qubit);
        NotifyObservers({qubit});
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        const QC::Gates::AppliedGate<> agate(gate, qubit0, qubit1);
        tensorNetwork->AddGate(agate, qubit1, qubit0);
        NotifyObservers({qubit0, qubit1});
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        pgate.SetPhaseShift(lambda);
        tensorNetwork->AddGate(pgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(xgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(ygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(zgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(h, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(sgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(sdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(tgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(tdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(sxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(sxdaggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        tensorNetwork->AddGate(k, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        rxgate.SetTheta(theta);
        tensorNetwork->AddGate(rxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        rygate.SetTheta(theta);
        tensorNetwork->AddGate(rygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        rzgate.SetTheta(theta);
        tensorNetwork->AddGate(rzgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        ugate.SetParams(theta, phi, lambda, gamma);
        tensorNetwork->AddGate(ugate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tensorNetwork->AddGate(cxgate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tensorNetwork->AddGate(cygate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tensorNetwork->AddGate(czgate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        cpgate.SetPhaseShift(lambda);
        tensorNetwork->AddGate(cpgate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crxgate.SetTheta(theta);
        tensorNetwork->AddGate(crxgate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crygate.SetTheta(theta);
        tensorNetwork->AddGate(crygate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crzgate.SetTheta(theta);
        tensorNetwork->AddGate(crzgate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tensorNetwork->AddGate(ch, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tensorNetwork->AddGate(csx, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tensorNetwork->AddGate(csxdag, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        tensorNetwork->AddGate(swapgate, static_cast<unsigned int>(qubit0), static_cast<unsigned int>(qubit1));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        const size_t q1 = qubit0;
        const size_t q2 = qubit1;
        const size_t q3 = qubit2;
        tensorNetwork->AddGate(csx, static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit1, qubit2});
        tensorNetwork->AddGate(cxgate, static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({qubit0, qubit1});
        tensorNetwork->AddGate(csxdag, static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit1, qubit2});
        tensorNetwork->AddGate(cxgate, static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({qubit0, qubit1});
        tensorNetwork->AddGate(csx, static_cast<unsigned int>(q1), static_cast<unsigned int>(q3));
        NotifyObservers({qubit0, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        const size_t q1 = ctrl_qubit;
        const size_t q2 = qubit0;
        const size_t q3 = qubit1;
        tensorNetwork->AddGate(cxgate, static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit1, qubit0});
        tensorNetwork->AddGate(csx, static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit0, qubit1});
        tensorNetwork->AddGate(cxgate, static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({ctrl_qubit, qubit0});
        pgate.SetPhaseShift(M_PI);
        tensorNetwork->AddGate(pgate, static_cast<unsigned int>(q3));
        NotifyObservers({qubit1});
        pgate.SetPhaseShift(-M_PI_2);
        tensorNetwork->AddGate(pgate, static_cast<unsigned int>(q2));
        NotifyObservers({qubit0});
        tensorNetwork->AddGate(csx, static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit0, qubit1});
        tensorNetwork->AddGate(cxgate, static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({ctrl_qubit, qubit0});
        pgate.SetPhaseShift(M_PI);
        tensorNetwork->AddGate(pgate, static_cast<unsigned int>(q3));
        NotifyObservers({qubit1});
        tensorNetwork->AddGate(csx, static_cast<unsigned int>(q1), static_cast<unsigned int>(q3));
        NotifyObservers({ctrl_qubit, qubit1});
        tensorNetwork->AddGate(cxgate, static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        cugate.SetParams(theta, phi, lambda, gamma);
        tensorNetwork->AddGate(cugate, static_cast<unsigned int>(ctrl_qubit), static_cast<unsigned int>(tgt_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<QCSimTensorNetworkSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        if (tensorNetwork)
            cloned->tensorNetwork = tensorNetwork->Clone();
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::unique_ptr<TensorNetworks::TensorNetwork> tensorNetwork;
};
} // namespace Simulators::Private
#endif
