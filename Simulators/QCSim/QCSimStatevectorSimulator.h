#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "QCSimGateSimulator.h"
#include "QubitRegister.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// statevector backend. Owns exactly one native QCSim implementation.
class QCSimStatevectorSimulator : public QCSimGateSimulator
{
  public:
    QCSimStatevectorSimulator()
    {
        configuration.SetConfiguration("method", "statevector");
    }

    const char *MethodName() const override
    {
        return "statevector";
    }
    friend class IndividualSimulator;

  public:
    void Initialize() override
    {
        if (nrQubits != 0)
        {
            state = std::make_unique<QC::QubitRegister<>>(nrQubits);
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        Eigen::VectorXcd amplitudesEigen(Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(), amplitudes.size()));
        state->setRegisterStorageFastNoNormalize(amplitudesEigen);
    }

#ifndef NO_QISKIT_AER
    void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        Eigen::VectorXcd amplitudesEigen(Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(), amplitudes.size()));
        state->setRegisterStorageFastNoNormalize(amplitudesEigen);
    }

#endif

    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        {
            state = std::make_unique<QC::QubitRegister<>>(nrQubits, amplitudes);
            state->SetMultithreading(enableMultithreading);
        }
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        state->setToBasisState(static_cast<size_t>(basisState));
    }

    void Reset() override
    {
        if (state)
            state->Reset();
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
            if (state)
                SeedBackend(state.get(), seed);
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
        if (state)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        state = nullptr;
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
                if (state->MeasureQubit(static_cast<unsigned int>(qubit)))
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
                if (state->MeasureQubit(static_cast<unsigned int>(qubits[q])))
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
                if (state->MeasureQubit(static_cast<unsigned int>(qubit)))
                    state->ApplyGate(xgate, static_cast<unsigned int>(qubit));
        }
        Notify();
        NotifyObservers(qubits);
    }

    std::vector<std::complex<double>> GetStateVector() override
    {
        if (!nrQubits)
            return {};
        const auto count = TensorQueries::DenseSize(nrQubits);
        const auto values = state->getRegisterStorage();
        if (static_cast<size_t>(values.size()) != count)
            throw std::runtime_error("QCSim statevector query returned an invalid size");
        return {values.data(), values.data() + values.size()};
    }

    double Probability(Types::qubit_t outcome) override
    {
        return state->getBasisStateProbability(static_cast<unsigned int>(outcome));
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        return state->getBasisStateAmplitude(static_cast<unsigned int>(outcome));
    }

    std::vector<double> AllProbabilities() override
    {
        const Eigen::VectorXcd probs = state->getRegisterStorage().cwiseAbs2();
        std::vector<double> result(probs.size());
        for (int i = 0; i < probs.size(); ++i)
            result[i] = probs[i].real();
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            const Eigen::VectorXcd &reg = state->getRegisterStorage();
            for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
                result[i] = std::norm(reg[qubits[i]]);
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
            if (shots > 1)
            {
                const auto &statev = state->getRegisterStorage();
                const Utils::Alias alias(statev);
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
                for (size_t shot = 0; shot < shots; ++shot)
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
            if (shots > 1)
            {
                const auto &statev = state->getRegisterStorage();
                const Utils::Alias alias(statev);
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
        std::vector<QC::Gates::AppliedGate<Eigen::MatrixXcd>> pauliStringVec;
        pauliStringVec.reserve(pauliString.size());
        for (size_t q = 0; q < pauliString.size(); ++q)
        {
            switch (toupper(pauliString[q]))
            {
            case 'X': {
                QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(xgate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
                pauliStringVec.emplace_back(std::move(ag));
            }
            break;
            case 'Y': {
                QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(ygate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
                pauliStringVec.emplace_back(std::move(ag));
            }
            break;
            case 'Z': {
                QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(zgate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
                pauliStringVec.emplace_back(std::move(ag));
            }
            break;
            case 'I':
                [[fallthrough]];
            default:
                break;
            }
        }
        if (pauliStringVec.empty())
            return 1.0;
        return state->ExpectationValue(pauliStringVec).real();
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kStatevector;
    }

    void SaveState() override
    {
        state->SaveState();
    }

    void RestoreState() override
    {
        state->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (state)
            state->SetMultithreading(multithreading);
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        return state->MeasureNoCollapse();
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        auto state = MeasureNoCollapse();
        std::vector<bool> res(nrQubits);
        for (size_t i = 0; i < nrQubits; ++i)
            res[i] = ((state >> i) & 1) == 1;
        return res;
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        const QC::Gates::AppliedGate<> agate(gate, qubit);
        ApplyNativeGate(agate);
        NotifyObservers({qubit});
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        const QC::Gates::AppliedGate<> agate(gate, qubit0, qubit1);
        ApplyNativeGate(agate);
        NotifyObservers({qubit0, qubit1});
    }

    void ApplyGenericThreeQubitGate(Types::qubit_t q0, Types::qubit_t q1, Types::qubit_t q2, const Matrix8cd &gate) override
    {
        const QC::Gates::AppliedGate<> applied(gate, q0, q1, q2);
        ApplyNativeGate(applied);
        NotifyObservers({q0, q1, q2});
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        pgate.SetPhaseShift(lambda);
        ApplyNativeGate(pgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        ApplyNativeGate(xgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        ApplyNativeGate(ygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        ApplyNativeGate(zgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        ApplyNativeGate(h, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        ApplyNativeGate(tgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(tdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sxdaggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        ApplyNativeGate(k, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        rxgate.SetTheta(theta);
        ApplyNativeGate(rxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        rygate.SetTheta(theta);
        ApplyNativeGate(rygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        rzgate.SetTheta(theta);
        ApplyNativeGate(rzgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        ugate.SetParams(theta, phi, lambda, gamma);
        ApplyNativeGate(ugate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(cxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(cygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(czgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        cpgate.SetPhaseShift(lambda);
        ApplyNativeGate(cpgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crxgate.SetTheta(theta);
        ApplyNativeGate(crxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crygate.SetTheta(theta);
        ApplyNativeGate(crygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crzgate.SetTheta(theta);
        ApplyNativeGate(crzgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(ch, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(csx, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(csxdag, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        ApplyNativeGate(swapgate, static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        ApplyNativeGate(ccxgate, static_cast<unsigned int>(qubit2), static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0));
        NotifyObservers({qubit2, qubit1, qubit0});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        ApplyNativeGate(cswapgate, static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        cugate.SetParams(theta, phi, lambda, gamma);
        ApplyNativeGate(cugate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<QCSimStatevectorSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        if (state)
            cloned->state = state->Clone();
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::unique_ptr<QC::QubitRegister<>> state;

  private:
    template <class Gate, class... Qubits> void ApplyNativeGate(const Gate &gate, Qubits... qubits)
    {
        state->ApplyGate(gate, qubits...);
    }
};
} // namespace Simulators::Private
#endif
