#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "DensityMatrix.h"
#include "QCSimGateSimulator.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// Keep density sampling on its existing native engine. The adapter exposes an
// immutable-table draw input without changing the external QCSim dependency.
class SamplingDensityMatrix : public QC::DensityMatrix<>
{
  public:
    using QC::DensityMatrix<>::DensityMatrix;
    double SamplingUniform() { return Utils::RandomStream::Uniform(rng); }
};

// density_matrix backend. Owns exactly one native QCSim implementation.
class QCSimDensityMatrixSimulator : public QCSimGateSimulator
{
  public:
    QCSimDensityMatrixSimulator()
    {
        configuration.SetConfiguration("method", "density_matrix");
    }

    const char *MethodName() const override
    {
        return "density_matrix";
    }

    void Initialize() override
    {
        if (nrQubits != 0)
        {
            InitializeSamplingSeed();
            {
                densityMatrix = std::make_unique<SamplingDensityMatrix>(nrQubits);
                SeedBackend(densityMatrix.get(), samplingSeed);
            }
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
        densityMatrix->setFromStatevector(amplitudesEigen);
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
        densityMatrix->setFromStatevector(amplitudesEigen);
    }

#endif

    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        densityMatrix->setFromStatevector(amplitudes);
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        densityMatrix->setToBasisState(static_cast<size_t>(basisState));
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<Types::qubit_t, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        std::vector<std::pair<size_t, double>> converted;
        converted.reserve(mixture.size());
        for (const auto &[basisState, weight] : mixture)
            converted.emplace_back(static_cast<size_t>(basisState), weight);
        densityMatrix->setToMixtureOfBasisStates(converted);
    }

    void Reset() override
    {
        if (densityMatrix)
            densityMatrix->Reset();
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
            if (densityMatrix)
                SeedBackend(densityMatrix.get(), seed);
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
        if (densityMatrix)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        densityMatrix = nullptr;
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
                if (densityMatrix->MeasureQubit(qubit))
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
                if (densityMatrix->MeasureQubit(qubits[q]))
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
                densityMatrix->ApplyReset(qubit);
        }
        Notify();
        NotifyObservers(qubits);
    }

    bool SupportsQuantumChannels() const override
    {
        return true;
    }

    void ApplyQuantumChannel(const Types::qubits_vector &targets, const QuantumChannel &channel) override
    {
        if (!SupportsQuantumChannels())
            throw std::runtime_error("QCSim quantum channels require density_matrix or "
                                     "matrix_product_operator simulation");
        if (targets.size() != channel.GetNumberOfQubits())
            throw std::invalid_argument("The number of channel targets does not match its Kraus operators");
        if (targets.empty() || targets.size() > 2)
            throw std::invalid_argument("QCSim supports only one- and two-qubit local channels");
        std::unordered_set<Types::qubit_t> uniqueTargets;
        for (const Types::qubit_t target : targets)
        {
            if (target >= nrQubits)
                throw std::invalid_argument("Quantum-channel qubit is out of range");
            if (!uniqueTargets.insert(target).second)
                throw std::invalid_argument("Quantum-channel target qubits must be distinct");
        }
        const auto &krausOperators = channel.GetKrausOperators();
        {
            if (!densityMatrix)
                throw std::runtime_error("QCSim density-matrix state is not initialized");
            if (targets.size() == 1)
                densityMatrix->ApplyChannel(krausOperators, targets[0]);
            else
                densityMatrix->ApplyChannel(krausOperators, targets[0], targets[1]);
        }
        NotifyObservers(targets);
    }

    std::complex<double> DensityMatrixTrace() const override
    {
        if (densityMatrix)
            return densityMatrix->Trace();
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    double DensityMatrixPurity() const override
    {
        if (densityMatrix)
            return densityMatrix->Purity();
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixTraceOfSquare() const override
    {
        if (densityMatrix)
        {
            const auto &rho = densityMatrix->getDensityMatrix();
            return (rho * rho).trace();
        }
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixOverlap(const IState &other) const override
    {
        const auto *rhs = dynamic_cast<const QCSimDensityMatrixSimulator *>(&other);
        if (!rhs)
            throw std::invalid_argument("Density-matrix overlap requires matching QCSim backends");
        if (densityMatrix && rhs->densityMatrix)
            return densityMatrix->HilbertSchmidtOverlap(*rhs->densityMatrix);
        throw std::invalid_argument("Density-matrix overlap requires two density matrices or two MPOs");
    }

    double DensityMatrixHermiticityResidual() const override
    {
        if (densityMatrix)
            return (densityMatrix->getDensityMatrix() - densityMatrix->getDensityMatrix().adjoint()).norm();
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    bool IsDensityMatrixHermitian(double eps = 1e-10) const override
    {
        if (densityMatrix)
            return densityMatrix->IsHermitian(eps);
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &qubits) const override
    {
        if (densityMatrix)
            return densityMatrix->PartialTrace(std::vector<size_t>(qubits.begin(), qubits.end()));
        throw std::runtime_error("Partial trace requires density_matrix or matrix_product_operator");
    }

    double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override
    {
        if (densityMatrix)
            return densityMatrix->FidelityWithStatevector(psi);
        throw std::runtime_error("Mixed-state fidelity requires density_matrix or "
                                 "matrix_product_operator");
    }

    double Probability(Types::qubit_t outcome) override
    {
        return densityMatrix->getBasisStateProbability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        throw std::runtime_error("QCSimState::Amplitude: Amplitudes are not defined for the density "
                                 "matrix simulator.");
    }

    std::vector<double> AllProbabilities() override
    {
        const size_t nrBasisStates = densityMatrix->getNrBasisStates();
        std::vector<double> result(nrBasisStates);
        for (size_t i = 0; i < nrBasisStates; ++i)
            result[i] = densityMatrix->getBasisStateProbability(i);
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
                result[i] = densityMatrix->getBasisStateProbability(qubits[i]);
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
            const auto sampler = densityMatrix->PrepareSampler();
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const size_t measured = densityMatrix->MeasureNoCollapse(sampler);
                Types::qubit_t packed = 0;
                for (size_t i = 0; i < qubits.size(); ++i)
                    if ((measured & (1ULL << qubits[i])) != 0)
                        packed |= 1ULL << i;
                ++result[packed];
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
            const auto sampler = densityMatrix->PrepareSampler();
            std::vector<bool> packed(qubits.size());
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const size_t measured = densityMatrix->MeasureNoCollapse(sampler);
                for (size_t i = 0; i < qubits.size(); ++i)
                    packed[i] = (measured & (1ULL << qubits[i])) != 0;
                ++result[packed];
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
        {
            pauliString.resize(GetNumberOfQubits(), 'I');
            return densityMatrix->ExpectationValue(pauliString).real();
        }
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kDensityMatrix;
    }

    void SaveState() override
    {
        densityMatrix->SaveState();
    }

    void RestoreState() override
    {
        densityMatrix->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (densityMatrix)
            densityMatrix->SetMultithreading(multithreading);
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        return densityMatrix->MeasureNoCollapse();
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        const auto measured = densityMatrix->MeasureNoCollapse();
        std::vector<bool> res(nrQubits);
        for (size_t i = 0; i < nrQubits; ++i)
            res[i] = ((measured >> i) & 1) == 1;
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
        return CloneForExecution(NextCloneSeed());
    }

    std::unique_ptr<ISimulator> CloneForExecution(uint64_t seed) override
    {
        auto cloned = std::make_unique<QCSimDensityMatrixSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        if (densityMatrix)
            cloned->densityMatrix = std::make_unique<SamplingDensityMatrix>(*densityMatrix);
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        // Native DensityMatrix copying includes its RNG. Replace every stream
        // before publishing the clone, even when the parent was randomly seeded.
        cloned->SetSeed(seed);
        return cloned;
    }

  protected:
    std::unique_ptr<SamplingDensityMatrix> densityMatrix;

  private:
    template <bool Many> Utils::Sampling::Counts<Many> SamplePrepared(const Types::qubits_vector &qubits, size_t shots)
    {
        if (qubits.empty() || !shots)
            return {};
        Utils::Sampling::ValidateQubits(qubits, nrQubits, Many);
        const size_t states = densityMatrix->getNrBasisStates();
        std::vector<size_t> selected, positions;
        for (const auto q : qubits)
        {
            const auto found = std::find(selected.begin(), selected.end(), q);
            positions.push_back(static_cast<size_t>(found - selected.begin()));
            if (found == selected.end())
                selected.push_back(q);
        }
        const bool marginal = selected.size() < nrQubits && selected.size() <= 12 && shots >= std::max<size_t>(256, states / 16);
        const size_t outcomes = marginal ? size_t{1} << selected.size() : states;
        auto options = SamplingOptions();
        if (outcomes > (options.memoryBytes - 4096) / sizeof(double))
            throw std::length_error("Density sampling snapshot exceeds the scratch budget");
        options.memoryBytes -= outcomes * sizeof(double);
        std::vector<double> probabilities(outcomes, 0.);
        const auto &matrix = densityMatrix->getDensityMatrix();
        double mass = 0.;
        for (size_t state = 0; state < states; ++state)
        {
            const auto diagonal = matrix(state, state);
            if (!std::isfinite(diagonal.real()) || !std::isfinite(diagonal.imag()) || std::abs(diagonal.imag()) > 1E-10 || diagonal.real() < -1E-12)
                throw std::domain_error("Density-matrix populations must be finite, real and nonnegative");
            const double probability = std::max(0., diagonal.real());
            mass += probability;
            size_t row = state;
            if (marginal)
            {
                row = 0;
                for (size_t bit = 0; bit < selected.size(); ++bit)
                    if ((state >> selected[bit]) & 1)
                        row |= size_t{1} << bit;
            }
            probabilities[row] += probability;
        }
        if (!std::isfinite(mass) || mass <= 1E-20)
            throw std::domain_error("Cannot sample a density matrix with no probability mass");
        const Utils::Sampling::Prepared plan(outcomes, shots, [&](size_t row) { return probabilities[row]; }, options);
        auto result = Utils::Sampling::Count<Many>(
            plan, shots, qubits.size(), [&](size_t row, size_t bit) { return ((row >> (marginal ? positions[bit] : qubits[bit])) & 1) != 0; },
            [&] { return densityMatrix->SamplingUniform(); });
        NotifyObservers(qubits);
        return result;
    }

    template <class Gate, class... Qubits> void ApplyNativeGate(const Gate &gate, Qubits... qubits)
    {
        // Native density diagonal loops can contract complex products
        // differently when outlined by OpenMP. Preserve their serial rounding
        // in reproducible trajectories; terminal preparation keeps its existing
        // threading policy, and the other gate kernels remain parallel.
        struct RestoreThreading
        {
            SamplingDensityMatrix &state;
            bool enabled;
            ~RestoreThreading() { state.SetMultithreading(enabled); }
        } restore{*densityMatrix, densityMatrix->GetMultithreading()};
        bool hasTeam = false;
#ifdef _OPENMP
        hasTeam = !omp_in_parallel() && omp_get_max_threads() > 1;
#endif
        const bool trajectory = configuration.GetConfiguration("reproducible_trajectory") == "true";
        if (!LegacySampling() && (!hasTeam || (trajectory && gate.getStructure().kind == QC::Gates::GateStructure::Kind::Diagonal)))
            densityMatrix->SetMultithreading(false);
        densityMatrix->ApplyGate(gate, static_cast<size_t>(qubits)...);
    }
};
} // namespace Simulators::Private
#endif
