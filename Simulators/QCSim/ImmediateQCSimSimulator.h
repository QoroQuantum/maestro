#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../../Utils/Sampling/Alias.h"
#include "../../Utils/Sampling/Sampling.h"
#include "../Core/Configuration.h"
#include "../Interfaces/Simulator.h"
#include "../TensorNetworks/MPOValidation.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <utility>

namespace Simulators::Private
{
template <typename T, typename = void> struct HasSetSeed : std::false_type
{
};

template <typename T> struct HasSetSeed<T, std::void_t<decltype(std::declval<T &>().SetSeed(std::declval<uint64_t>()))>> : std::true_type
{
};

template <typename T> void SeedBackend(T *backend, uint64_t seed)
{
    if constexpr (HasSetSeed<T>::value)
        backend->SetSeed(seed);
}

} // namespace Simulators::Private

namespace Simulators::Private
{
// Common configuration, sampling and circuit bookkeeping. Native state belongs
// exclusively to the concrete simulator selected by the factory.
class ImmediateQCSimSimulator : public ISimulator
{
  public:
    virtual const char *MethodName() const = 0;

  protected:
    void ValidateMethod(const char *value) const
    {
        if (std::string(value) != MethodName())
            throw std::invalid_argument("A concrete simulator cannot change method; select the backend through SimulatorsFactory");
    }

    virtual void RefreshRoutingCallback()
    {
    }

    ImmediateQCSimSimulator() : samplingSeed(Utils::RandomStream::FreshSeed()), rng(samplingSeed), uniformZeroOne(0, 1)
    {
        SeedAuxiliaryRng(samplingSeed);
    }

  public:
    void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("QCSimState::InitializeState: Invalid "
                                 "simulation type for initializing the state.");
    }

#ifndef NO_QISKIT_AER
    void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("QCSimState::InitializeState: Invalid "
                                 "simulation type for initializing the state.");
    }

#endif

    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("QCSimState::InitializeState: Invalid "
                                 "simulation type for initializing the state.");
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        for (size_t q = 0; q < num_qubits; ++q)
            if ((basisState >> q) & 1ULL)
                ApplyX(static_cast<Types::qubit_t>(q));
    }

    void InitializeToBasisState(size_t num_qubits, const std::vector<bool> &basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        for (size_t q = 0; q < num_qubits && q < basisState.size(); ++q)
            if (basisState[q])
                ApplyX(static_cast<Types::qubit_t>(q));
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<Types::qubit_t, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("QCSimState::InitializeToMixtureOfBasisStates: Invalid simulation "
                                 "type for initializing to a mixture of basis states.");
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<std::vector<bool>, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("QCSimState::InitializeToMixtureOfBasisStates: Invalid simulation "
                                 "type for initializing to a mixture of basis states.");
    }

    bool SupportsMPSSwapOptimization() const override
    {
        return true;
    }

    void SetUpcomingGates(const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &gates) override
    {
        upcomingGates = gates;
        upcomingGateIndex = 0;
        if (!gateCounterObserver)
            gateCounterObserver = std::make_shared<GateCounterObserver>(upcomingGateIndex);
        RegisterObserver(gateCounterObserver);
        RefreshRoutingCallback();
    }

    long long int GetGatesCounter() const override
    {
        return upcomingGateIndex;
    }

    void SetGatesCounter(long long int counter) override
    {
        upcomingGateIndex = counter;
    }

    void IncrementGatesCounter() override
    {
        ++upcomingGateIndex;
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        const bool adaptiveSampling = std::string(MethodName()) == "statevector" || std::string(MethodName()) == "density_matrix" ||
                                      std::string(MethodName()) == "path_integral";
        if (adaptiveSampling && std::string(key) == "sampling_seed")
            return std::to_string(samplingSeed);
        if (adaptiveSampling && std::string(key) == "sampling_policy" && !configuration.IsSet(key))
            return "reproducible_v1";
        if (std::string(key) == "precision")
            return "double";
        if (std::string(key) == "use_double_precision")
            return "1";
        if (std::string("method") == key)
            return MethodName();
        return configuration.GetConfiguration(key);
    }

    size_t GetNumberOfQubits() const override
    {
        return nrQubits;
    }

    bool SupportsQuantumChannels() const override
    {
        return false;
    }

    void ApplyQuantumChannel(const Types::qubits_vector &targets, const QuantumChannel &channel) override
    {
        throw std::runtime_error("QCSim quantum channels require density_matrix or matrix_product_operator simulation");
    }

    double ProbabilityBits(const std::vector<bool> &bits) override
    {
        return IState::ProbabilityBits(bits);
    }

    std::complex<double> DensityMatrixElementBits(const std::vector<bool> &row, const std::vector<bool> &col) const override
    {
        return IState::DensityMatrixElementBits(row, col);
    }

    Eigen::MatrixXcd GetDensityMatrix(bool normalized = true) const override
    {
        return IState::GetDensityMatrix(normalized);
    }

    std::complex<double> ExpectationValueComplex(const std::string &pauli, bool normalized = true) const override
    {
        return IState::ExpectationValueComplex(pauli, normalized);
    }

    std::vector<std::complex<double>> ExpectationValuesComplex(const std::vector<std::string> &paulis, bool normalized = true) const override
    {
        return IState::ExpectationValuesComplex(paulis, normalized);
    }

    void ApplyOperator(const Types::qubits_vector &qubits, const Eigen::MatrixXcd &matrix, bool normalize = false) override
    {
        return IState::ApplyOperator(qubits, matrix, normalize);
    }

    void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override
    {
        return IState::MoveAtBeginningOfChain(qubits);
    }

    std::vector<std::complex<double>> GetStateVector() override
    {
        return IState::GetStateVector();
    }

    std::complex<double> ExpectationValueOperators(const Types::qubits_vector &qubits, const std::vector<Eigen::MatrixXcd> &matrices) override
    {
        return IState::ExpectationValueOperators(qubits, matrices);
    }

    std::complex<double> DensityMatrixTrace() const override
    {
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixOverlap(const IState &other) const override
    {
        throw std::invalid_argument("Density-matrix overlap requires two density matrices or two MPOs");
    }

    double DensityMatrixPurity() const override
    {
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixTraceOfSquare() const override
    {
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    double DensityMatrixHermiticityResidual() const override
    {
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    bool IsDensityMatrixHermitian(double eps = 1e-10) const override
    {
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &qubits) const override
    {
        throw std::runtime_error("Partial trace requires density_matrix or matrix_product_operator");
    }

    double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override
    {
        throw std::runtime_error("Mixed-state fidelity requires density_matrix or "
                                 "matrix_product_operator");
    }

    void RestoreDensityMatrixTrace() override
    {
        throw std::runtime_error("Trace restoration is only available for QCSim MPO");
    }

    void HermitizeDensityMatrix() override
    {
        throw std::runtime_error("Hermitization is only available for QCSim MPO");
    }

    void Trim() override
    {
        throw std::runtime_error("Trim is only available for QCSim MPS and MPO");
    }

    void ReCanonicalize() override
    {
        throw std::runtime_error("Canonicalization is only available for QCSim MPS and MPO");
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        throw std::runtime_error("QCSimState::Amplitude: Invalid simulation type for obtaining the "
                                 "amplitude of the specified outcome.");
    }

    std::complex<double> ProjectOnZero() override
    {
        return Amplitude(0);
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        return ISimulator::ExpectationValues(paulis);
    }

    SimulatorType GetType() const override
    {
        return SimulatorType::kQCSim;
    }

    void Flush() override
    {
    }

    void SaveStateToInternalDestructive() override
    {
    }

    void RestoreInternalDestructiveSavedState() override
    {
    }

    std::complex<double> AmplitudeRaw(Types::qubit_t outcome) override
    {
        return Amplitude(outcome);
    }

    bool GetMultithreading() const override
    {
        return enableMultithreading;
    }

    bool IsQcsim() const override
    {
        return true;
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        throw std::runtime_error("QCSimState::MeasureNoCollapse: Invalid simulation type for "
                                 "measuring "
                                 "all the qubits without collapsing the state.");
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        throw std::runtime_error("QCSimState::MeasureNoCollapseMany: Invalid simulation type for "
                                 "measuring all the qubits without collapsing the state.");
    }

    const Configuration &GetConfiguration() const
    {
        return configuration;
    }

    const std::unordered_map<std::string, std::string> &GetConfigMap() const override
    {
        return configuration.GetConfigMap();
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyGenericOneQubitGate: Unsupported simulation "
                                 "type.");
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        throw std::runtime_error("QCSimSimulator::ApplyGenericTwoQubitGate: Unsupported simulation "
                                 "type.");
    }

    void ApplyGenericThreeQubitGate(Types::qubit_t q0, Types::qubit_t q1, Types::qubit_t q2, const Matrix8cd &gate) override
    {
        throw std::runtime_error("Generic three-qubit gates require a statevector or density matrix");
    }

    void ApplyNop() override
    {
    }

  protected:
    bool ConfigureSampling(const char *key, const char *value)
    {
        if (!Utils::Sampling::ValidateSetting(key, value))
            return false;
        configuration.SetConfiguration(key, value);
        return true;
    }

    bool LegacySampling() const { return configuration.GetConfiguration("sampling_policy") == "legacy"; }

    Utils::Sampling::Options SamplingOptions() const
    {
        Utils::Sampling::Options options;
        options.multithreading = GetMultithreading();
        if (configuration.IsSet("sampling_max_memory_mb"))
            options.memoryBytes = static_cast<size_t>(std::stoull(configuration.GetConfiguration("sampling_max_memory_mb"))) << 20;
        return options;
    }

    void SeedSampling(uint64_t seed)
    {
        samplingSeed = seed;
        nextSeedStream = 0;
        rng.seed(seed);
        uniformZeroOne.reset();
    }

    void InitializeSamplingSeed()
    {
        // Recreating an unseeded native state must not restart an old stream.
        // Explicitly configured seeds deliberately replay on initialization.
        if (!configuration.IsSet("seed"))
        {
            SeedSampling(Utils::RandomStream::FreshSeed());
            SeedAuxiliaryRng(samplingSeed);
        }
    }

    uint64_t NextCloneSeed()
    {
        if (nextSeedStream == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("Simulator clone stream ids exhausted");
        return DeriveSeed(samplingSeed, nextSeedStream++);
    }

    size_t CheckedBasisStateCountForQueries() const
    {
        if (nrQubits >= std::numeric_limits<size_t>::digits)
            throw std::runtime_error("ImmediateQCSimState: Too many qubits for enumerating basis states.");
        return 1ULL << nrQubits;
    }

    size_t nrQubits = 0;

    bool enableMultithreading = true;

    std::vector<std::shared_ptr<Circuits::IOperation<>>> upcomingGates;

    long long int upcomingGateIndex = 0;

    class GateCounterObserver : public ISimulatorObserver
    {
      public:
        GateCounterObserver(long long int &indexRef) : index(indexRef)
        {
        }

        void Update(const Types::qubits_vector &) override
        {
            ++index;
        }

      private:
        long long int &index;
    };

    std::shared_ptr<GateCounterObserver> gateCounterObserver;

    uint64_t samplingSeed;
    std::mt19937_64 rng;

    uint64_t nextSeedStream = 0;

    std::uniform_real_distribution<double> uniformZeroOne;

    Configuration configuration;
};
} // namespace Simulators::Private
#endif
