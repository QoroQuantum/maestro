#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../../Utils/Alias.h"
#include "../Core/Configuration.h"
#include "../Interfaces/Simulator.h"
#include "../TensorNetworks/MPOValidation.h"
#include <algorithm>
#include <cctype>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <utility>

namespace Simulators::Private
{
// Common configuration, sampling and circuit bookkeeping. Native state belongs
// exclusively to the concrete simulator selected by the factory.
class ImmediateGpuSimulator : public ISimulator
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

  public:
    void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("GpuState::InitializeState: Invalid simulation "
                                 "type for initializing the state.");
    }

#ifndef NO_QISKIT_AER
    void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("GpuState::InitializeState: Invalid simulation "
                                 "type for initializing the state.");
    }

#endif

    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("GpuState::InitializeState: Invalid simulation "
                                 "type for initializing the state.");
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
        throw std::runtime_error("GpuState::InitializeToMixtureOfBasisStates: Invalid simulation "
                                 "type for initializing to a mixture of basis states.");
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<std::vector<bool>, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        throw std::runtime_error("GpuState::InitializeToMixtureOfBasisStates: Invalid simulation "
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
        throw std::runtime_error("GPU quantum channels require an initialized density matrix or "
                                 "matrix product operator");
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

    std::complex<double> ExpectationValueOperators(const Types::qubits_vector &qubits, const std::vector<Eigen::MatrixXcd> &matrices) override
    {
        return IState::ExpectationValueOperators(qubits, matrices);
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

    std::complex<double> DensityMatrixTrace() const override
    {
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixOverlap(const IState &other) const override
    {
        throw std::invalid_argument("Density-matrix overlap requires two density matrices or two MPOs");
    }

    double DensityMatrixPurity() const override
    {
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixTraceOfSquare() const override
    {
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    double DensityMatrixHermiticityResidual() const override
    {
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    bool IsDensityMatrixHermitian(double eps = 1e-10) const override
    {
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &qubits) const override
    {
        std::vector<int> keep(qubits.begin(), qubits.end());
        throw std::runtime_error("GPU partial trace requires a mixed-state backend");
    }

    double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override
    {
        std::vector<double> raw(2 * static_cast<size_t>(psi.size()));
        for (Eigen::Index i = 0; i < psi.size(); ++i)
        {
            raw[2 * i] = psi[i].real();
            raw[2 * i + 1] = psi[i].imag();
        }
        throw std::runtime_error("GPU mixed-state fidelity requires density_matrix or "
                                 "matrix_product_operator");
    }

    void RestoreDensityMatrixTrace() override
    {
        throw std::runtime_error("Trace restoration is only available for GPU MPO");
    }

    void HermitizeDensityMatrix() override
    {
        throw std::runtime_error("Hermitization is only available for GPU MPO");
    }

    void Trim() override
    {
        throw std::runtime_error("Trim is only available for GPU MPS and MPO");
    }

    void ReCanonicalize() override
    {
        throw std::runtime_error("Canonicalization is only available for GPU MPS and MPO");
    }

    double Probability(Types::qubit_t outcome) override
    {
        const auto ampl = Amplitude(outcome);
        return std::norm(ampl);
    }

    std::complex<double> ProjectOnZero() override
    {
        return Amplitude(0);
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (size_t i = 0; i < qubits.size(); ++i)
            {
                const auto ampl = Amplitude(qubits[i]);
                result[i] = std::norm(ampl);
            }
        }
        return result;
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        return IState::ExpectationValues(paulis);
    }

    SimulatorType GetType() const override
    {
        return SimulatorType::kGpuSim;
    }

    void Flush() override
    {
        const bool done = true;
        if (!done)
            throw std::runtime_error("GpuState::Flush: device synchronization failed");
    }

    void SaveStateToInternalDestructive() override
    {
        throw std::runtime_error("GpuState::SaveStateToInternalDestructive: Invalid simulation type "
                                 "for saving the state destructively.");
    }

    void RestoreInternalDestructiveSavedState() override
    {
        throw std::runtime_error("GpuState::RestoreInternalDestructiveSavedState: Invalid simulation "
                                 "type for restoring the state destructively.");
    }

    std::complex<double> AmplitudeRaw(Types::qubit_t outcome) override
    {
        return Amplitude(outcome);
    }

    void SetMultithreading(bool multithreading = true) override
    {
    }

    bool GetMultithreading() const override
    {
        return true;
    }

    bool IsQcsim() const override
    {
        return false;
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (nrQubits > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        Types::qubits_vector fixedValues(nrQubits);
        std::iota(fixedValues.begin(), fixedValues.end(), 0);
        const auto res = SampleCounts(fixedValues, 1);
        if (res.empty())
            return 0;
        return res.begin()->first;
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        Types::qubits_vector fixedValues(nrQubits);
        std::iota(fixedValues.begin(), fixedValues.end(), 0);
        const auto res = SampleCountsMany(fixedValues, 1);
        if (res.empty())
            return std::vector<bool>(nrQubits, false);
        return res.begin()->first;
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
        throw std::runtime_error("GpuSimulator::ApplyGenericOneQubitGate: Not supported for GPU "
                                 "simulator yet.");
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        throw std::runtime_error("GpuSimulator::ApplyGenericTwoQubitGate: Not supported for GPU "
                                 "simulator yet.");
    }

    void ApplyGenericThreeQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2,
                                    const Eigen::Matrix<std::complex<double>, 8, 8> &gate) override
    {
        throw std::runtime_error("Generic three-qubit gates require GPU statevector");
    }

    void ApplyNop() override
    {
    }

  protected:
    static std::vector<size_t> SampleBitPositions(const Types::qubits_vector &qubits)
    {
        auto sorted = qubits;
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        std::vector<size_t> positions;
        positions.reserve(qubits.size());
        for (const auto q : qubits)
            positions.push_back(std::lower_bound(sorted.begin(), sorted.end(), q) - sorted.begin());
        return positions;
    }

    uint64_t nextSeedStream = 0;

    size_t nrQubits = 0;

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

    Configuration configuration;
};
} // namespace Simulators::Private
#endif
