/**
 * @file QiskitAerState.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The qiskit aer state class, derived from AER::AerState.
 *
 * Should not be used directly, this should allow exposing more from
 * AER::AerState than it's available through the public interface, if needed.
 */

#pragma once

#ifndef _QISKIT_AER_STATE_H_
#define _QISKIT_AER_STATE_H_

#ifndef NO_QISKIT_AER

#ifdef INCLUDED_BY_FACTORY

#include "controllers/state_controller.hpp"
#include <Eigen/Dense>
#include <numeric>

namespace AER
{

class AerStateFake
{
  public:
    virtual ~AerStateFake() = default;

    bool initialized_;
    uint_t num_of_qubits_;
    RngEngine rng_;
    int seed_ = std::random_device()();
    std::shared_ptr<QuantumState::Base> state_;
    json_t configs_;
    ExperimentResult last_result_;
};

} // namespace AER

namespace Simulators
{
// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private
{

/**
 * @class QiskitAerState
 * @brief Class for the qiskit aer simulator.
 *
 * Derived from AER::AerState, could add more functionality and expose more than
 * it's available through the base class public interface. Do not use this class
 * directly.
 */
class QiskitAerState : public AER::AerState
{
  public:
    const std::shared_ptr<AER::QuantumState::Base> &get_state() const
    {
        const AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;
        return fakeState->state_;
    }

    // AerState's buffer export extracts double-precision result storage even
    // when the actual state stores floats. Convert the concrete float state
    // before clearing it so saved states and clones retain their amplitudes.
    AER::Vector<complex_t> move_to_vector() override
    {
        flush_ops();
        const auto single = std::dynamic_pointer_cast<AER::Statevector::State<AER::QV::QubitVector<float>>>(get_state());
        if (!single)
            return AER::AerState::move_to_vector();
        auto values = single->move_to_vector();
        AER::Vector<complex_t> result(values.size(), false);
        for (size_t i = 0; i < values.size(); ++i)
            result[i] = values[i];
        clear();
        return result;
    }

    AER::cmatrix_t move_to_matrix() override
    {
        flush_ops();
        const auto single = std::dynamic_pointer_cast<AER::DensityMatrix::State<AER::QV::DensityMatrix<float>>>(get_state());
        if (!single)
            return AER::AerState::move_to_matrix();
        auto values = single->move_to_matrix();
        AER::cmatrix_t result(values.GetRows(), values.GetColumns());
        for (size_t i = 0; i < values.GetRows() * values.GetColumns(); ++i)
            result.data()[i] = values.data()[i];
        clear();
        return result;
    }

    double expval_pauli(const reg_t &qubits, const std::string &pauli)
    {
        if (qubits.empty() || pauli.empty())
            return 1.;

        const auto &state = get_state();
        if (!state)
            return 0.;

        flush_ops();

        return state->expval_pauli(qubits, pauli);
    }

    AER::Vector<complex_t> statevector()
    {
        const auto &state = get_state();
        if (!state)
            return {};

        flush_ops();

        AER::Operations::Op op;
        op.type = AER::Operations::OpType::save_statevec;
        op.name = "save_statevector";
        op.save_type = AER::Operations::DataSubType::single;
        op.string_params.push_back("statevector");
        for (uint_t qubit = 0; qubit < num_of_qubits(); ++qubit)
            op.qubits.push_back(qubit);

        AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;
        fakeState->last_result_ = AER::ExperimentResult();
        state->apply_op(op, fakeState->last_result_, fakeState->rng_);

        return std::move(
            static_cast<AER::DataMap<AER::SingleData, AER::Vector<complex_t>>>(std::move(fakeState->last_result_.data)).value()["statevector"].value());
    }

    // Aer's save_densmat preserves the register and packs the requested qubits
    // in the order supplied. An empty subset returns the complex trace.
    AER::cmatrix_t reduced_density_matrix(const reg_t &qubits)
    {
        if (!is_initialized() || !is_matrix())
            throw std::runtime_error("Density-matrix diagnostics require an initialized Aer density matrix");
        std::vector<bool> kept(num_of_qubits());
        for (auto qubit : qubits)
        {
            if (qubit >= kept.size() || kept[qubit])
                throw std::invalid_argument("Partial trace requires distinct qubits within the register");
            kept[qubit] = true;
        }
        flush_ops();
        AER::Operations::Op op;
        op.type = AER::Operations::OpType::save_densmat;
        op.name = "save_density_matrix";
        op.qubits = qubits;
        op.string_params = {"rho"};
        op.save_type = AER::Operations::DataSubType::average;
        AER::ExperimentResult result;
        AER::RngEngine rng(0);
        get_state()->apply_op(op, result, rng);
        auto &matrices = static_cast<AER::DataMap<AER::AverageData, AER::cmatrix_t, 1> &>(result.data);
        return std::move(matrices.value().at("rho").value());
    }

    template <class Read> auto read_density_matrix(Read read)
    {
        if (!is_initialized() || !is_matrix())
            throw std::runtime_error("Density-matrix diagnostics require an initialized Aer density matrix");
        flush_ops();
        // CPU registers expose their column-major storage. Reduce it directly,
        // accumulating in double precision even when the entries are floats.
        using DoubleMatrix = Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        using FloatMatrix = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        if (const auto *cpu = dynamic_cast<const AER::DensityMatrix::State<AER::QV::DensityMatrix<double>> *>(get_state().get()))
            return read(Eigen::Map<const DoubleMatrix>(cpu->qreg().data(), cpu->qreg().num_rows(), cpu->qreg().num_rows()));
        if (const auto *cpu = dynamic_cast<const AER::DensityMatrix::State<AER::QV::DensityMatrix<float>> *>(get_state().get()))
            return read(Eigen::Map<const FloatMatrix>(cpu->qreg().data(), cpu->qreg().num_rows(), cpu->qreg().num_rows()).cast<complex_t>());
        // Aer can also use on-device storage. Its native save operation
        // transfers the matrix to host memory for the scalar reduction.
        reg_t qubits(num_of_qubits());
        std::iota(qubits.begin(), qubits.end(), 0);
        const auto matrix = reduced_density_matrix(qubits);
        return read(Eigen::Map<const DoubleMatrix>(matrix.data(), matrix.GetRows(), matrix.GetColumns()));
    }

    // Hands the current settings to an existing extended stabilizer state,
    // which otherwise reads them only when it is created. Operations buffered
    // before the change run first, under the settings they were given with.
    void refresh_extended_stabilizer_config()
    {
        const auto extendedState = std::dynamic_pointer_cast<AER::ExtendedStabilizer::State>(get_state());
        if (!extendedState)
            return;
        flush_ops();
        const AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;
        extendedState->set_config(fakeState->configs_);
    }

    std::shared_ptr<AER::QuantumState::Base> clone_extended_stabilizer_state()
    {
        flush_ops();
        const auto extendedState = std::dynamic_pointer_cast<AER::ExtendedStabilizer::State>(get_state());
        if (!extendedState)
            throw std::runtime_error("QiskitAerState: current state is not an extended stabilizer state");
        return std::make_shared<AER::ExtendedStabilizer::State>(*extendedState);
    }

    AER::MatrixProductState::MPS &matrix_product_register()
    {
        flush_ops();
        return dynamic_cast<AER::MatrixProductState::State &>(*get_state()).qreg();
    }

    void restore_extended_stabilizer_state(const std::shared_ptr<AER::QuantumState::Base> &savedState)
    {
        const auto extendedState = std::dynamic_pointer_cast<AER::ExtendedStabilizer::State>(savedState);
        if (!extendedState)
            throw std::runtime_error("QiskitAerState: saved state is not an extended stabilizer state");

        AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;
        fakeState->state_ = std::make_shared<AER::ExtendedStabilizer::State>(*extendedState);
        fakeState->num_of_qubits_ = extendedState->qreg().get_n_qubits();
        fakeState->initialized_ = true;
        fakeState->last_result_ = AER::ExperimentResult();
        clear_ops();
    }

    std::vector<bool> apply_measure_many(const reg_t &qubits)
    {
        const auto &state = get_state();
        if (!state)
            return {};

        flush_ops();

        AER::Operations::Op op;
        op.type = AER::Operations::OpType::measure;
        op.name = "measure";
        op.qubits = qubits;
        op.memory = qubits;
        op.registers = qubits;

        AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;
        fakeState->last_result_ = AER::ExperimentResult();
        state->apply_op(op, fakeState->last_result_, fakeState->rng_);

        std::vector<bool> bitstring(qubits.size(), false);
        uint_t mem_size = state->creg().memory_size();
        for (size_t q = 0; q < qubits.size(); ++q)
        {
            const auto qubit = qubits[q];
            if (state->creg().creg_memory()[mem_size - qubit - 1] == '1')
                bitstring[q] = true;
        }
        return bitstring;
    }

    std::unordered_map<std::vector<bool>, uint_t> sample_counts_many(const reg_t &qubits, uint_t shots)
    {
        std::unordered_map<std::vector<bool>, uint_t> counts;
        for_each_sample(qubits, shots, [&](size_t, const std::vector<bool> &bits) { ++counts[bits]; });
        return counts;
    }

    // The individual rows let a composite pair independent component samples by shot index, without
    // copying their states or packing a global qubit number into a machine word.
    template <class Consumer> void for_each_sample(const reg_t &qubits, uint_t shots, Consumer &&consume)
    {
        const auto &state = get_state();
        if (!state)
            return;

        flush_ops();

        AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;

        // Aer's MPS backend returns samples sorted by ascending qubit index
        // regardless of the order of `qubits` (see sort_measured_values in
        // matrix_product_state_internal.cpp), while the statevector backend
        // returns them in the order of `qubits`. To get consistent behavior
        // across backends we sort the qubits ourselves and then remap the
        // bits back to the caller-requested order.
        reg_t sorted_qubits = qubits;
        std::sort(sorted_qubits.begin(), sorted_qubits.end());
        sorted_qubits.erase(std::unique(sorted_qubits.begin(), sorted_qubits.end()), sorted_qubits.end());
        reg_t positions;
        positions.reserve(qubits.size());
        for (const auto q : qubits)
            positions.push_back(std::lower_bound(sorted_qubits.begin(), sorted_qubits.end(), q) - sorted_qubits.begin());

        const auto samples = state->sample_measure(sorted_qubits, shots, fakeState->rng_);
        if (samples.size() != shots)
            throw std::runtime_error("Aer returned an incomplete sample");
        std::vector<bool> bitstring(qubits.size());
        for (size_t shot = 0; shot < samples.size(); ++shot)
        {
            for (size_t i = 0; i < qubits.size(); ++i)
                bitstring[i] = samples[shot][positions[i]] == 1;
            consume(shot, bitstring);
        }
    }
};

} // namespace Private
} // namespace Simulators

#endif
#endif
#endif
