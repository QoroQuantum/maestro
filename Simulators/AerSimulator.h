/**
 * @file AerSimulator.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The aer simulator class.
 *
 * Should not be used directly, create an instance with the factory and use the
 * generic interface instead.
 */

#pragma once

#ifndef _AER_SIMULATOR_H_
#define _AER_SIMULATOR_H_

#ifndef NO_QISKIT_AER

#ifdef INCLUDED_BY_FACTORY

#include "AerState.h"

namespace Simulators {
// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private {

/**
 * @class AerSimulator
 * @brief aer simulator clas.
 *
 * This is the implementation for the qiskit aer simulator.
 * Do not use this class directly, use the factory to create an instance.
 * Only the interface should be exposed.
 * @sa AerState
 * @sa ISimulator
 * @sa IState
 */
class AerSimulator : public AerState {
 public:
  AerSimulator() = default;
  // allow no copy or assignment
  AerSimulator(const AerSimulator &) = delete;
  AerSimulator &operator=(const AerSimulator &) = delete;

  // but allow moving
  AerSimulator(AerSimulator &&other) = default;
  AerSimulator &operator=(AerSimulator &&other) = default;

  /**
   * @brief Applies a phase shift gate to the qubit
   *
   * Applies a specified phase shift gate to the qubit
   * @param qubit The qubit to apply the gate to.
   * @param lambda The phase shift angle.
   */
  void ApplyP(Types::qubit_t qubit, double lambda) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error("P gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "p";
      op.qubits = AER::reg_t{qubit};
      op.params = {lambda};

      state->buffer_op(std::move(op));
    } else {
      const AER::cvector_t P = {{1, 0}, std::polar(1., lambda)};
      state->apply_diagonal_matrix(qubits, P);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a not gate to the qubit
   *
   * Applies a not (X) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyX(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};
    state->apply_x(qubit);
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Y gate to the qubit
   *
   * Applies a not (Y) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyY(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};
    state->apply_y(qubit);
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Z gate to the qubit
   *
   * Applies a not (Z) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyZ(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};
    state->apply_z(qubit);
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Hadamard gate to the qubit
   *
   * Applies a Hadamard gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyH(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};
    state->apply_h(qubit);
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a S gate to the qubit
   *
   * Applies a S gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyS(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kStabilizer || GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "s";
      op.qubits = AER::reg_t{qubit};

      state->buffer_op(std::move(op));
    } else {
      const AER::cvector_t S = {{1, 0}, complex_t(0.0, 1)};
      state->apply_diagonal_matrix(qubits, S);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a S dagger gate to the qubit
   *
   * Applies a S dagger gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplySDG(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kStabilizer ||
        GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "sdg";
      op.qubits = AER::reg_t{qubit};

      state->buffer_op(std::move(op));
    } else {
      const AER::cvector_t Sdg = {{1, 0}, complex_t(0.0, -1)};
      state->apply_diagonal_matrix(qubits, Sdg);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a T gate to the qubit
   *
   * Applies a T gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyT(Types::qubit_t qubit) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error("T gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit};
    // state->apply_u(qubit, 0, 0, M_PI / 4.0);

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "t";
      op.qubits = AER::reg_t{qubit};
      state->buffer_op(std::move(op));
    } else {
      const AER::cvector_t T = {{1, 0}, std::polar<double>(1, M_PI / 4.)};
      state->apply_diagonal_matrix(qubits, T);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a T dagger gate to the qubit
   *
   * Applies a T dagger gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyTDG(Types::qubit_t qubit) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "TDG gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit};
    // state->apply_u(qubit, 0, 0, -M_PI / 4.0);

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "tdg";
      op.qubits = AER::reg_t{qubit};
      state->buffer_op(std::move(op));
    } else {
      const AER::cvector_t Tdg = {{1, 0}, std::polar<double>(1, -M_PI / 4.)};
      state->apply_diagonal_matrix(qubits, Tdg);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Sx gate to the qubit
   *
   * Applies a Sx gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplySx(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kStabilizer || GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "sx";
      op.qubits = AER::reg_t{qubit};

      state->buffer_op(std::move(op));
    } else {
      // there is a difference in the global phase for this
      // changed to match qcsim behavior (and also the qiskit docs and
      // implementation - not the one exposed in 'contrib', though)
      // state->apply_mcrx({ qubit }, -M_PI / 4.0);
      state->apply_unitary(qubits, AER::Linalg::Matrix::SX);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Sx dagger gate to the qubit
   *
   * Applies a Sx dagger gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplySxDAG(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kStabilizer ||
        GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "sxdg";
      op.qubits = AER::reg_t{qubit};

      state->buffer_op(std::move(op));
    } else
      state->apply_unitary(qubits, AER::Linalg::Matrix::SXDG);

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a K gate to the qubit
   *
   * Applies a K (Hy) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyK(Types::qubit_t qubit) override {
    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kStabilizer ||
        GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplyZ(qubit);
      ApplyS(qubit);
      ApplyH(qubit);
      ApplyS(qubit);
    } else {
      static const cmatrix_t K = AER::Utils::make_matrix<complex_t>(
          {{{1. / std::sqrt(2.), 0}, {0, -1. / std::sqrt(2.)}},
           {{0, 1. / std::sqrt(2.)}, {-1. / std::sqrt(2.), 0}}});

      state->apply_unitary(qubits, K);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Rx gate to the qubit
   *
   * Applies an x rotation gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The rotation angle.
   */
  void ApplyRx(Types::qubit_t qubit, double theta) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "Rx gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplyH(qubit);
      ApplyRz(qubit, theta);
      ApplyH(qubit);
    } else {
      const cmatrix_t rx = AER::Linalg::Matrix::rx(theta);
      state->apply_unitary(qubits, rx);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Ry gate to the qubit
   *
   * Applies a y rotation gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The rotation angle.
   */
  void ApplyRy(Types::qubit_t qubit, double theta) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "Ry gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplySDG(qubit);
      ApplyRx(qubit, theta);
      ApplyS(qubit);
    } else {
      const cmatrix_t ry = AER::Linalg::Matrix::ry(theta);

      state->apply_unitary(qubits, ry);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a Rz gate to the qubit
   *
   * Applies a z rotation gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The rotation angle.
   */
  void ApplyRz(Types::qubit_t qubit, double theta) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "Rz gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit};
    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      // this does not suffice, as it is implemented only for pi/4 multiples by
      // qiskit aer... use p instead, which is implemented for any angle...
      // there is a phase difference, but it is not important for the stabilizer
      // simulation

      // it's very important as many other non-clifford gates are implemented based on rotations!

      /*
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "rz";
      op.qubits = AER::reg_t{qubit};
      op.params = {theta};

      state->buffer_op(std::move(op));
      */
      ApplyP(qubit, theta);
    } else {
      const cmatrix_t rz = AER::Linalg::Matrix::rz(theta);

      state->apply_unitary(qubits, rz);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a U gate to the qubit
   *
   * Applies a U gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The first parameter.
   * @param phi The second parameter.
   * @param lambda The third parameter.
   * @param gamma The fourth parameter.
   */
  void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda,
              double gamma) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error("U gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplyRz(qubit, lambda);
      ApplyRy(qubit, theta);
      ApplyRz(qubit, phi);
    } else {
      const cmatrix_t u = AER::Linalg::Matrix::u4(theta, phi, lambda, gamma);

      state->apply_unitary(qubits, u);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CX gate to the qubits
   *
   * Applies a controlled X gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};
    state->apply_cx(qubits);
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CY gate to the qubits
   *
   * Applies a controlled Y gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplySDG(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyS(tgt_qubit);
    } else {
      state->apply_cy(qubits);
    }
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CZ gate to the qubits
   *
   * Applies a controlled Z gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};
    state->apply_cz(qubits);
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CP gate to the qubits
   *
   * Applies a controlled phase gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param lambda The phase shift angle.
   */
  void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
               double lambda) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CP gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      const double halfAngle = lambda * 0.5;
      ApplyP(ctrl_qubit, halfAngle);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyP(tgt_qubit, -halfAngle);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyP(tgt_qubit, halfAngle);
    } else {
      const cmatrix_t CP = AER::Linalg::Matrix::cphase(lambda);
      // state->apply_mcphase(qubits, lambda);
      state->apply_unitary(qubits, CP);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CRx gate to the qubits
   *
   * Applies a controlled x rotation gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta The rotation angle.
   */
  void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CRx gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      const double halfAngle = theta * 0.5;

      ApplyH(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyRz(tgt_qubit, -halfAngle);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyRz(tgt_qubit, halfAngle);
      ApplyH(tgt_qubit);
    } else {
      cmatrix_t mat(4, 4);
      mat(0, 0) = 1;
      mat(2, 2) = 1;

      const double t2 = theta * 0.5;

      const complex_t i(0., 1.);
      mat(1, 1) = std::cos(t2);
      mat(1, 3) = -i * std::sin(t2);
      mat(3, 1) = mat(1, 3);
      mat(3, 3) = mat(1, 1);

      // state->apply_mcrx(qubits, theta);
      state->apply_unitary(qubits, mat);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CRy gate to the qubits
   *
   * Applies a controlled y rotation gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta The rotation angle.
   */
  void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CRy gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      const double halfAngle = theta * 0.5;

      ApplyRy(tgt_qubit, halfAngle);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyRy(tgt_qubit, -halfAngle);
      ApplyCX(ctrl_qubit, tgt_qubit);
    } else {
      cmatrix_t mat(4, 4);
      mat(0, 0) = 1;
      mat(2, 2) = 1;

      const double t2 = theta * 0.5;

      mat(1, 1) = std::complex<double>(cos(t2), 0);
      mat(1, 3) = std::complex<double>(-sin(t2), 0);
      mat(3, 1) = std::complex<double>(sin(t2), 0);
      mat(3, 3) = mat(1, 1);

      // state->apply_mcry(qubits, theta);
      state->apply_unitary(qubits, mat);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CRz gate to the qubits
   *
   * Applies a controlled z rotation gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta The rotation angle.
   */
  void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CRz gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      const double halfAngle = theta * 0.5;
      ApplyRz(tgt_qubit, halfAngle);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyRz(tgt_qubit, -halfAngle);
      ApplyCX(ctrl_qubit, tgt_qubit);
    } else {
      const double t2 = theta * 0.5;
      AER::cvector_t v = {
          {1, 0}, std::polar(1., -t2), {1, 0}, std::polar(1., t2)};

      // state->apply_mcrz(qubits, theta);
      state->apply_diagonal_matrix(qubits, v);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CH gate to the qubits
   *
   * Applies a controlled Hadamard gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CH gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplyH(tgt_qubit);
      ApplySDG(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyH(tgt_qubit);
      ApplyT(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyT(tgt_qubit);
      ApplyH(tgt_qubit);
      ApplyS(tgt_qubit);
      ApplyX(tgt_qubit);
      ApplyS(ctrl_qubit);
    } else {
      const cmatrix_t CU = AER::Linalg::Matrix::cu(M_PI_2, 0, M_PI, 0);
      state->apply_unitary(qubits, CU);
      // state->apply_cu(qubits, M_PI_2, 0, M_PI, 0);
    }
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CSx gate to the qubits
   *
   * Applies a controlled squared root not gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CSx gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};
    
    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplyH(tgt_qubit);
      ApplyT(ctrl_qubit);
      ApplyT(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyTDG(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyH(tgt_qubit);
    } else {
      static const cmatrix_t CSX = AER::Utils::make_matrix<complex_t>(
          {{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0.5, 0.5}, {0, 0}, {0.5, -0.5}},
           {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
           {{0, 0}, {0.5, -0.5}, {0, 0}, {0.5, 0.5}}});


      state->apply_unitary(qubits, CSX);
    }
    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a CSx dagger gate to the qubits
   *
   * Applies a controlled squared root not dagger gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCSxDAG(Types::qubit_t ctrl_qubit,
                   Types::qubit_t tgt_qubit) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CSxDAG gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      ApplyH(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyT(tgt_qubit);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyTDG(ctrl_qubit);
      ApplyTDG(tgt_qubit);
      ApplyH(tgt_qubit);
    } else {
      static const cmatrix_t CSXDG = AER::Utils::make_matrix<complex_t>(
          {{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0.5, -0.5}, {0, 0}, {0.5, 0.5}},
           {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
           {{0, 0}, {0.5, 0.5}, {0, 0}, {0.5, -0.5}}});

      state->apply_unitary(qubits, CSXDG);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a swap gate to the qubits
   *
   * Applies a swap gate to the specified qubits
   * @param qubit0 The first qubit
   * @param qubit1 The second qubit
   */
  void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override {
    const Types::qubits_vector qubits = {qubit0, qubit1};

    if (GetSimulationType() == SimulationType::kStabilizer || GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "swap";
      op.qubits = AER::reg_t{qubit0, qubit1};

      state->buffer_op(std::move(op));
    } else
      // state->apply_mcswap(qubits);
      state->apply_unitary(qubits, AER::Linalg::Matrix::SWAP);

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a controlled controlled not gate to the qubits
   *
   * Applies a controlled controlled not gate to the specified qubits
   * @param qubit0 The first control qubit
   * @param qubit1 The second control qubit
   * @param qubit2 The target qubit
   */
  void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1,
                Types::qubit_t qubit2) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CCX gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {qubit0, qubit1, qubit2};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      AER::Operations::Op op;
      op.type = AER::Operations::OpType::gate;
      op.name = "ccx";
      op.qubits = AER::reg_t{qubit0, qubit1, qubit2};

      state->buffer_op(std::move(op));
    } else {
      static const cmatrix_t mat = AER::Utils::make_matrix<complex_t>(
          {{{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});

      // state->apply_mcx(qubits);
      state->apply_unitary(qubits, mat);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a controlled swap gate to the qubits
   *
   * Applies a controlled swap gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param qubit0 The first qubit
   * @param qubit1 The second qubit
   */
  void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0,
                  Types::qubit_t qubit1) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CSwap gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, qubit0, qubit1};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      const int q1 = ctrl_qubit;  // control
      const int q2 = qubit0;
      const int q3 = qubit1;

      ApplyCX(q3, q2);
      ApplyCSx(q2, q3);
      ApplyCX(q1, q2);

      ApplyP(q3, M_PI);
      ApplyP(q2, -M_PI_2);

      ApplyCSx(q2, q3);
      ApplyCX(q1, q2);

      ApplyP(q3, M_PI);
      ApplyCSx(q1, q3);
      ApplyCX(q3, q2);
    } else {
      static const cmatrix_t mat = AER::Utils::make_matrix<complex_t>(
          {{{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}},
           {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}}});

      // state->apply_mcswap(qubits);
      state->apply_unitary(qubits, mat);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a controlled U gate to the qubits
   *
   * Applies a controlled U gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta Theta parameter for the U gate
   * @param phi Phi parameter for the U gate
   * @param lambda Lambda parameter for the U gate
   * @param gamma Gamma parameter for the U gate
   */
  void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
               double theta, double phi, double lambda, double gamma) override {
    if (GetSimulationType() == SimulationType::kStabilizer)
      throw std::runtime_error(
          "CU gate not supported in stabilizer simulation");

    const Types::qubits_vector qubits = {ctrl_qubit, tgt_qubit};

    if (GetSimulationType() == SimulationType::kExtendedStabilizer) {
      if (gamma != 0.0) ApplyP(ctrl_qubit, gamma);

      const double lambdaPlusPhiHalf = 0.5 * (lambda + phi);
      const double halfTheta = 0.5 * theta;
      ApplyP(tgt_qubit, 0.5 * (lambda - phi));
      ApplyP(ctrl_qubit, lambdaPlusPhiHalf);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyU(tgt_qubit, -halfTheta, 0, -lambdaPlusPhiHalf, 0);
      ApplyCX(ctrl_qubit, tgt_qubit);
      ApplyU(tgt_qubit, halfTheta, phi, 0, 0);
    } else {
      const cmatrix_t CU = AER::Linalg::Matrix::cu(theta, phi, lambda, gamma);
      // state->apply_mcu(qubits, theta, phi, lambda, gamma);
      state->apply_unitary(qubits, CU);
    }

    NotifyObservers(qubits);
  }

  /**
   * @brief Applies a nop
   *
   * Applies a nop (no operation).
   * Typically does (almost) nothing. Equivalent to an identity.
   * For qiskit aer it will send the 'nop' to the qiskit aer simulator.
   */
  void ApplyNop() override {
    AER::Operations::Op op;
    op.type =
        AER::Operations::OpType::barrier;  // there is also 'nop' but that one
                                           // doesn't work with stabilizer
    state->buffer_op(std::move(op));
  }

  /**
   * @brief Clones the simulator.
   *
   * Clones the simulator, including the state, the configuration and the
   * internally saved state, if any. Does not copy the observers. Should be used
   * mainly internally, to optimise multiple shots execution, copying the state
   * from the simulator used for timing.
   *
   * @return A shared pointer to the cloned simulator.
   */
  std::unique_ptr<ISimulator> Clone() override {
    auto sim = std::make_unique<AerSimulator>();

    // the tricky part is cloning the state
    if (simulationType == SimulationType::kMatrixProductState)
      sim->Configure("method", "matrix_product_state");
    else if (simulationType == SimulationType::kStabilizer)
      sim->Configure("method", "stabilizer");
    else if (simulationType == SimulationType::kTensorNetwork)
      sim->Configure("method", "tensor_network");
    else if (simulationType == SimulationType::kExtendedStabilizer)
      sim->Configure("method", "extended_stabilizer");
    else
      sim->Configure("method", "statevector");

    if (simulationType == SimulationType::kMatrixProductState) {
      if (limitSize)
        sim->Configure("matrix_product_state_max_bond_dimension",
                       std::to_string(chi).c_str());
      if (limitEntanglement)
        sim->Configure("matrix_product_state_truncation_threshold",
                       std::to_string(singularValueThreshold).c_str());
      sim->Configure("mps_sample_measure_algorithm", useMPSMeasureNoCollapse
                                                         ? "mps_probabilities"
                                                         : "mps_apply_measure");
    }

    sim->SetMultithreading(
        enableMultithreading); /**< The multithreading flag. */

    AER::Vector<complex_t> localSavedAmplitudes =
        savedAmplitudes; /**< The amplitudes, saved. */
    AER::Data localSavedState =
        savedState; /**< The saved data - here there will be the saved state of
                       the simulator */

    // now the tricky part
    if (state && state->is_initialized()) {
      SaveState();  // this also restores the state if it's destroyed in the
                    // saving process!
      // now the current state is saved in savedAmplitudes or savedState, so use
      // it to initialize the cloned state, like this
      sim->savedAmplitudes = std::move(savedAmplitudes);
      sim->savedState = std::move(savedState);

      sim->RestoreState();  // now the state is loaded in the cloned simulator

      // those saved previously can be an older state, so put them in the clone,
      // too
      sim->savedAmplitudes = localSavedAmplitudes;
      sim->savedState = localSavedState;

      // those saved previously can be an older state, so put them back
      savedAmplitudes = std::move(localSavedAmplitudes);
      savedState = std::move(localSavedState);
    } else {
      // std::cout << "Restored from non initialized state" << std::endl;
      sim->savedAmplitudes = std::move(localSavedAmplitudes);
      sim->savedState = std::move(localSavedState);

      sim->RestoreState();  // this might not be necessary, but sometimes, not
                            // very often, an exception is thrown about the
                            // state not being initialized, this might prevent
                            // that
    }

    return sim;
  }
};

}  // namespace Private
}  // namespace Simulators

#endif

#endif

#endif
