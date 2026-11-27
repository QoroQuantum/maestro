"""Stim Clifford gates that the translator builds from Maestro gates.

Each is a noiseless prefix followed by one noisy Maestro gate on the same
qubits, so a NoiseModel adds its gate noise once, after the whole gate, as it
does for a native gate. These tests check every decomposition against Stim's
tableau and check where the noise lands on each simulation method.
"""

import itertools

import numpy as np
import pytest
sinter = pytest.importorskip("sinter")
stim = pytest.importorskip("stim")

import maestro
from maestro.sinter import MaestroCompiledSampler, count_nontrivial_prefix_gates, translate_stim_to_maestro

SWAP_FAMILY = ["ISWAP", "ISWAP_DAG", "CXSWAP", "CZSWAP", "SWAPCX"]
NEW_TWO_QUBIT = SWAP_FAMILY + ["SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ", "SQRT_ZZ_DAG",
                               "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"]
SIGNED_ONE_QUBIT = ["C_NXYZ", "C_XNYZ", "C_XYNZ", "C_NZYX", "C_ZNYX", "C_ZYNX", "H_NXY", "H_NXZ", "H_NYZ"]
NEW_ONE_QUBIT = ["C_XYZ", "C_ZYX", "H_XY", "H_YZ", *SIGNED_ONE_QUBIT]
NATIVE = ["I", "X", "Y", "Z", "H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
          "CX", "CY", "CZ", "SWAP", "II"]
# Shapes other than the swap family: local gates before the noisy one
# (SQRT_ZZ), a noiseless entangling gate before it (SQRT_XX, XCX), a noisy
# gate on the reversed pair (XCZ, XCY), and one-qubit gates with one or two
# noiseless gates before the noisy one (C_XYZ, C_NXYZ).
OTHER_TWO_QUBIT_SHAPES = ["SQRT_XX", "SQRT_ZZ", "XCX", "XCZ", "XCY"]
ONE_QUBIT_SHAPES = ["C_XYZ", "C_NXYZ"]

METHOD = maestro.SimulationType


def _config(method):
    return maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QCSim,
                                   simulation_type=getattr(METHOD, method), fixed_backend=True)


EXACT = _config("DensityMatrix")
requires_stim_1_15 = pytest.mark.skipif("II" not in stim.gate_data(), reason="Stim 1.15 added II and the signed Clifford gates")


def _with_stim_reference(gates):
    """Only tests using Stim to construct these gates depend on its gate vocabulary."""
    return [pytest.param(gate, marks=requires_stim_1_15) if gate == "II" or gate in SIGNED_ONE_QUBIT else gate
            for gate in gates]


def _arity(gate):
    return 2 if stim.gate_data(gate).is_two_qubit_gate else 1


def _targets(gate):
    return " ".join(map(str, range(_arity(gate))))


def _unitary(text, width):
    """Maestro's unitary for a Stim circuit, one basis state at a time (qubit 0 least significant)."""
    columns = []
    for index in range(2**width):
        excited = [q for q in range(width) if index >> q & 1]
        prepare = f"X {' '.join(map(str, excited))}\n" if excited else ""
        circuit, _, _ = translate_stim_to_maestro(prepare + text)
        columns.append(np.asarray(maestro.get_statevector(circuit, _config("Statevector"))))
    return np.array(columns).T


# =============================================================================
# The decompositions are the Stim gates
# =============================================================================


@pytest.mark.parametrize("gate", _with_stim_reference(NATIVE + NEW_ONE_QUBIT + NEW_TWO_QUBIT))
def test_translation_matches_stim_tableau(gate):
    unitary = _unitary(f"{gate} {_targets(gate)}", _arity(gate))
    assert stim.Tableau.from_unitary_matrix(unitary, endian="little") == stim.Tableau.from_named_gate(gate)


@pytest.mark.parametrize("gate", ["ISWAP", "XCZ", "C_XYZ"])
def test_reversed_and_repeated_targets_match_stim(gate):
    """Each target group gets its own gate, in Stim's target order."""
    targets = "1 0 0 1" if _arity(gate) == 2 else "1 0"
    unitary = _unitary(f"{gate} {targets}", 2)
    assert stim.Tableau.from_unitary_matrix(unitary, endian="little") == stim.Circuit(f"{gate} {targets}").to_tableau()


@pytest.mark.parametrize("alias", ["CNOT", "ZCX", "ZCY", "ZCZ", "SWAPCZ", "H_XZ", "SQRT_Z", "SQRT_Z_DAG"])
def test_stim_gate_aliases_translate_from_text_and_tuples(alias):
    """A stim.Circuit names the canonical gate; text and tuples may use any of Stim's names."""
    gate = stim.gate_data(alias).name
    targets = list(range(_arity(gate)))
    unitary = _unitary(f"{alias} {_targets(gate)}", len(targets))
    assert stim.Tableau.from_unitary_matrix(unitary, endian="little") == stim.Tableau.from_named_gate(gate)
    from_tuples, _, _ = translate_stim_to_maestro([("R", [], targets), (alias, [], targets)])
    from_stim, _, _ = translate_stim_to_maestro(stim.Circuit(f"R {_targets(gate)}\n{alias} {_targets(gate)}"))
    state = maestro.get_statevector(from_tuples, _config("Statevector"))
    assert state == pytest.approx(maestro.get_statevector(from_stim, _config("Statevector")))


@pytest.mark.parametrize("gate,targets", [("I", [0]), ("II", [0, 1]), ("H_XZ", [0]), ("CNOT", [0, 1]),
                                          ("C_XYZ", [0]), ("SQRT_ZZ", [0, 1])])
def test_fixed_clifford_gates_reject_arguments(gate, targets):
    with pytest.raises(ValueError, match="takes 0 arguments"):
        translate_stim_to_maestro(f"{gate}(0) {' '.join(map(str, targets))}")
    with pytest.raises(ValueError, match="takes 0 arguments"):
        translate_stim_to_maestro([(gate, [0.0], targets)])


@pytest.mark.parametrize("text", ["ISWAP 0 1 2", "II 0", "SQRT_XX 3", "ISWAP 0 0", "XCZ 1 2 2 2", "CX 1 1"])
def test_two_qubit_gates_need_pairs_of_different_qubits(text):
    with pytest.raises(ValueError, match="pairs|two different qubits"):
        translate_stim_to_maestro(text)


def test_new_gates_count_as_nontrivial_prefix_gates():
    assert count_nontrivial_prefix_gates("ISWAP 0 1 2 3\nXCZ 0 1\nC_XYZ 0 1\nH_NXY 2") == (3, 3)
    assert count_nontrivial_prefix_gates(stim.Circuit("SQRT_XX 0 1\nH_XY 0")) == (1, 1)
    # II is a zero-duration idle, like I.
    assert count_nontrivial_prefix_gates("II 0 1\nI 0") == (0, 0)


def test_targetless_two_qubit_gate_keeps_prefix_trivial():
    prefix = "R 0\nISWAP"
    assert count_nontrivial_prefix_gates(prefix) == (0, 0)
    assert count_nontrivial_prefix_gates(stim.Circuit(prefix)) == (0, 0)
    sampler = MaestroCompiledSampler(stim.Circuit(prefix + "\nM 0"), config=EXACT)
    assert sampler.checkpoint_sim is None


# =============================================================================
# Maestro's two-qubit gates take noiseless=True
# =============================================================================


@pytest.mark.parametrize("method", ["cx", "cy", "cz", "swap"])
def test_noiseless_two_qubit_gate_gets_no_noise_model_noise(method):
    model = maestro.NoiseModel()
    for q in range(2):
        model.set_t1_2q(q, 1.0)  # |1> decays after every two-qubit gate; X stays clean
    records = {}
    for noiseless in (True, False):
        circuit = maestro.circuits.QuantumCircuit()
        circuit.x(0)
        circuit.x(1)
        getattr(circuit, method)(0, 1, noiseless=noiseless)
        circuit.measure([(0, 0), (1, 1)])
        records[noiseless] = circuit.noisy_execute(model, EXACT, shots=50)["counts"]
    ideal = {"cx": "10", "cy": "10", "cz": "11", "swap": "11"}[method]
    assert records[True] == {ideal: 50}
    assert records[False] == {"00": 50}


@requires_stim_1_15
def test_identity_pair_is_a_noiseless_idle():
    model = maestro.NoiseModel()
    model.set_2q_depolarizing(0, 1, 1.0)
    model.set_all_depolarizing(2, 1.0)
    rows = MaestroCompiledSampler(stim.Circuit("R 0 1\nII 0 1\nM 0 1"), noise_model=model,
                                  config=EXACT, seed=1).sample_measurements(200)
    assert not rows.any()


# =============================================================================
# Exact placement on the density matrix
# =============================================================================

PAULIS_1 = ["X", "Y", "Z"]
PAULIS_2 = ["".join(p) for p in itertools.product("IXYZ", repeat=2)][1:]
GAMMAS = (0.15, 0.3)
# A unital channel that is not symmetric in its targets: X on targets[0] and
# Z on targets[1] with probability 0.2 (targets[0] is the low bit).
X_THEN_Z = [np.sqrt(0.8) * np.eye(4), np.sqrt(0.2) * np.kron(np.diag([1, -1]), [[0, 1], [1, 0]])]


@pytest.mark.parametrize("pair", ["0 1", "1 0"])
@pytest.mark.parametrize("gate", NEW_TWO_QUBIT + ["II"])
def test_density_matrix_noise_follows_the_whole_two_qubit_gate(gate, pair):
    """Every layer fires once, after the gate, on its pair, as a native gate's does.

    The reference is the noiseless gate followed by the layers a native gate on
    the pair gets, in the order the model applies them: T1 on each qubit, the
    pair's Kraus channel in its configured target order, then two-qubit
    depolarizing. T1 differs per qubit, so a layer placed before the final SWAP
    of an ISWAP lands on the wrong qubit, and it does not commute with the
    local gates, so noise before them changes the state as well.
    """
    model = maestro.NoiseModel()
    for q, gamma in enumerate(GAMMAS):
        model.set_t1(q, gamma)
    model.set_kraus_channel([0, 1], [k.tolist() for k in X_THEN_Z])
    model.set_2q_depolarizing(0, 1, 0.1)
    text = f"RX 0\nRY 1\n{gate} {pair}"

    noisy, _, _ = translate_stim_to_maestro(text)
    actual = noisy.full_noise_estimate(PAULIS_2, model, config=EXACT)
    assert actual["noise_evaluation"] == "exact_channels"

    reference, _, _ = translate_stim_to_maestro(text)
    if gate != "II":
        for q, gamma in enumerate(GAMMAS):
            reference.channel([q], maestro.QuantumChannel.amplitude_damping(gamma))
        reference.channel([0, 1], maestro.QuantumChannel.from_kraus([k.tolist() for k in X_THEN_Z]))
        reference.channel([0, 1], maestro.QuantumChannel.depolarizing(0.1, 2))
    expected = reference.estimate(PAULIS_2, EXACT)["expectation_values"]
    np.testing.assert_allclose(actual["expectation_values"], expected, atol=1e-10)


@pytest.mark.parametrize("preparation", ["RX 0", "RY 0"])
@pytest.mark.parametrize("gate", NEW_ONE_QUBIT + ["SQRT_Y", "SQRT_Y_DAG"])
def test_density_matrix_noise_follows_the_whole_one_qubit_gate(gate, preparation):
    model = maestro.NoiseModel()
    model.set_t1(0, 0.3)
    model.set_depolarizing(0, 0.1)
    text = f"{preparation}\n{gate} 0"

    noisy, _, _ = translate_stim_to_maestro(text)
    actual = noisy.full_noise_estimate(PAULIS_1, model, config=EXACT)["expectation_values"]

    reference, _, _ = translate_stim_to_maestro(text)
    reference.channel([0], maestro.QuantumChannel.amplitude_damping(0.3))
    reference.channel([0], maestro.QuantumChannel.depolarizing(0.1))
    expected = reference.estimate(PAULIS_1, EXACT)["expectation_values"]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


# =============================================================================
# Statistical placement on the sampled methods
# =============================================================================

SHOTS = 4000
# Five binomial standard deviations at SHOTS for a probability near 1/2.
TOLERANCE = 0.04


def _product_setup(gate):
    """Resets and measurements, in Pauli bases, for which the gate maps a product state to a product state.

    Returns the reset and measurement instructions and the measured bases.
    """
    n = _arity(gate)
    for bases in itertools.product("ZXY", repeat=n):
        prepare = "\n".join(f"R{'' if b == 'Z' else b} {q}" for q, b in enumerate(bases))
        simulator = stim.TableauSimulator()
        simulator.do(stim.Circuit(f"{prepare}\n{gate} {_targets(gate)}"))
        measured = []
        for q in range(n):
            measured += [b for b, peek in (("Z", simulator.peek_z), ("X", simulator.peek_x),
                                           ("Y", simulator.peek_y)) if peek(q)][:1]
        if len(measured) == n:
            measure = "\n".join(f"M{'' if b == 'Z' else b} {q}" for q, b in enumerate(measured))
            return prepare, measure, measured
    raise AssertionError(f"No product-state setup for {gate}")


def _histogram(rows):
    """Probability of each record, as a vector indexed by the record's bits (bit k is column k)."""
    index = rows.astype(int) @ (1 << np.arange(rows.shape[1]))
    return np.bincount(index, minlength=2**rows.shape[1]) / len(rows)


@pytest.mark.parametrize("method", ["Stabilizer", "ExtendedStabilizer", "MatrixProductState", "DensityMatrix"])
@pytest.mark.parametrize("gate", SWAP_FAMILY + ["SQRT_ZZ", "XCZ"])
def test_t1_fires_once_after_the_gate(gate, method):
    """From |11>, each excitation left after the gate survives with 1 - gamma of its qubit.

    ISWAP keeps |11>, so with one gamma on both qubits P(11) = (1 - gamma)^2
    when the noise fires once and (1 - gamma)^4 when it fires after both its CZ
    and its SWAP. Each qubit has its own gamma, so noise that fires before a
    final SWAP lands on the wrong one.
    """
    gammas = np.array([0.3, 0.1])
    model = maestro.NoiseModel()
    for q, gamma in enumerate(gammas):
        model.set_t1_2q(q, gamma)  # the X preparation stays clean
    circuit = stim.Circuit(f"X 0 1\n{gate} 0 1\nM 0 1")
    ideal = circuit.compile_sampler().sample(1)[0]
    sampler = MaestroCompiledSampler(circuit, noise_model=model, config=_config(method), seed=1)
    if method == "Stabilizer":
        assert sampler._reference_sampler is None  # T1 runs shot by shot, not on frames
    rows = sampler.sample_measurements(SHOTS)
    expected = np.prod(1 - gammas[ideal])
    assert np.mean((rows == ideal).all(axis=1)) == pytest.approx(expected, abs=TOLERANCE)


@pytest.mark.parametrize("method", ["ExtendedStabilizer", "MatrixProductState", "DensityMatrix"])
@pytest.mark.parametrize("gate", _with_stim_reference(SWAP_FAMILY + OTHER_TWO_QUBIT_SHAPES + ONE_QUBIT_SHAPES))
@pytest.mark.parametrize("rotation", ["depolarizing", "bit_flip"])
def test_coherent_noise_fires_once_after_the_gate(rotation, gate, method):
    """Coherent rotations land once, after the gate, whatever their random signs.

    The coherent DEPOLARIZE1(p) rotation flips any Pauli eigenstate with
    probability 2p/3; two kicks would add coherently instead, to about 8p/3 on
    one qubit. Its body-diagonal axes map to one another under one-qubit
    Cliffords, so the bit-flip rotation, about X only, also checks that the
    noise does not fire before trailing local gates: it flips a Z or Y
    eigenstate with probability p and leaves an X eigenstate alone. Each qubit
    has its own p, so noise that fires before a final SWAP lands on the wrong one.
    """
    strengths = (0.3, 0.12)[:_arity(gate)]
    model = maestro.NoiseModel()
    prepare, measure, bases = _product_setup(gate)
    if rotation == "depolarizing":
        for q, p in enumerate(strengths):
            model.set_coherent_depolarizing(q, p)
        expected = [2 * p / 3 for p in strengths]
    else:
        for q, p in enumerate(strengths):
            model.set_coherent_bit_flip(q, p)
        expected = [0.0 if basis == "X" else p for basis, p in zip(bases, strengths)]
    circuit = stim.Circuit(f"{prepare}\n{gate} {_targets(gate)}\n{measure}")
    ideal = circuit.compile_sampler().sample(1)[0]
    rows = MaestroCompiledSampler(circuit, noise_model=model, config=_config(method), seed=2).sample_measurements(SHOTS)
    np.testing.assert_allclose(np.mean(rows != ideal, axis=0), expected, atol=TOLERANCE)


@pytest.mark.parametrize("method", ["Stabilizer", "ExtendedStabilizer", "MatrixProductState", "DensityMatrix"])
@pytest.mark.parametrize("gate", SWAP_FAMILY + OTHER_TWO_QUBIT_SHAPES)
def test_noise_model_two_qubit_depolarizing_matches_stim(gate, method):
    """NoiseModel.set_2q_depolarizing on the pair is Stim's DEPOLARIZE2 after the gate."""
    p = 0.2
    model = maestro.NoiseModel()
    model.set_2q_depolarizing(0, 1, p)
    prepare, measure, _ = _product_setup(gate)
    circuit = stim.Circuit(f"{prepare}\n{gate} 0 1\n{measure}")
    sampler = MaestroCompiledSampler(circuit, noise_model=model, config=_config(method), seed=3)
    if method == "Stabilizer":
        assert type(sampler._reference_sampler).__name__ == "ReferenceFrameSampler"
    stim_circuit = stim.Circuit(f"{prepare}\n{gate} 0 1\nDEPOLARIZE2({p}) 0 1\n{measure}")
    expected = _histogram(stim_circuit.compile_sampler(seed=3).sample(200_000))
    np.testing.assert_allclose(_histogram(sampler.sample_measurements(SHOTS)), expected, atol=TOLERANCE)


@pytest.mark.parametrize("method", ["Stabilizer", "ExtendedStabilizer", "MatrixProductState", "DensityMatrix"])
@pytest.mark.parametrize("gate", _with_stim_reference(ONE_QUBIT_SHAPES + ["H_YZ"]))
def test_noise_model_one_qubit_depolarizing_matches_stim(gate, method):
    p = 0.3
    model = maestro.NoiseModel()
    model.set_1q_gate_depolarizing(0, p)
    prepare, measure, _ = _product_setup(gate)
    circuit = stim.Circuit(f"{prepare}\n{gate} 0\n{measure}")
    rows = MaestroCompiledSampler(circuit, noise_model=model, config=_config(method), seed=4).sample_measurements(SHOTS)
    stim_circuit = stim.Circuit(f"{prepare}\n{gate} 0\nDEPOLARIZE1({p}) 0\n{measure}")
    expected = _histogram(stim_circuit.compile_sampler(seed=4).sample(200_000))
    np.testing.assert_allclose(_histogram(rows), expected, atol=TOLERANCE)


# =============================================================================
# Stim's own Pauli noise in the circuit
# =============================================================================


@pytest.mark.parametrize("gate", _with_stim_reference(NEW_ONE_QUBIT + NEW_TWO_QUBIT))
def test_stim_pauli_noise_through_each_gate_matches_stim(gate):
    """Noise before, between and after a gate and its inverse, measured where the ideal record is fixed."""
    targets = _targets(gate)
    bases = "XY"[:_arity(gate)]
    prepare = "\n".join(f"R{b} {q}" for q, b in enumerate(bases))
    measure = "\n".join(f"M{b} {q}" for q, b in enumerate(bases))
    middle = "DEPOLARIZE2(0.1) 0 1" if _arity(gate) == 2 else "DEPOLARIZE1(0.1) 0"
    text = (f"{prepare}\nDEPOLARIZE1(0.05) {targets}\n{gate} {targets}\n{middle}\n"
            f"{stim.gate_data(gate).inverse.name} {targets}\nZ_ERROR(0.05) 0\n{measure}")
    circuit = stim.Circuit(text)
    rows = MaestroCompiledSampler(circuit, seed=5).sample_measurements(SHOTS)
    expected = _histogram(circuit.compile_sampler(seed=5).sample(200_000))
    np.testing.assert_allclose(_histogram(rows), expected, atol=TOLERANCE)


def _cx_from_iswap(circuit):
    """Rebuild every CX from two ISWAPs: CX a b = H b; S b; H b; S_DAG a; ISWAP a b; H a; ISWAP a b."""
    rebuilt = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name != "CX":
            rebuilt.append(instruction)
            continue
        pairs = [t.value for t in instruction.targets_copy()]
        controls, targets = pairs[0::2], pairs[1::2]
        for name, qubits in (("H", targets), ("S", targets), ("H", targets), ("S_DAG", controls),
                             ("ISWAP", pairs), ("H", controls), ("ISWAP", pairs)):
            rebuilt.append(name, qubits)
    return rebuilt


def test_surface_code_with_cx_rebuilt_from_iswap_matches_stim():
    """A noisy rotated surface code whose CXs are ISWAPs: Stim runs ISWAP natively, Maestro decomposes it."""
    original = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=0.005, before_round_data_depolarization=0.005,
        after_reset_flip_probability=0.005, before_measure_flip_probability=0.005)
    circuit = _cx_from_iswap(original)
    names = {instruction.name for instruction in circuit}
    assert "ISWAP" in names and "CX" not in names
    # The rewrite leaves every noise channel where it was.
    assert circuit.detector_error_model() == original.flattened().detector_error_model()

    shots = 50_000
    sampler = MaestroCompiledSampler(circuit, decoder="pymatching", seed=11)
    dets, obs = sampler.sample_detection_events(shots)
    stim_dets, stim_obs = circuit.compile_detector_sampler(seed=11).sample(shots, separate_observables=True)

    def deviations(a, b):
        """Difference of two binomial rates over shots each, in standard deviations."""
        pooled = (a + b) / 2
        return np.abs(a - b) / np.sqrt(np.maximum(2 * pooled * (1 - pooled) / shots, 1e-12))

    assert deviations(dets.mean(axis=0), stim_dets.mean(axis=0)).max() < 5
    assert deviations(obs.mean(), stim_obs.mean()) < 5
    # The decoded logical error rate.
    errors = sampler.sample(shots=shots, seed=12).errors
    matching = sinter.BUILT_IN_DECODERS["pymatching"].compile_decoder_for_dem(
        dem=circuit.detector_error_model(decompose_errors=True))
    packed_dets, packed_obs = circuit.compile_detector_sampler(seed=12).sample(
        shots, separate_observables=True, bit_packed=True)
    predictions = matching.decode_shots_bit_packed(bit_packed_detection_event_data=packed_dets)
    stim_errors = int(np.any(predictions != packed_obs, axis=1).sum())
    assert deviations(errors / shots, stim_errors / shots) < 5
