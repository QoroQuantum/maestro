from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

p = 0.05
one_qubit = pauli_error([
    ("X", p / 3), ("Y", p / 3),
    ("Z", p / 3), ("I", 1 - p),
])
two_qubit = one_qubit.tensor(one_qubit)
noise = NoiseModel()
noise.add_all_qubit_quantum_error(one_qubit, ["h"])
noise.add_all_qubit_quantum_error(two_qubit, ["cx"])

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sim = AerSimulator(noise_model=noise,
                   seed_simulator=7)
result = sim.run(qc, shots=4000).result()
print(result.get_counts())
