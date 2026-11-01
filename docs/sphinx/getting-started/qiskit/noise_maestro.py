import maestro
from maestro.circuits import QuantumCircuit

p = 0.05
# After every gate, on each qubit it touches.
noise = maestro.NoiseModel()
noise.set_all_depolarizing(num_qubits=2, p=p)

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# One noise realization per shot, as in Aer.
result = maestro.noisy_execute(
    qc, noise, shots=4000,
    noise_realizations=4000, noise_seed=7,
)
print(result["counts"])
