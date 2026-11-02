from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(50)
qc.h(0)
for q in range(49):
    qc.cx(q, q + 1)
qc.measure_all()

sim = AerSimulator(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=64,
    matrix_product_state_truncation_threshold=1e-10,
)
result = sim.run(qc, shots=1000).result()
print(len(result.get_counts()), "distinct outcomes")
