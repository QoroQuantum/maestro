from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.save_statevector()

sim = AerSimulator(method="statevector")
state = sim.run(qc).result().get_statevector()
print(state)
