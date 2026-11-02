from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Qiskit puts qubit 0 on the right:
# "IIZ" is Z on qubit 0.
observables = [
    SparsePauliOp("IIZ"),
    SparsePauliOp("XXX"),
    SparsePauliOp(["IZZ", "XXX"], coeffs=[0.5, 0.3]),
]
job = EstimatorV2().run([(qc, observables)])
print(job.result()[0].data.evs)
