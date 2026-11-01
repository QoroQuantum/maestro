from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sim = AerSimulator(seed_simulator=7)
job = sim.run(transpile(qc, sim), shots=1000)
print(job.result().get_counts())
