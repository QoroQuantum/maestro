from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

state = qc.get_statevector()
print(state)
