import maestro
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

config = maestro.SimulatorConfig(seed=7)
result = qc.execute(shots=1000, config=config)
print(result["counts"])
