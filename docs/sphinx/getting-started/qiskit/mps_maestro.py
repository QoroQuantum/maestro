import maestro
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
for q in range(49):
    qc.cx(q, q + 1)
qc.measure_all()

mps = maestro.SimulationType.MatrixProductState
config = maestro.SimulatorConfig(
    simulation_type=mps,
    max_bond_dimension=64,
    singular_value_threshold=1e-10,
)
result = qc.execute(shots=1000, config=config)
print(len(result["counts"]), "distinct outcomes")
