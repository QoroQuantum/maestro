from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Maestro puts qubit 0 on the left:
# "ZII" is Z on qubit 0.
result = qc.estimate(["ZII", "XXX", "ZZI"])
z0, xxx, zz01 = result["expectation_values"]

# Weight Hamiltonian terms yourself.
print([z0, xxx, 0.5 * zz01 + 0.3 * xxx])
