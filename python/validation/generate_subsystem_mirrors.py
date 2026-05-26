import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, qasm2
import os

SAVE_TO_DIR = "windowed_mirror_circuits_iqm"
os.makedirs(SAVE_TO_DIR, exist_ok=True)

def generate_windowed_mirror_qasm(N: int, W: int, window_start: int, depth: int) -> str:
    """
    Generates an OpenQASM 2.0 string for a subsystem mirror circuit.

    Args:
        N (int): Total number of qubits on the device.
        W (int): Size of the target subsystem window.
        window_start (int): Starting index for the W-qubit window.
        depth (int): Number of entanglement layers for the random unitary U.

    Returns:
        str: The generated OpenQASM 2.0 circuit.
    """
    if window_start + W > N:
        raise ValueError("Window exceeds the total number of qubits (N).")

    # Initialize registers: N quantum bits, W classical bits for partial measurement
    qr = QuantumRegister(N, 'q')
    cr = ClassicalRegister(W, 'c')
    qc = QuantumCircuit(qr, cr)

    # Define indices
    target_qubits = list(range(window_start, window_start + W))
    spectator_qubits = [i for i in range(N) if i not in target_qubits]

    # 1. Build the random W-qubit subcircuit U
    U_circ = QuantumCircuit(W, name="U_rand")
    for _ in range(depth):
        # Random single-qubit rotations
        for i in range(W):
            U_circ.rx(np.random.uniform(0, 2 * np.pi), i)
            U_circ.ry(np.random.uniform(0, 2 * np.pi), i)
        # Linear nearest-neighbor entanglement
        for i in range(W - 1):
            U_circ.cx(i, i + 1)

    # 2. Apply U to the target window
    qc.append(U_circ.to_instruction(), target_qubits)

    # 3. Apply dummy operations to spectators (simulating activity)
    for q in spectator_qubits:
        for _ in range(depth * 2):
            qc.rx(np.random.uniform(0, 2 * np.pi), q)

    # 4. Apply U_dagger to the target window
    qc.append(U_circ.inverse().to_instruction(), target_qubits)

    # 5. Apply dummy operations again to spectators during the unscrambling phase
    for q in spectator_qubits:
        for _ in range(depth * 2):
            qc.rx(np.random.uniform(0, 2 * np.pi), q)

    # 6. Partial Measurement: Measure ONLY the target W qubits
    qc.measure(target_qubits, range(W))

    # Decompose custom instructions into standard gates for QASM export
    decomposed_qc = qc.decompose()

    return qasm2.dump(decomposed_qc, f"{SAVE_TO_DIR}/subsystem_mirror_{window_start//W:02}.qasm" )


# Example Usage:
# Total qubits = 54, Window = 5 qubits, starting at qubit 10, depth = 3
if __name__ == "__main__":
    for i in range(0, 16,5):
        qasm_string = generate_windowed_mirror_qasm(N=20, W=5, window_start=i, depth=1)

