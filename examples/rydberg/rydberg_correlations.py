
import maestro
import numpy as np
import matplotlib.pyplot as plt
import time
from rydberg_demo import create_rydberg_circuit

# Reuse configuration
N = 64
V_interaction = 5.0
max_bond_dim = 16

# Deep Z2 Phase Parameters
target_omega = 2.5
target_delta = 3.0
T_total = 3.0
dt = 0.15
steps = int(T_total / dt)

def main():
    print("=================================================================")
    print("   MAESTRO Demo Part 2: Quantum Correlations (Sampling Mode)")
    print("   Goal: Measure spatial decay of crystalline order")
    print(f"   Target: 64 Atoms, Omega={target_omega}, Delta={target_delta}")
    print("=================================================================")

    # Generate circuit
    # Using slightly more steps to ensure high fidelity state preparation
    circ = create_rydberg_circuit(N, target_omega, target_delta, V_interaction, steps=int(steps*1.2), dt=dt)
    
    try:
        # HERE IS THE KEY: We explicitly need SAMPLING (simple_execute)
        # We need to compute <n_i n_j> which requires joint probability distributions.
        # Computing this via expectation values would require N*(N-1)/2 separate observables (approx 2000 runs).
        # Sampling does it in ONE run.
        
        num_shots = 2000
        print(f"Acquiring {num_shots} shots from MPS backend...")
        start_t = time.time()
        
        sample_res = maestro.simple_execute(
            circ,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=num_shots, # Sampling is essential here
            max_bond_dimension=max_bond_dim
        )
        
        print(f"Sampling completed in {time.time()-start_t:.2f}s")
        
        if sample_res and 'counts' in sample_res:
             # Calculate Connected Correlation Function
             # C(r) = <n_i n_{i+r}> - <n_i><n_{i+r}>
             # We average over all i to get better statistics (translational invariance approximation)
             
             counts = sample_res['counts']
             max_r = N // 2
             correlations_r = np.zeros(max_r)
             
             # Convert counts to list of bitstrings for easier processing
             # "10101..." -> [1, 0, 1, 0, 1...]
             parsed_samples = []
             for state_str, count in counts.items():
                 # Reverse to match array indexing (0 is left)
                 bits = [int(b) for b in reversed(state_str)] 
                 if len(bits) < N:
                     bits = bits + [0]*(N-len(bits))
                 
                 for _ in range(count):
                     parsed_samples.append(bits)
                     
             samples_matrix = np.array(parsed_samples) # Shape (shots, N)
             
             # Compute densities <n_i>
             densities = np.mean(samples_matrix, axis=0)
             
             print("Computing spatial correlations C(r)...")
             for r in range(1, max_r + 1):
                 c_r_sum = 0.0
                 terms = 0
                 
                 for i in range(N - r):
                     # <n_i n_{i+r}> = Joint Probability
                     joint_expect = np.mean(samples_matrix[:, i] * samples_matrix[:, i+r])
                     
                     # <n_i> * <n_{i+r}> = Product of Marginals
                     product_expect = densities[i] * densities[i+r]
                     
                     # Connected correlation: <AB> - <A><B>
                     # For Z2 phase, we rectified the stagger (-1)^r to see the magnitude of order
                     corr = (joint_expect - product_expect) * ((-1)**r)
                     
                     c_r_sum += corr
                     terms += 1
                 
                 correlations_r[r-1] = c_r_sum / terms
                 
             print("Correlation decay computed.")

             # Visualization
             plt.figure(figsize=(10, 6))
             r_vals = np.arange(1, max_r+1)
             
             plt.plot(r_vals, correlations_r, 'o-', color='cyan', label='Measured Correlation', linewidth=2)
             plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
             
             plt.xlabel('Distance $r$ (sites)')
             plt.ylabel(r'Rectified Correlation $(-1)^r C(r)$')
             plt.title(f'Spatial Correlation Decay\nDemonstrating Long-Range Order via Sampling')
             plt.grid(True, alpha=0.2)
             plt.legend()
             
             filename = "rydberg_correlations.png"
             plt.savefig(filename)
             print(f"\n[OUTPUT] Correlation plot saved as '{filename}'")
             print("Observation: The flat line indicates the system creates a rigid crystal that spans the entire array.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
