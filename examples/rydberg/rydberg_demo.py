
import maestro
import numpy as np
import matplotlib.pyplot as plt
import time

def create_rydberg_circuit(num_atoms, omega, delta, interaction_strength, steps, dt):
    """
    Generates a QASM circuit simulating the adiabatic preparation of a Rydberg atom array state.
    Hamiltonian: H = Omega/2 * Sum(X) - Delta * Sum(n) + V * Sum(n_i n_{i+1})
    """
    
    qasm = []
    qasm.append("OPENQASM 2.0;")
    qasm.append('include "qelib1.inc";')
    qasm.append(f"qreg q[{num_atoms}];")
    qasm.append(f"creg c[{num_atoms}];")
    
    # Param ramp schedule
    # Start: Omega=0, Delta = large negative (Deep in Disordered Phase |00...0>)
    delta_start = -5.0
    
    for s in range(steps):
        t_frac = s / steps # Start exactly at 0 to avoid sudden quench
        
        curr_omega = t_frac * omega
        curr_delta = delta_start + t_frac * (delta - delta_start)
        
        # 1. ZZ Interactions (Rydberg Blockade)
        # theta = V * dt / 2
        zz_theta = interaction_strength * dt / 2
        
        # Even bonds
        for i in range(0, num_atoms - 1, 2):
            qasm.append(f"cx q[{i}], q[{i+1}];")
            qasm.append(f"rz({zz_theta}) q[{i+1}];")
            qasm.append(f"cx q[{i}], q[{i+1}];")
            
        # Odd bonds
        for i in range(1, num_atoms - 1, 2):
            qasm.append(f"cx q[{i}], q[{i+1}];")
            qasm.append(f"rz({zz_theta}) q[{i+1}];")
            qasm.append(f"cx q[{i}], q[{i+1}];")
            
        # 2. Single Qubit Gates
        for i in range(num_atoms):
            # Drive: Rx(Omega * dt)
            qasm.append(f"rx({curr_omega * dt}) q[{i}];")
            
            # Detuning: This was the bug!
            # Correct sign is POSITIVE.
            z_angle = curr_delta * dt
            
            # Interaction Corrections (Single Z terms from V)
            neighbors = 0
            if i > 0: neighbors += 1
            if i < num_atoms - 1: neighbors += 1
            
            z_angle += neighbors * (-interaction_strength * dt / 2)
            
            qasm.append(f"rz({z_angle}) q[{i}];")
            
    qasm.append(f"measure q -> c;")
    return "\n".join(qasm)

def calculate_order_parameter_from_expectations(z_expects, num_atoms):
    """
    Calculates the Z2 Staggered Magnetization Order Parameter from local Z expectation values.
    O = |Sum_i (-1)^i <n_i>| / (N/2)
    where <n_i> = (1 - <Z_i>)/2
    """
    stag_mag = 0.0
    for i, z_val in enumerate(z_expects):
        # <n_i> = (1 - <Z_i>) / 2
        n_val = (1.0 - z_val) / 2.0
        stag_mag += ((-1)**i) * n_val
        
    return abs(stag_mag) / (num_atoms / 2)

def main():
    print("=================================================================")
    print("   MAESTRO demo: 64-Atom Rydberg Array Digital Twin")
    print("   Simulating Adiabatic Preparation of Z2 Ordered Phase")
    print("=================================================================")
    
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    # Set to True to unleash computational beast mode (GPU territory)
    # Set to False for quick demo on laptop CPU
    USE_HPC_PRESET = False 
    
    if USE_HPC_PRESET:
        print(">>> HPC/GPU CHALLENGE MODE ENABLED <<<")
        N = 100                 # increased from 64
        max_bond_dim = 128      # increased from 16 (Massive difference in compute)
        
        # Adiabatic Parameters
        # We need simpler time steps for the demo to finish in reasonable time on CPU 
        # even with high bond dim, otherwise it will take hours.
        # Let's keep steps reasonable but high enough to feel the weight.
        T_total = 4.0
        dt = 0.1
        steps = int(T_total / dt) # 40 steps
        
        # Reduce grid size so the user doesn't wait all day for the sweep
        # effectively showing the "cost per point" increase.
        grid_size = 5           
        
    else:
        print(">>> FAST LAPTOP DEMO MODE <<<")
        N = 64
        max_bond_dim = 16
        T_total = 3.0
        dt = 0.15
        steps = int(T_total / dt)
        grid_size = 12

    V_interaction = 5.0
    min_delta = -1.0
    max_delta = 4.0
    max_omega = 3.0
    
    # ---------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------
    print(f"[Configuration]")
    print(f"  System Size: {N} atoms")
    print(f"  Grid: {grid_size}x{grid_size}")
    print(f"  Steps: {steps} (dt={dt})")
    print(f"  Backend: Matrix Product State (max_bond_dim={max_bond_dim})")
    print("-" * 65)
    print("Sweeping Phase Space (Detuning vs Rabi Frequency)...")
    print("Legend: [·] Disordered  [▒] Intermediate  [█] Z2 Ordered")
    print("-" * 65)

    deltas = np.linspace(min_delta, max_delta, grid_size)
    omegas = np.linspace(0.1, max_omega, grid_size)
    
    # Store results (Omega axis, Delta axis)
    heatmap_data = np.zeros((grid_size, grid_size))
    
    start_time = time.time()

    # Pre-generate observable strings: "ZII...", "IZI...", etc.
    # We want <Z_i> for all i.
    observables_list = []
    for i in range(N):
        # Create list of 'I's
        term = ['I'] * N
        # Set i-th element to 'Z'
        term[i] = 'Z'
        observables_list.append("".join(term))
        
    observables_str = ";".join(observables_list)
    
    # Loop over Omega (Y-axis, reversed for printing top-down)
    for i, omega in enumerate(reversed(omegas)):
        row_label = f"Ω={omega:4.2f} | "
        row_viz = ""
        
        for j, delta in enumerate(deltas):
            # Generate Circuit
            qasm_circ = create_rydberg_circuit(N, omega, delta, V_interaction, steps, dt)
            
            try:
                # Execute using Maestro MPS Estimate
                # This computes expectation values directly without sampling noise!
                res = maestro.simple_estimate(
                    qasm_circ,
                    observables_str,
                    simulator_type=maestro.SimulatorType.QCSim,
                    simulation_type=maestro.SimulationType.MatrixProductState,
                    max_bond_dimension=max_bond_dim
                )
                
                if res and 'expectation_values' in res:
                    z_expects = res['expectation_values']
                    op = calculate_order_parameter_from_expectations(z_expects, N)
                    
                    # Store in matrix. 
                    heatmap_data[grid_size - 1 - i, j] = op
                    
                    if op > 0.6: char = "█"
                    elif op > 0.3: char = "▒"
                    else: char = "·"
                else:
                    char = "X"
                    
            except Exception as e:
                # print(e)
                char = "!"
                
            row_viz += char + " "
            
        print(row_label + row_viz)
        
    total_time = time.time() - start_time
    print("-" * 65)
    print(f"Sweep Completed in {total_time:.2f}s")
    
    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    try:
        plt.figure(figsize=(10, 8))
        # extent = [left, right, bottom, top]
        extent = [min_delta, max_delta, 0.1, max_omega]
        
        plt.imshow(heatmap_data, origin='lower', extent=extent, cmap='inferno', aspect='auto')
        plt.colorbar(label='Z2 Staggered Magnetization')
        plt.xlabel(r'Detuning ($\Delta$)')
        plt.ylabel(r'Rabi Frequency ($\Omega$)')
        plt.title(f'Rydberg Atom Array Phase Diagram\nAdiabatic Preparation (N={N})')
        
        # Add annotation for regions
        plt.text(max_delta*0.8, 0.2, "Z2 Phase", color='white', ha='center', weight='bold')
        plt.text(min_delta + 0.5, 0.2, "Disordered", color='white', ha='center')
        
        filename = "rydberg_phase_diagram.png"
        plt.savefig(filename)
        print(f"\n[OUTPUT] Phase diagram saved as '{filename}'")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
