import maestro

def main():
    print("Maestro Python Bindings: Expectation Values Example")

    # Define a GHZ state circuit in QASM 2.0
    ghz_qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
"""

    print("\n--- Testing simple_estimate ---")
    # Define observables as a semicolon-separated string of Pauli operators
    # We want to measure parity (ZZZ) and some local observables
    observables = "ZZZ;III;XII;IXI;IIX;XXX"

    # Execute the estimation
    # By default it uses QCSim with Statevector simulation
    result = maestro.simple_estimate(ghz_qasm, observables)

    if result:
        print(f"Simulator: {result['simulator']}")
        print(f"Method:    {result['method']}")
        print(f"Time Taken: {result['time_taken']:.6f} seconds")

        obs_list = observables.split(';')
        exp_vals = result['expectation_values']

        print("\nExpectation Values:")
        for obs, val in zip(obs_list, exp_vals):
            print(f"  <{obs}> = {val:.4f}")
    else:
        print("Error: simple_estimate failed.")

    print("\n--- Testing direct ExpectationValue calls ---")
    # You can also use the object-oriented interface for more control
    m = maestro.Maestro()
    sim_handle = m.CreateSimulator(
        maestro.SimulatorType.QCSim,
        maestro.SimulationType.Statevector
    )
    simulator = m.GetSimulator(sim_handle)

    if simulator:
        # Prepare the state manually
        simulator.AllocateQubits(3)
        simulator.Initialize()

        simulator.ApplyH(0)
        simulator.ApplyCX(0, 1)
        simulator.ApplyCX(1, 2)

        # Measure individual observables
        zzz_val = simulator.ExpectationValue("ZZZ")
        xxx_val = simulator.ExpectationValue("XXX")

        print(f"Direct <ZZZ> = {zzz_val:.4f}")
        print(f"Direct <XXX> = {xxx_val:.4f}")

        m.DestroySimulator(sim_handle)
    else:
        print("Error: Failed to create simulator.")

if __name__ == "__main__":
    main()
