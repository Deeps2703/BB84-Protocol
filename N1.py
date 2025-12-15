from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt

# Define quaternionic rotation gates
def quaternionic_rotation(theta, axis):
    """
    Create a quaternionic rotation matrix.
    :param theta: Rotation angle in radians.
    :param axis: Axis of rotation ('x', 'y', or 'z').
    :return: 2x2 unitary matrix representing the quaternionic rotation.
    """
    if axis == 'x':
        return np.array([[np.cos(theta / 3), -1j * np.sin(theta / 3)],
                         [-1j * np.sin(theta / 3), np.cos(theta / 3)]], dtype=complex)
    elif axis == 'y':
        return np.array([[np.cos(theta / 3), -np.sin(theta / 3)],
                         [np.sin(theta / 3), np.cos(theta / 3)]], dtype=complex)
    elif axis == 'z':
        return np.array([[np.exp(-1j * theta / 3), 0],
                         [0, np.exp(1j * theta / 3)]], dtype=complex)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

# Encoding function using quaternionic gates
def encode_message_quaternion(bits, bases):
    """Create a quantum circuit for encoding using quaternionic operations."""
    circuits = []
    for bit, basis in zip(bits, bases):
        qc = QuantumCircuit(1, 1)

        if basis == 0:  # Z-basis encoding
            if bit == 1:
                qc.unitary(quaternionic_rotation(np.pi, 'z'), [0], label="Rz_Q")
        else:  # X-basis encoding
            if bit == 0:
                qc.unitary(quaternionic_rotation(np.pi / 2, 'x'), [0], label="Rx_Q")
            else:
                qc.unitary(quaternionic_rotation(np.pi, 'z'), [0], label="Rz_Q")
                qc.unitary(quaternionic_rotation(np.pi / 2, 'x'), [0], label="Rx_Q")

        circuits.append(qc)
    return circuits

# Measurement function using quaternionic gates
def measure_message_quaternion(circuits, bases):
    """Apply quaternionic measurement bases."""
    measured_circuits = []
    for qc, basis in zip(circuits, bases):
        measured_qc = qc.copy()
        if basis == 1:  # X-basis requires quaternionic Hadamard equivalent before measurement
            measured_qc.unitary(quaternionic_rotation(np.pi / 2, 'x'), [0], label="Rx_Q")
        measured_qc.measure(0, 0)
        measured_circuits.append(measured_qc)
    return measured_circuits

# Remove mismatched bases
def remove_garbage(a_bases, b_bases, bits):
    return [bit for i, bit in enumerate(bits) if a_bases[i] == b_bases[i]]

# Full circuit creation with quaternionic gates
def create_full_circuit_quaternion(alice_bits, alice_bases, bob_bases):
    n_qubits = len(alice_bits)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Alice prepares states
    for i in range(n_qubits):
        if alice_bases[i] == 0:  # Z-basis
            if alice_bits[i] == 1:
                qc.unitary(quaternionic_rotation(np.pi, 'z'), [i], label="Rz_Q")
        else:
            if alice_bits[i] == 0:
                qc.unitary(quaternionic_rotation(np.pi / 2, 'x'), [i], label="Rx_Q")
            else:
                qc.unitary(quaternionic_rotation(np.pi, 'z'), [i], label="Rz_Q")
                qc.unitary(quaternionic_rotation(np.pi / 2, 'x'), [i], label="Rx_Q")

    qc.barrier()

    # Bob's measurement
    for i in range(n_qubits):
        if bob_bases[i] == 1:
            qc.unitary(quaternionic_rotation(np.pi / 2, 'x'), [i], label="Rx_Q")
    qc.measure_all()

    return qc

# BB84 protocol implementation with quaternionic gates
def bb84_quaternion_protocol(n_bits=24):
    # Alice's random bits and bases
    alice_bits = np.random.randint(2, size=n_bits)
    alice_bases = np.random.randint(2, size=n_bits)

    # Encoding with quaternionic operations
    message = encode_message_quaternion(alice_bits, alice_bases)

    # Bob's random measurement bases
    bob_bases = np.random.randint(2, size=n_bits)
    bob_circuits = measure_message_quaternion(message, bob_bases)

    # Full quaternionic circuit
    full_circuit = create_full_circuit_quaternion(alice_bits, alice_bases, bob_bases)

    # Use AerSimulator for local simulation
    simulator = AerSimulator()
    transpiled_circuits = transpile(bob_circuits, simulator)

    # Run the circuits using the simulator
    job = simulator.run(transpiled_circuits, shots=1024)
    result = job.result()

    # Extract measurement outcomes and probabilities
    counts_list = [result.get_counts(i) for i in range(len(bob_circuits))]
    probabilities = []
    for count in counts_list:
        total_counts = sum(count.values())
        prob0 = count.get('0', 0) / total_counts
        prob1 = count.get('1', 0) / total_counts
        probabilities.append({'0': prob0, '1': prob1})

    # Generate probability plot
    plt.figure(figsize=(18,8))
    
    # Create subplots for better spacing
    ax = plt.subplot(111)
    x_indices = np.arange(n_bits)
    width = 0.4
    
    # Plot bars with better visual separation
    rects1 = ax.bar(x_indices - width/2, [p['0'] for p in probabilities], 
                   width, label='0', color='green', edgecolor='black')
    rects2 = ax.bar(x_indices + width/2, [p['1'] for p in probabilities], 
                   width, label='1', color='lightgreen', edgecolor='black')

    # Add labels and title
    ax.set_xlabel('Qubit Index', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Measurement Outcome Probabilities per Qubit (1024 shots)', fontsize=14)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'Q{i}' for i in range(n_bits)])
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('measurement_probabilities.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Generate frequency plot for all qubits on one graph
    all_outcomes = []
    all_frequencies = []
    qubit_labels = []

    for i, count in enumerate(counts_list):
        for outcome, frequency in count.items():
            all_outcomes.append(f'Q{i}: {outcome}')  # Label each outcome with qubit index
            all_frequencies.append(frequency)
            

    plt.figure(figsize=(20, 8))  # Adjust figure size as needed
    plt.bar(all_outcomes, all_frequencies)
    plt.xlabel('Qubit and Measurement Outcome', fontsize=12)
    plt.ylabel('Frequency (1024 shots)', fontsize=12)
    plt.title('Measurement Frequencies for All Qubits', fontsize=14)
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.savefig('measurement_frequencies_all.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Generate sifted keys
    alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
    bob_key = remove_garbage(alice_bases, bob_bases, [int(list(count.keys())[0], 2) for count in counts_list])

    # Prepare final output
    output = {
        'initial_bits': alice_bits.tolist(),
        'alice_bases': alice_bases.tolist(),
        'bob_bases': bob_bases.tolist(),
        'bob_results': [int(list(count.keys())[0], 2) for count in counts_list],
        'alice_key': alice_key,
        'bob_key': bob_key,
        'probabilities': probabilities,
        'measurement_plot': 'measurement_probabilities.png',
        'full_circuit': full_circuit
    }
    return output

# Analyze results of the protocol
def analyze_results_quaternion(results):
    print("Initial Bits:")
    print(f"Alice's bits: {results['initial_bits']}")
    print(f"Alice's bases: {results['alice_bases']}")
    print(f"Bob's bases: {results['bob_bases']}")
    print(f"Bob's results: {results['bob_results']}")
    
    print("\nComparison of Bases:")
    matching_bases = [i for i in range(len(results['alice_bases'])) 
                     if results['alice_bases'][i] == results['bob_bases'][i]]
    print(f"Matching bases indices: {matching_bases}")
    
    print("\nMeasurement Probabilities:")
    for i, prob in enumerate(results['probabilities']):
        print(f"Qubit {i}: 0={prob['0']:.3f}, 1={prob['1']:.3f}")
    
    print("\nFinal Keys:")
    print(f"Alice's key: {results['alice_key']}")
    print(f"Bob's key: {results['bob_key']}")
    print(f"\nMeasurement plot saved to: {results['measurement_plot']}")

if __name__ == "__main__":
    try:
        print("Executing Quaternionic BB84 protocol using Qiskit Aer Simulator...")
        results = bb84_quaternion_protocol(n_bits=24)
        analyze_results_quaternion(results)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
