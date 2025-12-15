from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import math  # FIXED: Added missing import
#from qiskit_ibm_provider import IBMProvider
#IBMProvider.save_account("un0Nxdi9hHI2o9OxBSrLiE9yPyhMi8J8do18yvBmaTz0", overwrite=True)
# Define quaternionic rotation gates
def quaternionic_rotation(theta, axis):
    
    if axis == 'x':
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                         [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    elif axis == 'y':
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    elif axis == 'z':
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

# Encoding function using quaternionic gates
def encode_message_quaternion(bits, bases):
    circuits = []
    for bit, basis in zip(bits, bases):
        qc = QuantumCircuit(1, 1)
        
        if basis == 0:  # Z-basis encoding
            if bit == 1:
                qc.x(0)  # |1⟩ state
        else:  # X-basis encoding  
            if bit == 0:
                # Create |+⟩ = (|0⟩ + |1⟩)/√2
                qc.unitary(quaternionic_rotation(np.pi / 2, 'y'), [0])
            else:
                # Create |-⟩ = (|0⟩ - |1⟩)/√2  
                qc.x(0)
                qc.unitary(quaternionic_rotation(np.pi / 2, 'y'), [0])
        
        circuits.append(qc)
    return circuits

# Measurement function using quaternionic gates
def measure_message_quaternion(circuits, bases):
    measured_circuits = []
    for qc, basis in zip(circuits, bases):
        measured_qc = qc.copy()
        if basis == 1:  # X-basis measurement
            # Rotate X-basis states to Z-basis for measurement
            measured_qc.unitary(quaternionic_rotation(-np.pi / 2, 'y'), [0])
        measured_qc.measure(0, 0)
        measured_circuits.append(measured_qc)
    return measured_circuits

# FIXED: Function to re-prepare states after Eve's measurement
def eve_resend_states(eve_results, eve_bases):
    
    circuits = []
    for result, basis in zip(eve_results, eve_bases):
        qc = QuantumCircuit(1, 1)
        
        if basis == 0:  # Eve measured in Z-basis
            if result == 1:
                qc.x(0)  # Prepare |1⟩
            # |0⟩ is default, no operation needed
        else:  # Eve measured in X-basis
            if result == 0:
                # Prepare |+⟩
                qc.unitary(quaternionic_rotation(np.pi / 2, 'y'), [0])
            else:
                # Prepare |-⟩
                qc.x(0)
                qc.unitary(quaternionic_rotation(np.pi / 2, 'y'), [0])
        
        circuits.append(qc)
    return circuits

# Remove mismatched bases
def remove_garbage(a_bases, b_bases, bits):
    return [bit for i, bit in enumerate(bits) if a_bases[i] == b_bases[i]]

# FIXED: Full circuit creation with quaternionic gates
def create_full_circuit_quaternion(alice_bits, alice_bases, bob_bases):
    n_qubits = len(alice_bits)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Alice prepares states
    for i in range(n_qubits):
        if alice_bases[i] == 0:  # Z-basis
            if alice_bits[i] == 1:
                qc.unitary(quaternionic_rotation(np.pi, 'x'), [i], label="X_Q")  # X gate
        else:  # X-basis - FIXED: Use correct encoding
            if alice_bits[i] == 0:
                # Create |+⟩
                qc.unitary(quaternionic_rotation(np.pi / 2, 'y'), [i], label="Ry_Q")
            else:
                # Create |-⟩
                qc.unitary(quaternionic_rotation(np.pi, 'x'), [i], label="X_Q")
                qc.unitary(quaternionic_rotation(np.pi / 2, 'y'), [i], label="Ry_Q")

    qc.barrier()

    # Bob's measurement - FIXED: Use Y-rotation for basis change
    for i in range(n_qubits):
        if bob_bases[i] == 1:
            qc.unitary(quaternionic_rotation(-np.pi / 2, 'y'), [i], label="Ry_Q")
    qc.measure_all()

    return qc

# Privacy amplification using universal hashing
def binary_entropy(q):
   
    if q == 0 or q == 1:
        return 0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)
    
def privacy_amplification(raw_key, qber, f=1.2, qber_threshold=0.11):
    if len(raw_key) == 0 or qber > qber_threshold:
        return "", 0

    s = len(raw_key)
    h = binary_entropy(qber)
    final_length = int(s * (1 - f * h))
    final_length = max(final_length, 0)
    
    if final_length == 0:
        return "", 0
    
    # Use a hash function to compress
    raw_key_str = ''.join(map(str, raw_key))
    hashed_key = hashlib.sha256(raw_key_str.encode()).hexdigest()
    hex_bitstring = bin(int(hashed_key, 16))[2:].zfill(256)  # SHA256 is 256 bits
    final_key = hex_bitstring[:final_length]
    return final_key, final_length
def error_correction(alice_key, bob_key, qber):
    """Simple error correction - in practice this would be more sophisticated"""
    if len(alice_key) != len(bob_key):
        return alice_key, bob_key
    
    # For simplicity, use Alice's key as reference (in practice, they'd use error correction codes)
    corrected_bob_key = alice_key.copy()  # Bob corrects his key to match Alice's
    
    return alice_key, corrected_bob_key
# FIXED: Helper function to get most frequent measurement result
def get_most_frequent_result(counts):
    """Get the most frequent measurement result from counts dictionary."""
    if not counts:
        return 0
    return int(max(counts.keys(), key=lambda x: counts[x]))

# BB84 protocol implementation with quaternionic gates
def bb84_quaternion_protocol(n_bits=24):
    # Alice's random bits and bases
    alice_bits = np.random.randint(2, size=n_bits)
    alice_bases = np.random.randint(2, size=n_bits)

    # Encoding with quaternionic operations
    message = encode_message_quaternion(alice_bits, alice_bases)

    # Eve's eavesdropping
    eve_bases = np.random.randint(2, size=n_bits)
    eve_circuits = measure_message_quaternion(message, eve_bases)

    # Simulate Eve's measurement results
    simulator = AerSimulator()
    transpiled_eve_circuits = transpile(eve_circuits, simulator)
    job_eve = simulator.run(transpiled_eve_circuits, shots=1024)
    result_eve = job_eve.result()
    
    # FIXED: Use more robust result extraction
    eve_results = []
    for i in range(len(eve_circuits)):
        counts = result_eve.get_counts(i)
        eve_results.append(get_most_frequent_result(counts))

    # FIXED: Eve re-prepares states based on her measurements
    eve_resent_circuits = eve_resend_states(eve_results, eve_bases)

    # Bob's random measurement bases
    bob_bases = np.random.randint(2, size=n_bits)
    # FIXED: Bob measures Eve's re-sent states
    bob_circuits = measure_message_quaternion(eve_resent_circuits, bob_bases)

    # Full quaternionic circuit (for reference)
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

    for i, count in enumerate(counts_list):
        for outcome, frequency in count.items():
            all_outcomes.append(f'Q{i}: {outcome}')
            all_frequencies.append(frequency)

    plt.figure(figsize=(20, 8))
    plt.bar(all_outcomes, all_frequencies)
    plt.xlabel('Qubit and Measurement Outcome', fontsize=12)
    plt.ylabel('Frequency (1024 shots)', fontsize=12)
    plt.title('Measurement Frequencies for All Qubits', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('measurement_frequencies_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIXED: Use robust result extraction for Bob's measurements
    bob_results = []
    for counts in counts_list:
        bob_results.append(get_most_frequent_result(counts))

    # Generate sifted keys
    alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
    bob_key = remove_garbage(alice_bases, bob_bases, bob_results)

    # Eve's Sifted Key
    eve_key_before_sifting = remove_garbage(alice_bases, eve_bases, eve_results)
    # Eve's key should be sifted based on Alice and Bob's matching bases
    eve_key_indices = [i for i in range(len(alice_bases)) if alice_bases[i] == bob_bases[i]]
    eve_key = [eve_results[i] for i in range(len(alice_bases)) if alice_bases[i] == bob_bases[i]]

    # Calculate QBER
    qber = calculate_qber(alice_key, bob_key)

    # Perform privacy amplification
    alice_final_key, alice_final_key_length = privacy_amplification(alice_key, qber)
    bob_final_key, bob_final_key_length = privacy_amplification(bob_key, qber)

    # Analyze Eve's success
    correct_eve_bits = sum(1 for a, e in zip(alice_key, eve_key) if a == e)
    eve_accuracy = correct_eve_bits / len(alice_key) if len(alice_key) > 0 else 0

    # Categorize basis selections
    right_basis_selections = sum(1 for a, b in zip(alice_bases, bob_bases) if a == b)
    wrong_basis_selections = n_bits - right_basis_selections

    # Prepare final output
    output = {
        'initial_bits': alice_bits.tolist(),
        'alice_bases': alice_bases.tolist(),
        'bob_bases': bob_bases.tolist(),
        'eve_bases': eve_bases.tolist(),  # Added for analysis
        'bob_results': bob_results,
        'eve_results': eve_results,  # Added for analysis
        'alice_key': alice_key,
        'bob_key': bob_key,
        'alice_final_key': alice_final_key,
        'alice_final_key_length': alice_final_key_length,
        'bob_final_key': bob_final_key,
        'bob_final_key_length': bob_final_key_length,
        'probabilities': probabilities,
        'measurement_plot': 'measurement_probabilities.png',
        'full_circuit': full_circuit,
        'qber': qber,
        'eve_accuracy': eve_accuracy,
        'eve_key': eve_key,
        'right_basis_selections': right_basis_selections,
        'wrong_basis_selections': wrong_basis_selections
    }
    return output

# Calculate QBER
def calculate_qber(alice_key, bob_key):
    if len(alice_key) == 0:
        return 0
    errors = sum(1 for a, b in zip(alice_key, bob_key) if a != b)
    return errors / len(alice_key)

def check_key_security(qber):
    if qber >= QBER_THRESHOLD:
        print("Key discarded due to high QBER.")
        # Code to restart or discard key
    else:
        print("Key accepted.")
        # Proceed to next step (privacy amplification, etc.)


# Analyze results of the protocol
def analyze_results_quaternion(results):
    print("Initial Bits:")
    print(f"Alice's bits: {results['initial_bits']}")
    print(f"Alice's bases: {results['alice_bases']}")
    print(f"Bob's bases: {results['bob_bases']}")
    print(f"Eve's bases: {results['eve_bases']}")  # Added
    print(f"Bob's results: {results['bob_results']}")
    print(f"Eve's results: {results['eve_results']}")  # Added
    
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
    print(f"Alice's final key: {results['alice_final_key']}")
    print(f"Alice's final key length: {results['alice_final_key_length']}")
    print(f"Bob's final key: {results['bob_final_key']}")
    print(f"Bob's final key length: {results['bob_final_key_length']}")
    print(f"\nMeasurement plot saved to: {results['measurement_plot']}")
    print(f"\nQBER: {results['qber']:.3f}")

    # Print Eve's Information
    print("\nEve's Information:")
    print(f"Eve's accuracy: {results['eve_accuracy']:.3f}")
    print(f"Eve's key: {results['eve_key']}")

    print("\nBasis Selections:")
    print(f"Right basis selections: {results['right_basis_selections']}")
    print(f"Wrong basis selections: {results['wrong_basis_selections']}")

if __name__ == "__main__":
    try:
        print("Executing Quaternionic BB84 protocol using Qiskit Aer Simulator...")
        results = bb84_quaternion_protocol(n_bits=24)
        analyze_results_quaternion(results)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
