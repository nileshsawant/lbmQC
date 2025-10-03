"""
Working quantum discrete Gaussian implementation with correct state construction
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def create_discrete_gaussian_circuit(probs):
    """
    Create quantum circuit for discrete Gaussian distribution over {-1, 0, 1}
    using a working state construction approach
    """
    # Normalize probabilities
    probs = probs / np.sum(probs)
    p_minus1, p_0, p_plus1 = probs
    
    # We want to create: |ψ⟩ = √p₋₁|00⟩ + √p₀|01⟩ + √p₊₁|10⟩
    
    # Use a multi-controlled approach with explicit amplitude construction
    qc = QuantumCircuit(2, 2)
    
    # Method: Use RY gates with specific angles to create the desired amplitudes
    
    # Step 1: First qubit determines if we're in {-1,0} vs {+1} subspace
    # P(first qubit = 0) = p₋₁ + p₀
    prob_first_0 = p_minus1 + p_0
    
    if prob_first_0 > 0 and prob_first_0 < 1:
        # RY rotation to set P(first qubit = 0) = prob_first_0
        theta1 = 2 * np.arccos(np.sqrt(prob_first_0))
        qc.ry(theta1, 0)
    elif prob_first_0 == 0:
        # Force first qubit to |1⟩
        qc.x(0)
    # If prob_first_0 == 1, qubit stays in |0⟩
    
    # Step 2: Second qubit determines choice within subspace
    if prob_first_0 > 1e-10:
        # When first qubit is 0: P(second = 1) = p₀ / (p₋₁ + p₀)
        prob_second_1_given_first_0 = p_0 / prob_first_0
        
        if prob_second_1_given_first_0 > 0 and prob_second_1_given_first_0 < 1:
            theta2 = 2 * np.arcsin(np.sqrt(prob_second_1_given_first_0))
            
            # Apply controlled rotation: rotate qubit 1 when qubit 0 is |0⟩
            qc.x(0)  # Flip to make control on |0⟩ into control on |1⟩
            qc.cry(theta2, 0, 1)
            qc.x(0)  # Flip back
        elif prob_second_1_given_first_0 == 1:
            # When first=0, always set second=1
            qc.x(0)
            qc.cx(0, 1)  # CNOT: if first=0 (now 1), flip second
            qc.x(0)
    
    # Measure
    qc.measure(0, 0)
    qc.measure(1, 1)
    
    return qc

def test_quantum_circuit():
    """Test the quantum circuit with known probabilities"""
    
    print("Testing Quantum Discrete Gaussian Circuit")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ([1/3, 1/3, 1/3], "Equal probabilities"),
        ([0.5, 0.3, 0.2], "Skewed towards -1"),
        ([0.1, 0.8, 0.1], "Peaked at 0"),
        ([0.2, 0.2, 0.6], "Skewed towards +1")
    ]
    
    for probs, description in test_cases:
        print(f"\\nTest: {description}")
        print(f"Target: P(-1)={probs[0]:.3f}, P(0)={probs[1]:.3f}, P(1)={probs[2]:.3f}")
        
        # Create and run circuit
        qc = create_discrete_gaussian_circuit(np.array(probs))
        
        # Simulate
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(1, simulator)
        qc_compiled = pass_manager.run(qc)
        
        job = simulator.run(qc_compiled, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        # Parse results with correct bitstring interpretation
        outcome_counts = {-1: 0, 0: 0, 1: 0}
        total = 0
        
        for bitstring, count in counts.items():
            if len(bitstring) >= 2:
                qubit1_bit = bitstring[0]  # Second qubit
                qubit0_bit = bitstring[1]  # First qubit
                
                if qubit0_bit == '0' and qubit1_bit == '0':
                    outcome_counts[-1] += count  # |00⟩ → -1
                elif qubit0_bit == '0' and qubit1_bit == '1':
                    outcome_counts[0] += count   # |01⟩ → 0
                elif qubit0_bit == '1' and qubit1_bit == '0':
                    outcome_counts[1] += count   # |10⟩ → +1
                
                total += count
        
        # Calculate empirical probabilities
        if total > 0:
            empirical = [outcome_counts[-1]/total, outcome_counts[0]/total, outcome_counts[1]/total]
        else:
            empirical = [0, 0, 0]
        
        print(f"Empirical: P(-1)={empirical[0]:.3f}, P(0)={empirical[1]:.3f}, P(1)={empirical[2]:.3f}")
        
        # Calculate error
        error = sum(abs(probs[i] - empirical[i]) for i in range(3))
        print(f"Total error: {error:.4f}")
        
        print(f"Raw counts: {counts}")

if __name__ == "__main__":
    test_quantum_circuit()