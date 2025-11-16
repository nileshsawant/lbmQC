"""
Quantum implementation for discrete Gaussian distribution on 1D grid
with sine wave variations in mean (u) and variance (T).

Grid: 10 points
Mean: u(x) = 0.1 * sin(2π * x/10)  
Variance: T(x) = T0 + 0.05 * sin(2π * x/10), where T0 = 1/3
Outcomes: {-1, 0, 1}
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from typing import Tuple, Dict, Optional

class QuantumDiscreteGaussian:
    def __init__(self, grid_size: int = 10, circuit_type: str = 'symmetric', 
                 grid_3d: Optional[Tuple[int, int, int]] = None):
        self.grid_size = grid_size
        self.T0 = 1/3  # Base variance
        self.outcomes = [-1, 0, 1]
        self.circuit_type = circuit_type  # Track which circuit implementation to use
        self.grid_3d = grid_3d  # 3D grid dimensions (Nx, Ny, Nz)
        
        # Validate circuit type
        if circuit_type not in ['symmetric', 'original']:
            raise ValueError(f"circuit_type must be 'symmetric' or 'original', got '{circuit_type}'")
        
    def compute_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and variance for each grid point"""
        x_points = np.arange(self.grid_size)
        
        # Mean: u(x) = 0.1 * sin(2π * x/10)
        means = 0.1 * np.sin(2 * np.pi * x_points / self.grid_size)
        
        # Variance: T(x) = T0 + 0.05 * sin(2π * x/10)
        variances = self.T0 + 0.05 * np.sin(2 * np.pi * x_points / self.grid_size)
        
        return means, variances
    
    def compute_parameters_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D grid parameters with sine wave variations.
        
        GRID STRUCTURE:
        3D grid with dimensions (Nx, Ny, Nz) specified in self.grid_3d
        Default: (10, 6, 4) for 240 total grid points
        
        PARAMETER VARIATIONS:
        - μₓ(x): 0.1 * sin(2π * x/Nx) - varies with x-position only
        - μᵧ(y): 0.1 * sin(2π * y/Ny) - varies with y-position only
        - μᵧ(z,x): 0.1 * sin(2π * z/Nz) + 0.02 * sin(2π * x/Nx) - varies with z (primary) and x (secondary)
        - T(x): T₀ + 0.05 * sin(2π * x/Nx) - temperature varies with x-position
        
        PHYSICAL INTERPRETATION:
        - Mean velocities show spatial variation (flow patterns)
        - Temperature varies with x (energy gradient)
        - Each dimension has independent sinusoidal variation
        
        RETURNS:
        Tuple of 4 arrays, each with shape (Nx, Ny, Nz):
        - means_x: x-component mean velocities
        - means_y: y-component mean velocities
        - means_z: z-component mean velocities
        - temperatures: temperature field (isotropic at each point)
        """
        if self.grid_3d is None:
            raise ValueError("3D grid dimensions not specified. Set grid_3d=(Nx, Ny, Nz) in __init__")
        
        Nx, Ny, Nz = self.grid_3d
        
        # Create coordinate arrays for each dimension
        x_coords = np.arange(Nx)
        y_coords = np.arange(Ny)
        z_coords = np.arange(Nz)
        
        # Create 3D meshgrid: X[i,j,k] gives x-coordinate, Y[i,j,k] gives y-coordinate, etc.
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Mean velocity variations (each depends on one coordinate only)
        # μₓ varies with x: flow in x-direction depends on x-position
        means_x = 0.1 * np.sin(2 * np.pi * X / Nx)
        
        # μᵧ varies with y: flow in y-direction depends on y-position
        means_y = 0.1 * np.sin(2 * np.pi * Y / Ny)
        
        # μᵧ varies with z: flow in z-direction depends on z-position
        # Also add small x-variation for better visualization in 2D slices
        means_z = 0.1 * np.sin(2 * np.pi * Z / Nz) + 0.02 * np.sin(2 * np.pi * X / Nx)
        
        # Temperature varies with x (energy gradient in x-direction)
        temperatures = self.T0 + 0.05 * np.sin(2 * np.pi * X / Nx)
        
        return means_x, means_y, means_z, temperatures
    
    def classical_discrete_gaussian_probs(self, mu: float, sigma_sq: float) -> np.ndarray:
        """
        Calculate classical discrete Gaussian probabilities for outcomes {-1, 0, 1}
        
        MATHEMATICAL FOUNDATION:
        Discrete Gaussian distribution: P(x) = exp(-(x-μ)²/(2σ²)) / Z
        - μ (mu): mean parameter (varies spatially as sine wave across grid)
        - σ² (sigma_sq): variance parameter (varies spatially as sine wave across grid)
        - Z: normalization constant (partition function) ensuring Σ P(x) = 1
        
        PHYSICAL INTERPRETATION:
        - When μ > 0: probability mass shifts toward +1 outcome
        - When μ < 0: probability mass shifts toward -1 outcome  
        - When μ = 0: symmetric distribution peaked at 0
        - Larger σ²: broader distribution (more uncertainty)
        - Smaller σ²: sharper distribution (more certainty)
        
        This classical calculation serves as the "ground truth" for our quantum implementation.
        """
        # STEP 1: Compute unnormalized probabilities using Gaussian kernel
        # For each outcome x ∈ {-1, 0, 1}, calculate exp(-(x-μ)²/(2σ²))
        # This measures the "fitness" or "likelihood" of each outcome given parameters μ, σ²
        unnorm_probs = np.array([
            np.exp(-(k - mu)**2 / (2 * sigma_sq)) for k in self.outcomes
        ])
        
        # STEP 2: Normalize to satisfy probability axiom: Σ P(x) = 1
        # Z is the partition function (total unnormalized probability mass)
        Z = np.sum(unnorm_probs)
        normalized_probs = unnorm_probs / Z
        
        # RESULT: [P(-1), P(0), P(1)] - classical probability distribution
        return normalized_probs
    
    def standardLattice_discrete_gaussian_probs(self, mu: float, sigma_sq: float) -> np.ndarray:
        """
        Calculate discrete Gaussian probabilities for outcomes {-1, 0, 1} using 
        Maxwell-Boltzmann formulation.
        
        MATHEMATICAL FOUNDATION:
        Reference: Section 2.2.4 of Sawant (2024)
        https://doi.org/10.3929/ethz-b-000607045
        
        Using the product form of Maxwell-Boltzmann distribution:
        p = u*u + T  (where T is temperature/variance)
        
        Probabilities for each discrete velocity:
        P(-1) = -0.5 * u + 0.5 * p  (backward motion)
        P(0)  = -p + 1.0            (at rest)
        P(+1) = +0.5 * u + 0.5 * p  (forward motion)
        
        These probabilities ensure:
        - Sum to 1 (normalization)
        - Mean = u (by construction)
        - Variance = T (temperature)
        
        PARAMETERS:
        - mu: mean velocity (u in the paper)
        - sigma_sq: temperature/variance (T in the paper)
        
        RETURNS:
        Array [P(-1), P(0), P(+1)] - normalized probability distribution
        
        This classical calculation serves as the "ground truth" for our quantum implementation.
        """
        p = mu * mu + sigma_sq
        
        normalized_probs = np.array([
            -0.5 * mu + 0.5 * p,  # P(-1)
            -p + 1.0,              # P(0)
            +0.5 * mu + 0.5 * p   # P(+1)
        ])
        
        # RESULT: [P(-1), P(0), P(1)] - classical probability distribution
        return normalized_probs
    
    #wrapper to select discrete gaussian probability calculation method
    def discrete_gaussian_probs(self, mu: float, sigma_sq: float) -> np.ndarray:
        return self.standardLattice_discrete_gaussian_probs(mu, sigma_sq)

    
    def create_quantum_circuit_symmetric(self, probs: np.ndarray) -> QuantumCircuit:
        """
        Symmetric quantum circuit with improved decomposition: {-1, +1} vs {0}
        
        QUANTUM COMPUTING FOUNDATION:
        This implements a symmetric hierarchical decomposition that better matches
        the symmetry of the velocity encoding formulas.
        
        MATHEMATICAL MAPPING:
        Classical: P(-1), P(0), P(+1) ∈ [0,1] with P(-1) + P(0) + P(+1) = 1
        Quantum:   |ψ⟩ = √P(-1)|00⟩ + √P(+1)|01⟩ + √P(0)|10⟩
        
        SYMMETRIC QUBIT ENCODING SCHEME:
        - |00⟩ (both qubits in state 0) → outcome -1
        - |01⟩ (first qubit 0, second qubit 1) → outcome +1
        - |10⟩ (first qubit 1, second qubit 0) → outcome 0
        - |11⟩ (both qubits in state 1) → unused
        
        ADVANTAGES OF THIS DECOMPOSITION:
        - First qubit splits "moving" {-1, +1} vs "stationary" {0}
        - Simpler angle formulas: θ₁ = 2*arccos(√(mu² + sigma_sq))
        - Better captures physical symmetry of velocity distribution
        - More numerically stable for velocity encoding
        """
        # STEP 1: Ensure probability normalization
        probs = probs / np.sum(probs)
        p_minus1, p_0, p_plus1 = probs  # Extract individual probabilities
        
        # STEP 2: Initialize quantum circuit with 2 qubits + 2 classical bits
        qc = QuantumCircuit(2, 2)
        
        # STEP 3: HIERARCHICAL DECOMPOSITION - First Qubit (Coarse Splitting)
        # 
        # QUANTUM STRATEGY: Split "moving particles" vs "stationary particles"
        # Split the 3-outcome problem into: {-1, +1} vs {0}
        # 
        # MATHEMATICAL FOUNDATION (from Section 2.2.4 of https://doi.org/10.3929/ethz-b-000607045):
        # First qubit encodes: P(outcome ∈ {-1, +1}) vs P(outcome = 0)
        # P(first qubit = 0) = P(-1) + P(+1) = p = μ² + T
        # P(first qubit = 1) = P(0) = -p + 1.0
        #
        # QUANTUM GATE FORMULATION:
        # RY gate (rotation around Y-axis): RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        # We want: |cos(θ/2)|² = P(first=0) and |sin(θ/2)|² = P(first=1)
        # Solving: cos²(θ/2) = p → θ₁ = 2*arccos(√p) = 2*arccos(√(μ² + T))
        # 
        # This angle depends only on the combined parameter p, making it very stable!
        
        prob_first_0 = p_minus1 + p_plus1  # Combined probability for {-1, +1} outcomes
        
        if prob_first_0 > 0 and prob_first_0 < 1:
            # GENERAL CASE: Create superposition between motion and rest
            theta1 = 2 * np.arccos(np.sqrt(prob_first_0))  # Calculate rotation angle
            qc.ry(theta1, 0)  # Apply Y-rotation to first qubit
            
        elif prob_first_0 == 0:
            # EDGE CASE: Only outcome 0 possible (P(-1)=P(+1)=0, P(0)=1)
            qc.x(0)  # X gate: |0⟩ → |1⟩ (deterministic flip to rest state)
            
        # EDGE CASE: If prob_first_0 == 1, first qubit stays |0⟩ (no gates needed)
        # This happens when P(0) = 0, so only {-1, +1} outcomes are possible
        
        # STEP 4: CONDITIONAL DECOMPOSITION - Second Qubit (Fine Splitting)
        #
        # QUANTUM STRATEGY: Conditional probability encoding for directional motion
        # Given first qubit = 0 (we're in {-1, +1} subspace), distinguish -1 from +1
        #
        # MATHEMATICAL FOUNDATION (using Bayes' rule):  
        # P(outcome=+1 | first=0) = P(+1) / (P(-1) + P(+1))
        #                         = (0.5*μ + 0.5*p) / p
        #                         = 0.5*(1 + μ/p)
        #
        # This conditional probability is symmetric around 0.5:
        # - When μ > 0: favors +1 (forward motion)
        # - When μ < 0: favors -1 (backward motion)
        # - When μ = 0: equal probability (no net flow)
        #
        # QUANTUM IMPLEMENTATION:
        # Use controlled-RY gate: rotates second qubit only when first qubit = 0
        # Angle: θ₂ = 2*arcsin(√(P(+1|first=0))) = 2*arcsin(√(0.5*(1 + μ/p)))
        #
        # The control ensures this rotation applies only in the "moving" subspace!
        
        if prob_first_0 > 1e-10:  # Only proceed if {-1,+1} outcomes are possible
            
            # Calculate conditional probability using Bayes' rule
            prob_second_1_given_first_0 = p_plus1 / prob_first_0  # P(outcome=+1 | outcome∈{-1,+1})
            
            if prob_second_1_given_first_0 > 0 and prob_second_1_given_first_0 < 1:
                # GENERAL CASE: Both -1 and +1 outcomes possible within {-1,+1} subspace
                
                # Calculate rotation angle for conditional probability
                theta2 = 2 * np.arcsin(np.sqrt(prob_second_1_given_first_0))
                
                # QUANTUM TRICK: Convert "control on |0⟩" to "control on |1⟩"
                # Most quantum gates control on |1⟩ state, but we need control on |0⟩
                qc.x(0)              # X gate: |0⟩ ↔ |1⟩ (flip first qubit state)
                qc.cry(theta2, 0, 1) # Controlled-RY: rotate qubit 1 when qubit 0 is |1⟩ (originally |0⟩)
                qc.x(0)              # X gate: flip first qubit back to original state
                
            elif prob_second_1_given_first_0 == 1:
                # EDGE CASE: Only outcome +1 possible when first qubit is |0⟩ (P(-1)=0, P(+1)>0)
                qc.x(0)          # Flip first qubit: |0⟩ → |1⟩  
                qc.cx(0, 1)      # CNOT gate: if control=|1⟩ (originally |0⟩), flip target qubit
                qc.x(0)          # Restore first qubit to original state
                
            # EDGE CASE: If prob_second_1_given_first_0 == 0, second qubit stays |0⟩
            # This happens when P(+1)=0 but P(-1)>0, so only -1 outcome possible in {-1,+1} subspace
        
        # STEP 5: QUANTUM MEASUREMENT
        #
        # NEW MEASUREMENT SCHEME:
        # Measure both qubits simultaneously to get 2-bit classical string
        # |00⟩ → classical bits "00" → decode to outcome -1  
        # |01⟩ → classical bits "01" → decode to outcome +1  [CHANGED]
        # |10⟩ → classical bits "10" → decode to outcome 0   [CHANGED]
        # |11⟩ → classical bits "11" → unused (should have 0 probability)
        #
        # QISKIT CONVENTION:
        # qc.measure(qubit_index, classical_bit_index) 
        # Stores measurement result of quantum qubit in classical bit register
        
        qc.measure(0, 0)  # Measure first qubit → store result in classical bit 0
        qc.measure(1, 1)  # Measure second qubit → store result in classical bit 1
        
        # CIRCUIT COMPLETE: Returns quantum circuit ready for execution
        # This symmetric decomposition produces identical probability distributions
        # but with simpler mathematical structure for velocity encoding
        return qc

    
    def create_quantum_circuit_original(self, probs: np.ndarray) -> QuantumCircuit:
        """
        Create quantum circuit to generate discrete Gaussian distribution using amplitude encoding
        
        QUANTUM COMPUTING FOUNDATION:
        This function implements "amplitude encoding" - a fundamental quantum technique where
        classical probabilities are encoded as quantum amplitudes in a superposition state.
        
        MATHEMATICAL MAPPING:
        Classical: P(-1), P(0), P(1) ∈ [0,1] with P(-1) + P(0) + P(1) = 1
        Quantum:   |ψ = √P(-1)|00 + √P(0)|01 + √P(1)|10
        
        WHY SQUARE ROOTS?
        Born's rule in quantum mechanics: |amplitude|² = probability
        So if amplitude = √P, then |√P|² = P (correct probability)
        
        QUBIT ENCODING SCHEME:
        - |00 (both qubits in state 0) → outcome -1
        - |01 (first qubit 0, second qubit 1) → outcome 0  
        - |10 (first qubit 1, second qubit 0) → outcome +1
        - |11 (both qubits in state 1) → unused (helps with circuit construction)
        
        QUANTUM ADVANTAGE:
        Once encoded, quantum circuits can manipulate all outcomes simultaneously
        through superposition, potentially offering speedup over classical sampling.
        """
        # STEP 1: Ensure probability normalization (quantum states must have unit norm)
        probs = probs / np.sum(probs)
        p_minus1, p_0, p_plus1 = probs  # Extract individual probabilities
        
        # STEP 2: Initialize quantum circuit with 2 qubits + 2 classical bits for measurement
        # Qubit 0: "first" qubit in our encoding scheme
        # Qubit 1: "second" qubit in our encoding scheme  
        # Classical bits: store measurement results
        qc = QuantumCircuit(2, 2)
        
        # STEP 3: HIERARCHICAL DECOMPOSITION - First Qubit (Coarse Splitting)
        # 
        # QUANTUM STRATEGY: Use hierarchical probability tree decomposition
        # Split the 3-outcome problem into: {-1,0} vs {+1}
        # 
        # MATHEMATICAL FOUNDATION:
        # First qubit encodes: P(outcome ∈ {-1, 0}) vs P(outcome = +1)
        # P(first qubit = 0) = P(-1) + P(0) = combined probability of negative/zero outcomes
        # P(first qubit = 1) = P(+1) = probability of positive outcome
        #
        # QUANTUM GATE SELECTION:
        # RY gate (rotation around Y-axis) creates superposition: RY(θ)|0 = cos(θ/2)|0 + sin(θ/2)|1
        # We want: |cos(θ/2)|² = P(first=0) and |sin(θ/2)|² = P(first=1)
        # Solving: cos²(θ/2) = prob_first_0 → θ = 2*arccos(√prob_first_0)
        
        prob_first_0 = p_minus1 + p_0  # Combined probability for {-1, 0} outcomes
        
        if prob_first_0 > 0 and prob_first_0 < 1:
            # GENERAL CASE: Create superposition between |0 and |1 states
            theta1 = 2 * np.arccos(np.sqrt(prob_first_0))  # Calculate rotation angle
            qc.ry(theta1, 0)  # Apply Y-rotation to first qubit
            
        elif prob_first_0 == 0:
            # EDGE CASE: Only +1 outcome possible (P(-1)=P(0)=0, P(+1)=1)
            qc.x(0)  # X gate: |0 → |1 (deterministic flip to |1 state)
            
        # EDGE CASE: If prob_first_0 == 1, first qubit stays |0 (no gates needed)
        # This happens when P(+1) = 0, so only {-1, 0} outcomes are possible
        
        # STEP 4: CONDITIONAL DECOMPOSITION - Second Qubit (Fine Splitting)
        #
        # QUANTUM STRATEGY: Conditional probability encoding using controlled gates
        # Given first qubit = 0 (we're in {-1, 0} subspace), distinguish between -1 and 0
        #
        # MATHEMATICAL FOUNDATION:  
        # P(second=1 | first=0) = P(outcome=0) / P(outcome∈{-1,0}) = P(0) / (P(-1) + P(0))
        # This uses conditional probability: P(A|B) = P(A∩B) / P(B)
        # Here: A="second qubit is 1", B="first qubit is 0"
        #
        # QUANTUM IMPLEMENTATION:
        # Use controlled-RY gate: only rotates second qubit when first qubit is in specific state
        # CRY gate: applies RY rotation to target qubit conditioned on control qubit state
        
        if prob_first_0 > 1e-10:  # Only proceed if {-1,0} outcomes are possible
            
            # Calculate conditional probability using Bayes' rule
            prob_second_1_given_first_0 = p_0 / prob_first_0  # P(outcome=0 | outcome∈{-1,0})
            
            if prob_second_1_given_first_0 > 0 and prob_second_1_given_first_0 < 1:
                # GENERAL CASE: Both -1 and 0 outcomes possible within {-1,0} subspace
                
                # Calculate rotation angle for conditional probability
                theta2 = 2 * np.arcsin(np.sqrt(prob_second_1_given_first_0))
                
                # QUANTUM TRICK: Convert "control on |0" to "control on |1"
                # Most quantum gates control on |1 state, but we need control on |0
                qc.x(0)              # X gate: |0 ↔ |1 (flip first qubit state)
                qc.cry(theta2, 0, 1) # Controlled-RY: rotate qubit 1 when qubit 0 is |1 (originally |0)
                qc.x(0)              # X gate: flip first qubit back to original state
                
            elif prob_second_1_given_first_0 == 1:
                # EDGE CASE: Only outcome 0 possible when first qubit is |0 (P(-1)=0, P(0)>0)
                qc.x(0)          # Flip first qubit: |0 → |1  
                qc.cx(0, 1)      # CNOT gate: if control=|1 (originally |0), flip target qubit
                qc.x(0)          # Restore first qubit to original state
                
            # EDGE CASE: If prob_second_1_given_first_0 == 0, second qubit stays |0
            # This happens when P(0)=0 but P(-1)>0, so only -1 outcome possible in {-1,0} subspace
        
        # STEP 5: QUANTUM MEASUREMENT
        #
        # MEASUREMENT FOUNDATION:
        # Quantum measurement collapses superposition |ψ = Σᵢ αᵢ|i into classical outcome
        # Probability of measuring state |i = |αᵢ|² (Born's rule)
        # After measurement, quantum state is destroyed and becomes classical
        #
        # OUR MEASUREMENT SCHEME:
        # Measure both qubits simultaneously to get 2-bit classical string
        # |00 → classical bits "00" → decode to outcome -1  
        # |01 → classical bits "01" → decode to outcome 0
        # |10 → classical bits "10" → decode to outcome +1
        # |11 → classical bits "11" → unused (should have 0 probability by construction)
        #
        # QISKIT CONVENTION:
        # qc.measure(qubit_index, classical_bit_index) 
        # Stores measurement result of quantum qubit in classical bit register
        
        qc.measure(0, 0)  # Measure first qubit → store result in classical bit 0
        qc.measure(1, 1)  # Measure second qubit → store result in classical bit 1
        
        # CIRCUIT COMPLETE: Returns quantum circuit ready for execution
        # When run multiple times (shots), produces random samples from discrete Gaussian
        return qc
    
    def create_quantum_circuit_parametric(self, mu: float, sigma_sq: float) -> QuantumCircuit:
        """
        Create symmetric quantum circuit directly from (mu, sigma_sq) parameters.
        
        DIRECT PARAMETRIZATION:
        This method computes gate angles directly from Maxwell-Boltzmann parameters
        without the intermediate step of computing classical probabilities.
        
        MATHEMATICAL FOUNDATION:
        For velocity encoding with p = mu² + sigma_sq:
        - P(-1) = -0.5*mu + 0.5*p
        - P(0)  = -p + 1
        - P(+1) = +0.5*mu + 0.5*p
        
        SYMMETRIC DECOMPOSITION ANGLES:
        - θ₁ = 2*arccos(√p) where p = mu² + sigma_sq
          This angle splits {-1,+1} (motion) vs {0} (rest)
          
        - θ₂ = 2*arcsin(√[0.5*(1 + mu/p)])
          This angle splits -1 vs +1 given motion
        
        ADVANTAGES:
        - No classical probability computation needed
        - Fewer floating-point operations
        - Direct mathematical relationship to physical parameters
        - Only works with symmetric circuit (simpler formulas)
        
        PARAMETERS:
        - mu: mean velocity (Maxwell-Boltzmann parameter)
        - sigma_sq: temperature/variance (Maxwell-Boltzmann parameter)
        
        CONSTRAINT:
        - Requires sigma_sq < 1 for valid probabilities (T < 1 constraint)
        """
        # STEP 1: Compute the key parameter p = mu² + sigma_sq
        # This represents the combined probability of motion: P(-1) + P(+1)
        p = mu * mu + sigma_sq
        
        # STEP 2: Validate physical constraints
        # For valid probabilities, we need 0 < p < 1
        # This ensures all probabilities are positive and sum to 1
        if p <= 0:
            raise ValueError(f"Invalid parameters: p = mu² + sigma_sq = {p:.4f} must be positive")
        if p >= 1:
            raise ValueError(f"Invalid parameters: p = mu² + sigma_sq = {p:.4f} must be < 1 "
                           f"(Maxwell-Boltzmann constraint T < 1 violated)")
        
        # STEP 3: Initialize quantum circuit with 2 qubits + 2 classical bits
        qc = QuantumCircuit(2, 2)
        
        # STEP 4: FIRST ROTATION - Split motion {-1,+1} vs rest {0}
        #
        # DIRECT ANGLE FORMULA:
        # θ₁ = 2*arccos(√p) where p = mu² + sigma_sq
        #
        # PHYSICAL INTERPRETATION:
        # This angle encodes the probability of particle motion vs rest
        # - cos²(θ₁/2) = p (probability of motion)
        # - sin²(θ₁/2) = 1-p (probability of rest)
        #
        # SIMPLIFICATION from velocity encoding:
        # Original: P(-1) + P(+1) = (-0.5*mu + 0.5*p) + (+0.5*mu + 0.5*p) = p
        # Direct: θ₁ = 2*arccos(√p) - no probability computation needed!
        
        theta1 = 2 * np.arccos(np.sqrt(p))
        qc.ry(theta1, 0)  # Apply first rotation to qubit 0
        
        # STEP 5: SECOND ROTATION (CONDITIONAL) - Split -1 vs +1 given motion
        #
        # DIRECT ANGLE FORMULA:
        # θ₂ = 2*arcsin(√[0.5*(1 + mu/p)])
        #
        # PHYSICAL INTERPRETATION:
        # Given the particle is moving (first qubit = 0), this angle determines
        # whether it moves left (-1) or right (+1)
        # - cos²(θ₂/2) = P(-1|motion) = [-0.5*mu + 0.5*p]/p = 0.5*(1 - mu/p)
        # - sin²(θ₂/2) = P(+1|motion) = [+0.5*mu + 0.5*p]/p = 0.5*(1 + mu/p)
        #
        # SIMPLIFICATION from velocity encoding:
        # Conditional probability: P(+1|motion) = P(+1)/[P(-1)+P(+1)]
        #                                        = [+0.5*mu + 0.5*p]/p
        #                                        = 0.5*(1 + mu/p)
        # Direct: θ₂ = 2*arcsin(√[0.5*(1 + mu/p)])
        
        # Calculate conditional probability directly
        prob_plus1_given_motion = 0.5 * (1.0 + mu / p)
        
        # Validate conditional probability is in valid range
        if prob_plus1_given_motion < 0 or prob_plus1_given_motion > 1:
            raise ValueError(f"Invalid conditional probability: {prob_plus1_given_motion:.4f}")
        
        # Calculate second angle
        theta2 = 2 * np.arcsin(np.sqrt(prob_plus1_given_motion))
        
        # Apply controlled rotation (only when first qubit is |0⟩)
        # Use X-gates to convert "control on |0⟩" to "control on |1⟩"
        qc.x(0)              # Flip first qubit
        qc.cry(theta2, 0, 1) # Controlled-RY: rotate qubit 1 when qubit 0 is |1⟩
        qc.x(0)              # Flip first qubit back
        
        # STEP 6: MEASUREMENT
        # Symmetric encoding: |00⟩→-1, |01⟩→+1, |10⟩→0
        qc.measure(0, 0)  # Measure first qubit
        qc.measure(1, 1)  # Measure second qubit
        
        # CIRCUIT COMPLETE: Direct parametrization without classical probability computation!
        return qc
    
    def create_quantum_circuit_3d_parametric(self, mu_x: float, mu_y: float, mu_z: float, T: float) -> QuantumCircuit:
        """
        Create 6-qubit circuit for 3D velocity sampling with parallel execution.
        
        HARDWARE PARALLELIZATION:
        All three dimensions execute simultaneously on independent qubit pairs.
        Circuit depth remains 4 layers (same as 1D), achieving 3× speedup over
        sequential execution of three 1D circuits.
        
        QUBIT ALLOCATION:
        - Qubits 0-1: vₓ component (x-direction velocity)
        - Qubits 2-3: vᵧ component (y-direction velocity)
        - Qubits 4-5: vᵧ component (z-direction velocity)
        
        PHYSICAL INTERPRETATION:
        All components share same temperature T (isotropic kinetic energy) but have
        different mean velocities (directional flow). This represents a 3D Maxwell-
        Boltzmann distribution for lattice velocities in 3D Lattice Boltzmann Method.
        
        ENCODING SCHEME (per dimension):
        |00⟩ → -1 (negative velocity)
        |01⟩ → +1 (positive velocity)
        |10⟩ →  0 (zero velocity)
        |11⟩ → unused
        
        3D MEASUREMENT:
        Single measurement yields 6-bit string → (vₓ, vᵧ, vᵧ) tuple
        Example: |001001⟩ → vₓ=+1, vᵧ=-1, vᵧ=+1
        
        PARAMETERS:
        - mu_x, mu_y, mu_z: mean velocities in each dimension
        - T: temperature/variance (shared across all dimensions)
        
        CONSTRAINT:
        - Requires T < 1 and muᵢ² + T < 1 for all i ∈ {x,y,z}
        
        PARALLELIZATION BENEFIT:
        Time complexity: T_circuit (not 3×T_circuit for sequential)
        """
        # STEP 1: Initialize 6-qubit circuit
        qc = QuantumCircuit(6, 6)
        
        # STEP 2: X-COMPONENT (qubits 0-1) - Parallel execution
        p_x = mu_x * mu_x + T
        
        # Validate physical constraints
        if p_x <= 0:
            raise ValueError(f"Invalid parameters for X: p_x = μ_x² + T = {p_x:.4f} must be positive")
        if p_x >= 1:
            raise ValueError(f"Invalid parameters for X: p_x = μ_x² + T = {p_x:.4f} must be < 1")
        
        # Calculate angles directly from parameters
        theta1_x = 2 * np.arccos(np.sqrt(p_x))
        prob_plus1_x = 0.5 * (1.0 + mu_x / p_x)
        
        if prob_plus1_x < 0 or prob_plus1_x > 1:
            raise ValueError(f"Invalid conditional probability for X: {prob_plus1_x:.4f}")
        
        theta2_x = 2 * np.arcsin(np.sqrt(prob_plus1_x))
        
        # Apply gates to qubits 0-1
        qc.ry(theta1_x, 0)
        qc.x(0)
        qc.cry(theta2_x, 0, 1)
        qc.x(0)
        
        # STEP 3: Y-COMPONENT (qubits 2-3) - Parallel execution
        p_y = mu_y * mu_y + T
        
        if p_y <= 0:
            raise ValueError(f"Invalid parameters for Y: p_y = μ_y² + T = {p_y:.4f} must be positive")
        if p_y >= 1:
            raise ValueError(f"Invalid parameters for Y: p_y = μ_y² + T = {p_y:.4f} must be < 1")
        
        theta1_y = 2 * np.arccos(np.sqrt(p_y))
        prob_plus1_y = 0.5 * (1.0 + mu_y / p_y)
        
        if prob_plus1_y < 0 or prob_plus1_y > 1:
            raise ValueError(f"Invalid conditional probability for Y: {prob_plus1_y:.4f}")
        
        theta2_y = 2 * np.arcsin(np.sqrt(prob_plus1_y))
        
        # Apply gates to qubits 2-3
        qc.ry(theta1_y, 2)
        qc.x(2)
        qc.cry(theta2_y, 2, 3)
        qc.x(2)
        
        # STEP 4: Z-COMPONENT (qubits 4-5) - Parallel execution
        p_z = mu_z * mu_z + T
        
        if p_z <= 0:
            raise ValueError(f"Invalid parameters for Z: p_z = μ_z² + T = {p_z:.4f} must be positive")
        if p_z >= 1:
            raise ValueError(f"Invalid parameters for Z: p_z = μ_z² + T = {p_z:.4f} must be < 1")
        
        theta1_z = 2 * np.arccos(np.sqrt(p_z))
        prob_plus1_z = 0.5 * (1.0 + mu_z / p_z)
        
        if prob_plus1_z < 0 or prob_plus1_z > 1:
            raise ValueError(f"Invalid conditional probability for Z: {prob_plus1_z:.4f}")
        
        theta2_z = 2 * np.arcsin(np.sqrt(prob_plus1_z))
        
        # Apply gates to qubits 4-5
        qc.ry(theta1_z, 4)
        qc.x(4)
        qc.cry(theta2_z, 4, 5)
        qc.x(4)
        
        # STEP 5: MEASURE ALL QUBITS
        # Single measurement captures all 3 velocity components simultaneously
        qc.measure(range(6), range(6))
        
        # CIRCUIT COMPLETE: 3D velocity sampling with hardware parallelization!
        return qc
    
    def create_quantum_circuit(self, probs: np.ndarray) -> QuantumCircuit:
        """
        Create quantum circuit using the configured circuit type.
        
        This method automatically routes to the correct implementation based on
        self.circuit_type (set in __init__):
        - 'symmetric': {-1, +1} vs {0} decomposition (default, recommended)
        - 'original': {-1, 0} vs {+1} decomposition
        
        The decoder will automatically match the circuit type, so users don't need
        to manually track which decoder to use.
        
        The symmetric decomposition is preferred because it:
        - Has simpler angle formulas
        - Better captures physical symmetry
        - Is more numerically stable for velocity encoding
        """
        if self.circuit_type == 'symmetric':
            return self.create_quantum_circuit_symmetric(probs)
        else:  # 'original'
            return self.create_quantum_circuit_original(probs)
    
    def quantum_sample_grid_point(self, mu: float, sigma_sq: float, shots: int = 1000) -> Dict[int, int]:
        """
        Execute quantum sampling for discrete Gaussian distribution at single grid point
        
        QUANTUM SAMPLING PROCESS:
        1. Classical preprocessing: Calculate target probabilities P(-1), P(0), P(1)
        2. Quantum encoding: Convert probabilities to quantum circuit amplitudes  
        3. Quantum execution: Run circuit multiple times (shots) to get samples
        4. Classical postprocessing: Decode quantum measurement results to outcomes
        
        PARAMETERS:
        - mu: mean parameter for this specific grid location
        - sigma_sq: variance parameter for this specific grid location  
        - shots: number of quantum circuit executions (sample size)
        
        QUANTUM ADVANTAGE:
        Each quantum circuit execution processes the full superposition simultaneously,
        potentially offering advantages over classical rejection sampling methods.
        """
        
        # STEP 1: CLASSICAL PREPROCESSING
        # Calculate the target discrete Gaussian probabilities for this grid point
        # This gives us the classical "ground truth" that our quantum circuit should reproduce
        probs = self.discrete_gaussian_probs(mu, sigma_sq)
        
        # STEP 2: QUANTUM CIRCUIT CREATION  
        # Convert classical probabilities into quantum circuit using amplitude encoding
        # This is the core quantum advantage: encoding probabilistic information as quantum superposition
        qc = self.create_quantum_circuit(probs)
        
        # STEP 3: QUANTUM CIRCUIT COMPILATION
        # Prepare circuit for execution on quantum simulator
        # - AerSimulator: High-performance classical simulator of quantum circuits
        # - Pass manager: Optimizes circuit for better execution (gate optimization, noise modeling, etc.)
        # - Optimization level 1: Basic optimizations (gate cancellation, routing, etc.)
        simulator = AerSimulator()  # Create quantum circuit simulator
        pass_manager = generate_preset_pass_manager(1, simulator)  # Create optimization pipeline  
        qc_compiled = pass_manager.run(qc)  # Apply optimizations to circuit
        
        # STEP 4: QUANTUM EXECUTION
        # Run the quantum circuit multiple times to collect statistical samples
        # Each "shot" represents one complete quantum computation: prepare state → measure → get classical result
        job = simulator.run(qc_compiled, shots=shots)  # Submit job to quantum simulator
        result = job.result()  # Wait for completion and get results
        counts = result.get_counts()  # Extract measurement statistics: {'bitstring': count, ...}
        return self._decode_quantum_counts(counts)

    def quantum_sample_grid_point_parametric(self, mu: float, sigma_sq: float, shots: int = 1000) -> Dict[int, int]:
        """
        Execute quantum sampling using direct parametrization (symmetric circuit only).
        
        DIRECT PARAMETRIC SAMPLING:
        This method bypasses classical probability computation entirely by computing
        gate angles directly from (mu, sigma_sq) parameters.
        
        WORKFLOW:
        1. Direct angle computation: θ₁, θ₂ = f(mu, sigma_sq) - no probability step!
        2. Quantum circuit creation: Build circuit with computed angles
        3. Quantum execution: Run circuit multiple times (shots)
        4. Classical postprocessing: Decode measurement results
        
        ADVANTAGES over quantum_sample_grid_point:
        - Fewer classical operations (no probability computation)
        - Direct mathematical relationship to physical parameters
        - More efficient for large-scale sampling
        - Only available for symmetric circuit (simpler angle formulas)
        
        PARAMETERS:
        - mu: mean velocity (Maxwell-Boltzmann parameter)
        - sigma_sq: temperature/variance (Maxwell-Boltzmann parameter)
        - shots: number of quantum circuit executions (sample size)
        
        CONSTRAINT:
        - Only works with symmetric circuit (circuit_type='symmetric')
        - Requires sigma_sq < 1 for valid probabilities
        """
        # STEP 1: Validate circuit type
        if self.circuit_type != 'symmetric':
            raise ValueError(f"Parametric sampling only available for symmetric circuit, "
                           f"but circuit_type='{self.circuit_type}'")
        
        # STEP 2: DIRECT QUANTUM CIRCUIT CREATION
        # Compute gate angles directly from parameters without classical probability step
        qc = self.create_quantum_circuit_parametric(mu, sigma_sq)
        
        # STEP 3: QUANTUM CIRCUIT COMPILATION
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(1, simulator)
        qc_compiled = pass_manager.run(qc)
        
        # STEP 4: QUANTUM EXECUTION
        job = simulator.run(qc_compiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # STEP 5: DECODE RESULTS
        # Use symmetric decoder (automatically selected via self.circuit_type)
        return self._decode_quantum_counts(counts)

    def quantum_sample_grid_point_3d_parametric(
        self, 
        mu_x: float, 
        mu_y: float, 
        mu_z: float, 
        T: float, 
        shots: int = 1000
    ) -> Dict[Tuple[int, int, int], int]:
        """
        Execute quantum sampling for 3D velocity distribution at a single grid point.
        
        WORKFLOW:
        1. Direct circuit creation: Build 6-qubit circuit from (μₓ, μᵧ, μᵧ, T)
        2. Quantum execution: Run circuit multiple times (shots) on simulator
        3. Measurement: Single measurement yields full 3D velocity tuple
        4. Decoding: Convert 6-bit strings to (vₓ, vᵧ, vᵧ) tuples
        
        PARALLELIZATION ADVANTAGE:
        All three dimensions are sampled simultaneously in a single circuit execution.
        Time = T_circuit (not 3×T_circuit for sequential 1D sampling).
        
        PARAMETERS:
        - mu_x, mu_y, mu_z: mean velocities in each dimension
        - T: temperature/variance (shared isotropic property)
        - shots: number of quantum circuit executions (sample size)
        
        OUTPUT:
        Dictionary mapping (vₓ, vᵧ, vᵧ) tuples to counts
        Example: {(-1, 0, +1): 235, (0, 0, 0): 189, ...}
        
        CONSTRAINT:
        - Requires T < 1 and μᵢ² + T < 1 for all i ∈ {x,y,z}
        """
        # STEP 1: Create 3D quantum circuit directly from parameters
        qc = self.create_quantum_circuit_3d_parametric(mu_x, mu_y, mu_z, T)
        
        # STEP 2: Compile circuit for quantum simulator
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(1, simulator)
        qc_compiled = pass_manager.run(qc)
        
        # STEP 3: Execute quantum circuit
        job = simulator.run(qc_compiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # STEP 4: Decode 6-qubit measurements to 3D velocity tuples
        return self._decode_quantum_counts_3d(counts)

    def compute_moments_from_samples_3d(
        self, 
        velocity_counts: Dict[Tuple[int, int, int], int]
    ) -> Dict[str, float]:
        """
        Compute statistical moments from 3D velocity samples.
        
        COMPUTED MOMENTS:
        For each dimension (x, y, z):
        - mean_x, mean_y, mean_z: E[vᵢ] = Σ vᵢ * P(vᵢ)
        - var_x, var_y, var_z: Var[vᵢ] = E[vᵢ²] - E[vᵢ]²
        
        These moments characterize the distribution without requiring
        all 27 individual probabilities.
        
        VALIDATION STRATEGY:
        Compare empirical moments from quantum samples against theoretical
        moments computed from Maxwell-Boltzmann parameters.
        
        INPUT:
        velocity_counts: {(vₓ, vᵧ, vᵧ): count} from quantum measurements
        
        OUTPUT:
        Dictionary with keys: 'mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z'
        """
        total_shots = sum(velocity_counts.values())
        
        if total_shots == 0:
            return {
                'mean_x': 0.0, 'mean_y': 0.0, 'mean_z': 0.0,
                'var_x': 0.0, 'var_y': 0.0, 'var_z': 0.0
            }
        
        # Compute means for each dimension
        mean_x = sum(vx * count for (vx, _, _), count in velocity_counts.items()) / total_shots
        mean_y = sum(vy * count for (_, vy, _), count in velocity_counts.items()) / total_shots
        mean_z = sum(vz * count for (_, _, vz), count in velocity_counts.items()) / total_shots
        
        # Compute second moments E[vᵢ²]
        mean_x2 = sum(vx**2 * count for (vx, _, _), count in velocity_counts.items()) / total_shots
        mean_y2 = sum(vy**2 * count for (_, vy, _), count in velocity_counts.items()) / total_shots
        mean_z2 = sum(vz**2 * count for (_, _, vz), count in velocity_counts.items()) / total_shots
        
        # Compute variances: Var[vᵢ] = E[vᵢ²] - E[vᵢ]²
        var_x = mean_x2 - mean_x**2
        var_y = mean_y2 - mean_y**2
        var_z = mean_z2 - mean_z**2
        
        return {
            'mean_x': mean_x,
            'mean_y': mean_y,
            'mean_z': mean_z,
            'var_x': var_x,
            'var_y': var_y,
            'var_z': var_z
        }
    
    def compute_theoretical_moments_3d(
        self,
        mu_x: float,
        mu_y: float,
        mu_z: float,
        T: float
    ) -> Dict[str, float]:
        """
        Compute theoretical moments from Maxwell-Boltzmann parameters.
        
        THEORETICAL FORMULAS:
        For 1D discrete Gaussian with outcomes {-1, 0, +1}:
        - E[v] = μ (by construction of discrete Gaussian)
        - Var[v] = E[v²] - E[v]² 
        
        where E[v²] = Σ v² * P(v) computed from discrete Gaussian probabilities.
        
        INDEPENDENCE:
        Since dimensions are independent:
        - E[vₓ] = μₓ, E[vᵧ] = μᵧ, E[vᵧ] = μᵧ
        - Var[vₓ], Var[vᵧ], Var[vᵧ] computed independently
        
        PARAMETERS:
        - mu_x, mu_y, mu_z: mean velocities
        - T: temperature (variance parameter)
        
        OUTPUT:
        Dictionary with theoretical moment values
        """
        # For each dimension, compute moments using 1D discrete Gaussian
        moments = {}
        
        for dim, mu in [('x', mu_x), ('y', mu_y), ('z', mu_z)]:
            # Compute 1D probabilities for this dimension
            probs_1d = self.discrete_gaussian_probs(mu, T)
            p_minus1, p_0, p_plus1 = probs_1d
            
            # Theoretical mean: E[v] = Σ v * P(v)
            mean = (-1) * p_minus1 + 0 * p_0 + (+1) * p_plus1
            
            # Theoretical second moment: E[v²] = Σ v² * P(v)
            mean_v2 = (-1)**2 * p_minus1 + 0**2 * p_0 + (+1)**2 * p_plus1
            
            # Theoretical variance: Var[v] = E[v²] - E[v]²
            variance = mean_v2 - mean**2
            
            moments[f'mean_{dim}'] = mean
            moments[f'var_{dim}'] = variance
        
        return moments

    def compute_moments_lbm_style(
        self,
        probs_27: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute macroscopic moments from 27 probabilities using LBM formulas.
        
        STANDARD LBM MOMENT COMPUTATION:
        ρ = Σᵢ fᵢ                    (density)
        ρuₓ = Σᵢ fᵢ cᵢₓ               (momentum x)
        ρuᵧ = Σᵢ fᵢ cᵢᵧ               (momentum y)
        ρuᵧ = Σᵢ fᵢ cᵢᵧ               (momentum z)
        
        For probability distribution (ρ = 1):
        uₓ = Σᵢ₌₀²⁶ fᵢ cᵢₓ
        uᵧ = Σᵢ₌₀²⁶ fᵢ cᵢᵧ
        uᵧ = Σᵢ₌₀²⁶ fᵢ cᵢᵧ
        
        SECOND MOMENTS (for variance):
        Var[uₓ] = Σᵢ fᵢ cᵢₓ² - (Σᵢ fᵢ cᵢₓ)²
        Var[uᵧ] = Σᵢ fᵢ cᵢᵧ² - (Σᵢ fᵢ cᵢᵧ)²
        Var[uᵧ] = Σᵢ fᵢ cᵢᵧ² - (Σᵢ fᵢ cᵢᵧ)²
        
        PARAMETERS:
        - probs_27: Probability distribution in D3Q27 ordering (27 elements)
        
        RETURNS:
        Dictionary with moments: 'mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z'
        
        USAGE:
        ```python
        # From theoretical probabilities
        probs_27 = qdg.compute_3d_probability_distribution_lbm_order(mu_x, mu_y, mu_z, T)
        moments = qdg.compute_moments_lbm_style(probs_27)
        
        # From quantum samples
        velocity_counts = qdg.quantum_sample_grid_point_3d_parametric(mu_x, mu_y, mu_z, T)
        probs_27 = qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
        moments = qdg.compute_moments_lbm_style(probs_27)
        ```
        """
        # Get D3Q27 velocity components
        vX, vY, vZ = self.get_d3q27_velocity_ordering()
        
        # Compute density (should be 1.0 for normalized probabilities)
        rho = np.sum(probs_27)
        
        # First moments: uᵢ = Σ fⱼ cⱼᵢ
        mean_x = np.sum(probs_27 * vX) / rho
        mean_y = np.sum(probs_27 * vY) / rho
        mean_z = np.sum(probs_27 * vZ) / rho
        
        # Second moments: E[cᵢ²] = Σ fⱼ cⱼᵢ²
        mean_x2 = np.sum(probs_27 * vX**2) / rho
        mean_y2 = np.sum(probs_27 * vY**2) / rho
        mean_z2 = np.sum(probs_27 * vZ**2) / rho
        
        # Variance: Var[uᵢ] = E[cᵢ²] - E[cᵢ]²
        var_x = mean_x2 - mean_x**2
        var_y = mean_y2 - mean_y**2
        var_z = mean_z2 - mean_z**2
        
        return {
            'mean_x': mean_x,
            'mean_y': mean_y,
            'mean_z': mean_z,
            'var_x': var_x,
            'var_y': var_y,
            'var_z': var_z,
            'rho': rho
        }

    def get_d3q27_velocity_ordering(self):
        """
        Return D3Q27 lattice velocity ordering for LBM integration.
        
        LATTICE STRUCTURE:
        - SC (Simple Cubic): 6 velocities (face neighbors)
        - FCC (Face-Centered Cubic): 12 velocities (edge neighbors)
        - BCC (Body-Centered Cubic): 8 velocities (corner neighbors)
        - Rest: 1 velocity (center)
        Total: 27 velocities
        
        ORDERING:
        Index 0: (0, 0, 0) - rest particle
        Index 1-6: SC stencil (±1 on single axis)
        Index 7-18: FCC stencil (±1 on two axes, 0 on third)
        Index 19-26: BCC stencil (±1 on all three axes)
        
        RETURNS:
        Tuple of three arrays (vX, vY, vZ) each with 27 elements
        """
        cSC1 = 1
        cFCC1 = 1
        cBCC1 = 1
        
        vX = np.array([0, cSC1, -cSC1, 0, 0, 0, 0, cFCC1, -cFCC1, cFCC1, -cFCC1, 
                       cFCC1, -cFCC1, cFCC1, -cFCC1, 0, 0, 0, 0, cBCC1, -cBCC1, 
                       cBCC1, -cBCC1, cBCC1, -cBCC1, cBCC1, -cBCC1])
        
        vY = np.array([0, 0, 0, cSC1, -cSC1, 0, 0, cFCC1, -cFCC1, -cFCC1, cFCC1, 
                       0, 0, 0, 0, cFCC1, -cFCC1, cFCC1, -cFCC1, cBCC1, -cBCC1, 
                       -cBCC1, cBCC1, -cBCC1, cBCC1, cBCC1, -cBCC1])
        
        vZ = np.array([0, 0, 0, 0, 0, cSC1, -cSC1, 0, 0, 0, 0, cFCC1, -cFCC1, 
                       -cFCC1, cFCC1, cFCC1, -cFCC1, -cFCC1, cFCC1, cBCC1, -cBCC1, 
                       cBCC1, -cBCC1, -cBCC1, cBCC1, -cBCC1, cBCC1])
        
        return vX, vY, vZ
    
    def compute_3d_probability_distribution_lbm_order(
        self, 
        mu_x: float, 
        mu_y: float, 
        mu_z: float, 
        T: float
    ) -> np.ndarray:
        """
        Compute all 27 probabilities in D3Q27 LBM ordering.
        
        This function computes the theoretical probability distribution for 3D
        discrete Maxwell-Boltzmann velocities and returns them in the standard
        D3Q27 lattice ordering used in your LBM code.
        
        PROBABILITY COMPUTATION:
        Uses independence: P(vₓ, vᵧ, vᵧ) = P(vₓ) × P(vᵧ) × P(vᵧ)
        
        ORDERING:
        Matches your LBM code's velocity sequence:
        [0]: (0,0,0), [1]: (+1,0,0), [2]: (-1,0,0), etc.
        
        PARAMETERS:
        - mu_x, mu_y, mu_z: mean velocities in each direction
        - T: temperature/variance
        
        RETURNS:
        np.ndarray with shape (27,): probabilities in LBM ordering
        
        USAGE EXAMPLE:
        ```python
        # In your LBM code:
        qdg = QuantumDiscreteGaussian(circuit_type='symmetric')
        probs = qdg.compute_3d_probability_distribution_lbm_order(mu_x, mu_y, mu_z, T)
        # probs[i] is the probability for velocity direction i
        ```
        """
        # Get 1D probabilities for each dimension
        probs_x = self.discrete_gaussian_probs(mu_x, T)  # [P(-1), P(0), P(+1)]
        probs_y = self.discrete_gaussian_probs(mu_y, T)
        probs_z = self.discrete_gaussian_probs(mu_z, T)
        
        # Create mapping from velocity value to probability index
        # P(-1) at index 0, P(0) at index 1, P(+1) at index 2
        prob_map = {-1: 0, 0: 1, 1: 2}
        
        # Get D3Q27 velocity ordering
        vX, vY, vZ = self.get_d3q27_velocity_ordering()
        
        # Compute joint probabilities for all 27 directions
        probs_27 = np.zeros(27)
        for i in range(27):
            # Get velocity components for this direction
            vx = vX[i]
            vy = vY[i]
            vz = vZ[i]
            
            # Lookup 1D probabilities and multiply (independence)
            px = probs_x[prob_map[vx]]
            py = probs_y[prob_map[vy]]
            pz = probs_z[prob_map[vz]]
            
            probs_27[i] = px * py * pz
        
        return probs_27
    
    def convert_quantum_samples_to_lbm_order(
        self,
        velocity_counts: Dict[Tuple[int, int, int], int]
    ) -> np.ndarray:
        """
        Convert quantum velocity samples to LBM probability array.
        
        Takes the dictionary output from quantum_sample_grid_point_3d_parametric()
        and converts it to a 27-element probability array in D3Q27 LBM ordering.
        
        PARAMETERS:
        - velocity_counts: {(vₓ, vᵧ, vᵧ): count} from quantum sampling
        
        RETURNS:
        np.ndarray with shape (27,): empirical probabilities in LBM ordering
        
        USAGE EXAMPLE:
        ```python
        # Quantum sampling
        velocity_counts = qdg.quantum_sample_grid_point_3d_parametric(mu_x, mu_y, mu_z, T, shots=5000)
        
        # Convert to LBM format
        probs_lbm = qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
        
        # Now probs_lbm[i] matches your LBM code's velocity[i]
        ```
        """
        total_shots = sum(velocity_counts.values())
        
        if total_shots == 0:
            return np.zeros(27)
        
        # Get D3Q27 velocity ordering
        vX, vY, vZ = self.get_d3q27_velocity_ordering()
        
        # Convert counts to probabilities in LBM ordering
        probs_27 = np.zeros(27)
        for i in range(27):
            velocity_tuple = (vX[i], vY[i], vZ[i])
            count = velocity_counts.get(velocity_tuple, 0)
            probs_27[i] = count / total_shots
        
        return probs_27


    def _decode_quantum_counts_original(self, counts: Dict[str, int]) -> Dict[int, int]:
        """Convert raw Qiskit bitstring counts into outcome buckets {-1, 0, 1}.
        
        ORIGINAL ENCODING (for create_quantum_circuit_original):
        - |00⟩ → -1
        - |01⟩ → 0
        - |10⟩ → +1
        """
        outcome_counts = {-1: 0, 0: 0, 1: 0}
        for bitstring, count in counts.items():
            if len(bitstring) >= 2:
                qubit1_bit = bitstring[0]
                qubit0_bit = bitstring[1]
                if qubit0_bit == '0' and qubit1_bit == '0':
                    outcome_counts[-1] += count
                elif qubit0_bit == '0' and qubit1_bit == '1':
                    outcome_counts[0] += count
                elif qubit0_bit == '1' and qubit1_bit == '0':
                    outcome_counts[1] += count
        return outcome_counts
    
    def _decode_quantum_counts_symmetric(self, counts: Dict[str, int]) -> Dict[int, int]:
        """Convert raw Qiskit bitstring counts into outcome buckets {-1, 0, 1}.
        
        SYMMETRIC ENCODING (for create_quantum_circuit_symmetric):
        - |00⟩ → -1
        - |01⟩ → +1
        - |10⟩ → 0
        """
        outcome_counts = {-1: 0, 0: 0, 1: 0}
        for bitstring, count in counts.items():
            if len(bitstring) >= 2:
                qubit1_bit = bitstring[0]
                qubit0_bit = bitstring[1]
                if qubit0_bit == '0' and qubit1_bit == '0':
                    outcome_counts[-1] += count
                elif qubit0_bit == '0' and qubit1_bit == '1':
                    outcome_counts[1] += count  # Changed: |01⟩ → +1
                elif qubit0_bit == '1' and qubit1_bit == '0':
                    outcome_counts[0] += count  # Changed: |10⟩ → 0
        return outcome_counts
    
    def _decode_quantum_counts_3d(self, counts: Dict[str, int]) -> Dict[Tuple[int, int, int], int]:
        """
        Decode 6-qubit measurement results to 3D velocity tuples.
        
        ENCODING SCHEME (per 2-qubit pair):
        - |00⟩ → -1
        - |01⟩ → +1
        - |10⟩ → 0
        - |11⟩ → unused
        
        QUBIT MAPPING:
        - Qubits 0-1: vₓ component
        - Qubits 2-3: vᵧ component
        - Qubits 4-5: vᵧ component
        
        QISKIT BIT ORDER:
        Measurement string is reversed: bit 5 is leftmost, bit 0 is rightmost
        Example: '010001' means qubit[5]=0, qubit[4]=1, ..., qubit[0]=1
        
        OUTPUT:
        Dictionary mapping (vₓ, vᵧ, vᵧ) tuples to counts
        Example: {(-1, 0, +1): 235, (0, 0, 0): 189, ...}
        """
        velocity_counts = {}
        
        for bitstring, count in counts.items():
            if len(bitstring) >= 6:
                # Qiskit bit order: bit[5] bit[4] bit[3] bit[2] bit[1] bit[0]
                # Extract bits for each dimension (remember: rightmost = qubit 0)
                
                # X-component (qubits 0-1): rightmost 2 bits
                qubit0 = bitstring[5]  # bit[0] in Qiskit ordering
                qubit1 = bitstring[4]  # bit[1] in Qiskit ordering
                
                if qubit0 == '0' and qubit1 == '0':
                    vx = -1
                elif qubit0 == '0' and qubit1 == '1':
                    vx = 1
                elif qubit0 == '1' and qubit1 == '0':
                    vx = 0
                else:  # '11' - unused state
                    continue
                
                # Y-component (qubits 2-3): middle 2 bits
                qubit2 = bitstring[3]  # bit[2] in Qiskit ordering
                qubit3 = bitstring[2]  # bit[3] in Qiskit ordering
                
                if qubit2 == '0' and qubit3 == '0':
                    vy = -1
                elif qubit2 == '0' and qubit3 == '1':
                    vy = 1
                elif qubit2 == '1' and qubit3 == '0':
                    vy = 0
                else:  # '11' - unused state
                    continue
                
                # Z-component (qubits 4-5): leftmost 2 bits
                qubit4 = bitstring[1]  # bit[4] in Qiskit ordering
                qubit5 = bitstring[0]  # bit[5] in Qiskit ordering
                
                if qubit4 == '0' and qubit5 == '0':
                    vz = -1
                elif qubit4 == '0' and qubit5 == '1':
                    vz = 1
                elif qubit4 == '1' and qubit5 == '0':
                    vz = 0
                else:  # '11' - unused state
                    continue
                
                # Accumulate counts for this velocity tuple
                velocity_tuple = (vx, vy, vz)
                velocity_counts[velocity_tuple] = velocity_counts.get(velocity_tuple, 0) + count
        
        return velocity_counts
    
    def _decode_quantum_counts(self, counts: Dict[str, int]) -> Dict[int, int]:
        """
        Decode quantum measurement results using the configured circuit type.
        
        This method automatically routes to the correct decoder based on
        self.circuit_type (set in __init__), ensuring the decoder always
        matches the circuit implementation used.
        
        Users don't need to manually select the decoder - it's automatically
        paired with the circuit type.
        """
        if self.circuit_type == 'symmetric':
            return self._decode_quantum_counts_symmetric(counts)
        else:  # 'original'
            return self._decode_quantum_counts_original(counts)
    
    def print_circuit_example(self, grid_point: int = 0):
        """Print the quantum circuit for a specific grid point"""
        means, variances = self.compute_parameters()
        mu = means[grid_point]
        sigma_sq = variances[grid_point]
        probs = self.discrete_gaussian_probs(mu, sigma_sq)
        qc = self.create_quantum_circuit(probs)
        
        print(f"Grid Point {grid_point}: μ={mu:.4f}, σ²={sigma_sq:.4f}")
        print(qc.draw())
        return qc
    
    def test_alternative_circuit(self, grid_point: int = 0, shots: int = 10000):
        """
        Test and compare both circuit implementations to verify they produce identical results.
        
        This function validates that the symmetric decomposition {-1,+1} vs {0}
        produces the same probability distribution as the original decomposition {-1,0} vs {+1}.
        
        Note: This method temporarily overrides self.circuit_type to test both implementations.
        """
        means, variances = self.compute_parameters()
        mu = means[grid_point]
        sigma_sq = variances[grid_point]
        
        # Compute theoretical probabilities
        probs = self.discrete_gaussian_probs(mu, sigma_sq)
        
        print("=" * 70)
        print(f"CIRCUIT COMPARISON TEST - Grid Point {grid_point}")
        print("=" * 70)
        print(f"Parameters: μ={mu:.4f}, σ²={sigma_sq:.4f}")
        print(f"Theoretical: P(-1)={probs[0]:.4f}, P(0)={probs[1]:.4f}, P(+1)={probs[2]:.4f}")
        print()
        
        # Test original circuit
        print("ORIGINAL CIRCUIT (decomposition: {-1,0} vs {+1}):")
        qc_original = self.create_quantum_circuit_original(probs)
        print(qc_original.draw())
        print()
        
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(1, simulator)
        qc_compiled = pass_manager.run(qc_original)
        job = simulator.run(qc_compiled, shots=shots)
        result = job.result()
        counts_original = self._decode_quantum_counts_original(result.get_counts())
        
        probs_original = {k: v / shots for k, v in counts_original.items()}
        print(f"Original Circuit Results ({shots} shots):")
        print(f"  P(-1)={probs_original[-1]:.4f}, P(0)={probs_original[0]:.4f}, P(+1)={probs_original[1]:.4f}")
        
        tvd_original = 0.5 * sum(abs(probs_original[k] - probs[i]) for i, k in enumerate([-1, 0, 1]))
        print(f"  TVD from theoretical: {tvd_original:.6f}")
        print()
        
        # Test symmetric circuit
        print("SYMMETRIC CIRCUIT (decomposition: {-1,+1} vs {0}):")
        qc_symmetric = self.create_quantum_circuit_symmetric(probs)
        print(qc_symmetric.draw())
        print()
        
        qc_compiled = pass_manager.run(qc_symmetric)
        job = simulator.run(qc_compiled, shots=shots)
        result = job.result()
        counts_symmetric = self._decode_quantum_counts_symmetric(result.get_counts())
        
        probs_symmetric = {k: v / shots for k, v in counts_symmetric.items()}
        print(f"Symmetric Circuit Results ({shots} shots):")
        print(f"  P(-1)={probs_symmetric[-1]:.4f}, P(0)={probs_symmetric[0]:.4f}, P(+1)={probs_symmetric[1]:.4f}")
        
        tvd_symmetric = 0.5 * sum(abs(probs_symmetric[k] - probs[i]) for i, k in enumerate([-1, 0, 1]))
        print(f"  TVD from theoretical: {tvd_symmetric:.6f}")
        print()
        
        # Compare the two circuits
        tvd_between = 0.5 * sum(abs(probs_original[k] - probs_symmetric[k]) for k in [-1, 0, 1])
        print("COMPARISON:")
        print(f"  TVD between original and symmetric: {tvd_between:.6f}")
        print(f"  Both circuits produce equivalent distributions: {tvd_between < 0.01}")
        print("=" * 70)
        
        return {
            'theoretical': probs,
            'original': probs_original,
            'symmetric': probs_symmetric,
            'tvd_original': tvd_original,
            'tvd_symmetric': tvd_symmetric,
            'tvd_between': tvd_between
        }
    
    def test_parametric_circuit(self, grid_point: int = 0, shots: int = 10000):
        """
        Test and validate parametric circuit against probability-based symmetric circuit.
        
        This function verifies that the direct parametric implementation (which computes
        gate angles directly from mu and sigma_sq) produces identical results to the
        probability-based symmetric circuit implementation.
        
        VALIDATION STRATEGY:
        1. Compute theoretical probabilities from (mu, sigma_sq)
        2. Build circuit via probability-based method: (mu, sigma_sq) → probs → circuit
        3. Build circuit via parametric method: (mu, sigma_sq) → circuit (direct)
        4. Compare empirical distributions from both circuits
        5. Verify both match theoretical distribution within statistical error
        
        EXPECTED RESULT:
        TVD between parametric and probability-based < 0.01 (statistical noise only)
        """
        # Get parameters for the specified grid point
        means, variances = self.compute_parameters()
        mu = means[grid_point]
        sigma_sq = variances[grid_point]
        
        # Compute theoretical probabilities
        probs = self.discrete_gaussian_probs(mu, sigma_sq)
        
        print("=" * 70)
        print(f"PARAMETRIC CIRCUIT VALIDATION - Grid Point {grid_point}")
        print("=" * 70)
        print(f"Parameters: μ={mu:.4f}, σ²={sigma_sq:.4f}")
        print(f"Theoretical: P(-1)={probs[0]:.4f}, P(0)={probs[1]:.4f}, P(+1)={probs[2]:.4f}")
        print()
        
        # Verify we're using symmetric circuit
        if self.circuit_type != 'symmetric':
            print(f"WARNING: Switching to symmetric circuit for parametric test")
            print(f"         (current circuit_type='{self.circuit_type}')")
            print()
        
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(1, simulator)
        
        # Test probability-based symmetric circuit
        print("PROBABILITY-BASED SYMMETRIC CIRCUIT:")
        qc_probs = self.create_quantum_circuit_symmetric(probs)
        print(qc_probs.draw())
        print()
        
        qc_compiled = pass_manager.run(qc_probs)
        job = simulator.run(qc_compiled, shots=shots)
        result = job.result()
        counts_probs = self._decode_quantum_counts_symmetric(result.get_counts())
        
        probs_empirical = {k: v / shots for k, v in counts_probs.items()}
        print(f"Probability-Based Results ({shots} shots):")
        print(f"  P(-1)={probs_empirical[-1]:.4f}, P(0)={probs_empirical[0]:.4f}, P(+1)={probs_empirical[1]:.4f}")
        
        tvd_probs = 0.5 * sum(abs(probs_empirical[k] - probs[i]) for i, k in enumerate([-1, 0, 1]))
        print(f"  TVD from theoretical: {tvd_probs:.6f}")
        print()
        
        # Test parametric circuit
        print("PARAMETRIC CIRCUIT (direct angle computation):")
        qc_param = self.create_quantum_circuit_parametric(mu, sigma_sq)
        print(qc_param.draw())
        print()
        
        qc_compiled = pass_manager.run(qc_param)
        job = simulator.run(qc_compiled, shots=shots)
        result = job.result()
        counts_param = self._decode_quantum_counts_symmetric(result.get_counts())
        
        probs_parametric = {k: v / shots for k, v in counts_param.items()}
        print(f"Parametric Results ({shots} shots):")
        print(f"  P(-1)={probs_parametric[-1]:.4f}, P(0)={probs_parametric[0]:.4f}, P(+1)={probs_parametric[1]:.4f}")
        
        tvd_param = 0.5 * sum(abs(probs_parametric[k] - probs[i]) for i, k in enumerate([-1, 0, 1]))
        print(f"  TVD from theoretical: {tvd_param:.6f}")
        print()
        
        # Compare parametric vs probability-based
        tvd_between = 0.5 * sum(abs(probs_empirical[k] - probs_parametric[k]) for k in [-1, 0, 1])
        print("COMPARISON:")
        print(f"  TVD between probability-based and parametric: {tvd_between:.6f}")
        print(f"  Both methods produce equivalent distributions: {tvd_between < 0.01}")
        print()
        
        # Mathematical validation: verify angles are identical
        p = mu * mu + sigma_sq
        theta1_expected = 2 * np.arccos(np.sqrt(p))
        prob_plus1_given_motion = 0.5 * (1.0 + mu / p)
        theta2_expected = 2 * np.arcsin(np.sqrt(prob_plus1_given_motion))
        
        print("MATHEMATICAL VALIDATION:")
        print(f"  p = μ² + σ² = {p:.6f}")
        print(f"  θ₁ = 2*arccos(√p) = {theta1_expected:.6f} rad = {np.degrees(theta1_expected):.3f}°")
        print(f"  P(+1|motion) = 0.5*(1 + μ/p) = {prob_plus1_given_motion:.6f}")
        print(f"  θ₂ = 2*arcsin(√P(+1|motion)) = {theta2_expected:.6f} rad = {np.degrees(theta2_expected):.3f}°")
        print("=" * 70)
        
        return {
            'theoretical': probs,
            'probability_based': probs_empirical,
            'parametric': probs_parametric,
            'tvd_probability_based': tvd_probs,
            'tvd_parametric': tvd_param,
            'tvd_between': tvd_between,
            'angles': {
                'theta1': theta1_expected,
                'theta2': theta2_expected,
                'p': p
            }
        }
    
     
    def quantum_parallel_grid_sampling(
        self,
        shots_per_point: int = 1000,
        means: Optional[np.ndarray] = None,
        variances: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict[int, int]]:
        """Run the quantum sampler serially across the grid.

        The name retains the earlier "parallel" terminology for backward compatibility,
        but the implementation executes one circuit per grid point on the host.
        """
        if means is None or variances is None:
            means, variances = self.compute_parameters()
        
        print("Quantum grid sampling (one circuit per point)...")
        print("=" * 60)
        print(f"Grid points: {self.grid_size}")
        print(f"Shots per point: {shots_per_point}")
        print("Execution: Serial evaluation on classical host")
        print("Accuracy: High (individual parameters per grid point)")
        print("-" * 60)
        
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(1, simulator)

        compiled_circuits = []
        point_metadata = []  # Store classical parameters for later reporting

        for i in range(self.grid_size):
            mu = means[i]
            sigma_sq = variances[i]
            theoretical = self.discrete_gaussian_probs(mu, sigma_sq)

            qc = self.create_quantum_circuit(theoretical)
            qc.name = f"grid_point_{i}"
            qc_compiled = pass_manager.run(qc)
            compiled_circuits.append(qc_compiled)
            point_metadata.append((i, mu, sigma_sq, theoretical))

        job = simulator.run(compiled_circuits, shots=shots_per_point)
        result = job.result()

        results = {}

        for qc_compiled, (i, mu, sigma_sq, theoretical) in zip(compiled_circuits, point_metadata):
            raw_counts = result.get_counts(qc_compiled)
            outcome_counts = self._decode_quantum_counts(raw_counts)
            results[i] = outcome_counts

            total_shots = sum(outcome_counts.values())
            probs_empirical = {k: v / total_shots for k, v in outcome_counts.items()}

            print(f"Point {i}: μ={mu:.4f}, σ²={sigma_sq:.4f}")
            print(f"  Quantum:     P(-1)={probs_empirical[-1]:.3f}, P(0)={probs_empirical[0]:.3f}, P(1)={probs_empirical[1]:.3f}")
            print(f"  Theoretical: P(-1)={theoretical[0]:.3f}, P(0)={theoretical[1]:.3f}, P(1)={theoretical[2]:.3f}")

            tv_distance = 0.5 * sum(
                abs(probs_empirical[outcome] - theoretical[idx])
                for idx, outcome in enumerate([-1, 0, 1])
            )
            print(f"  TV Distance: {tv_distance:.4f}")
            print()

        print("Quantum sampling pass complete!")
        print("High accuracy maintained with individual parameter encoding")

        return results
    
    
    def plot_results(
        self,
        results: Dict[int, Dict[int, int]],
        means: np.ndarray,
        variances: np.ndarray,
        filename: str = 'results_quantum_sampling.png',
    ):
        """Enhanced visualization with comprehensive analysis.

        Expects precomputed ``means`` and ``variances`` to avoid redundant work.

        filename: path where the plot will be saved (defaults to 'results_quantum_sampling.png')
        """
        
        # Check if we have valid results
        total_measurements = sum(sum(point_results.values()) for point_results in results.values())
        if total_measurements == 0:
            print("  No valid measurements found. Skipping visualization.")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Parameter variations with improved dual y-axes
        x_points = np.arange(self.grid_size)
        
        # Left y-axis for mean (blue)
        color1 = 'tab:blue'
        ax1.plot(x_points, means, color=color1, marker='o', linewidth=2, markersize=6, label='Mean μ(x)')
        ax1.set_xlabel('Grid Point')
        ax1.set_ylabel('Mean μ(x)', color=color1, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([min(means)*1.1, max(means)*1.1])  # Add some margin
        
        # Right y-axis for variance (red)
        ax1_twin = ax1.twinx()
        color2 = 'tab:red'
        ax1_twin.plot(x_points, variances, color=color2, marker='s', linewidth=2, markersize=6, label='Variance T(x)')
        ax1_twin.axhline(y=self.T0, color='gray', linestyle='--', alpha=0.7, label=f'Base T₀={self.T0:.3f}')
        ax1_twin.set_ylabel('Variance T(x)', color=color2, fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor=color2)
        ax1_twin.set_ylim([min(variances)*0.95, max(variances)*1.05])  # Add some margin
        
        # Title and combined legend
        ax1.set_title('Spatially Varying Parameters (Dual Y-Axes)', fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
        
        # Prepare probability data for plots
        prob_matrix_quantum = np.zeros((self.grid_size, 3))
        prob_matrix_theoretical = np.zeros((self.grid_size, 3))
        
        for i in range(self.grid_size):
            # Quantum empirical probabilities
            total_shots = sum(results[i].values())
            if total_shots > 0:
                prob_matrix_quantum[i, 0] = results[i][-1] / total_shots  # P(-1)
                prob_matrix_quantum[i, 1] = results[i][0] / total_shots   # P(0)
                prob_matrix_quantum[i, 2] = results[i][1] / total_shots   # P(1)
            else:
                # Use theoretical probabilities if no measurements
                theoretical_probs = self.discrete_gaussian_probs(means[i], variances[i])
                prob_matrix_quantum[i, :] = theoretical_probs
            
            # Theoretical probabilities
            theoretical_probs = self.discrete_gaussian_probs(means[i], variances[i])
            prob_matrix_theoretical[i, :] = theoretical_probs
        
        # Plot 2: Line plots of all probabilities across grid
        colors = ['red', 'green', 'blue']
        markers = ['o', 's', '^']
        outcomes = ['-1', '0', '+1']
        
        for j, (outcome, color, marker) in enumerate(zip(outcomes, colors, markers)):
            # Theoretical lines (solid)
            ax2.plot(x_points, prob_matrix_theoretical[:, j], 
                    color=color, linestyle='-', linewidth=2.5, 
                    label=f'Theory P({outcome})', alpha=0.8)
            
            # Quantum empirical points and lines (dashed)
            ax2.plot(x_points, prob_matrix_quantum[:, j], 
                    color=color, linestyle='--', linewidth=2, 
                    marker=marker, markersize=6, markerfacecolor='white',
                    label=f'Quantum P({outcome})', alpha=0.9)
        
        ax2.set_xlabel('Grid Point')
        ax2.set_ylabel('Probability')
        ax2.set_title('Theoretical vs Quantum Probabilities Across Grid')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Mean and Variance Recovery from Quantum Moments
        # Compute recovered moments from quantum samples
        recovered_means = np.zeros(self.grid_size)
        recovered_variances = np.zeros(self.grid_size)
        
        for i in range(self.grid_size):
            total_shots = sum(results[i].values())
            if total_shots > 0:
                # Compute mean: E[v] = sum(v * count) / total
                mean_recovered = (results[i][-1] * (-1) + results[i][0] * 0 + results[i][1] * 1) / total_shots
                
                # Compute variance: Var[v] = E[v²] - E[v]²
                mean_v2 = (results[i][-1] * 1 + results[i][0] * 0 + results[i][1] * 1) / total_shots
                var_recovered = mean_v2 - mean_recovered**2
                
                recovered_means[i] = mean_recovered
                recovered_variances[i] = var_recovered
            else:
                # If no measurements, use theoretical values
                recovered_means[i] = means[i]
                recovered_variances[i] = variances[i]
        
        # Plot mean comparison
        ax3.plot(x_points, means, 'b-o', linewidth=2.5, markersize=7, 
                label='Input μ(x)', alpha=0.8)
        ax3.plot(x_points, recovered_means, 'r--s', linewidth=2, markersize=6, 
                markerfacecolor='white', label='Recovered E[v]', alpha=0.9)
        ax3.set_xlabel('Grid Point')
        ax3.set_ylabel('Mean Velocity')
        ax3.set_title('Mean: Input vs Recovered from Quantum Moments', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        # Add error statistics for mean
        mean_error = np.mean(np.abs(means - recovered_means))
        max_mean_error = np.max(np.abs(means - recovered_means))
        ax3.text(0.02, 0.95, f'Mean Error: {mean_error:.5f}\nMax Error: {max_mean_error:.5f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Plot 4: Variance comparison
        ax4.plot(x_points, variances, 'b-o', linewidth=2.5, markersize=7, 
                label='Input T(x)', alpha=0.8)
        ax4.plot(x_points, recovered_variances, 'r--s', linewidth=2, markersize=6, 
                markerfacecolor='white', label='Recovered Var[v]', alpha=0.9)
        ax4.axhline(y=self.T0, color='gray', linestyle=':', alpha=0.7, 
                   label=f'Base T₀={self.T0:.3f}')
        ax4.set_xlabel('Grid Point')
        ax4.set_ylabel('Variance / Temperature')
        ax4.set_title('Variance: Input vs Recovered from Quantum Moments', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best')
        
        # Add error statistics for variance
        var_error = np.mean(np.abs(variances - recovered_variances))
        max_var_error = np.max(np.abs(variances - recovered_variances))
        ax4.text(0.02, 0.95, f'Mean Error: {var_error:.5f}\nMax Error: {max_var_error:.5f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()



def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Quantum Discrete Gaussian Distribution on 1D Grid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--shots',
        type=int,
        default=5000,
        help='Number of quantum circuit shots per grid point'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=10,
        help='Number of grid points'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results_quantum_sampling.png',
        help='Output filename for visualization'
    )
    args = parser.parse_args()
    
    print("Quantum Discrete Gaussian Distribution on 1D Grid")
    print("=" * 55)
    
    # Initialize quantum discrete Gaussian
    qdg = QuantumDiscreteGaussian(grid_size=args.grid_size)
    
    # Display parameter setup
    means, variances = qdg.compute_parameters()
    print(f"Grid size: {qdg.grid_size}")
    print(f"Base variance T₀: {qdg.T0}")
    print(f"Mean range: [{means.min():.4f}, {means.max():.4f}]")
    print(f"Variance range: [{variances.min():.4f}, {variances.max():.4f}]")
    print(f"Outcomes: {qdg.outcomes}")
    print(f"Shots per point: {args.shots}")
    print()
    
    # Show example circuit for the first grid point
    print("EXAMPLE QUANTUM CIRCUIT:")
    qdg.print_circuit_example(grid_point=1)
    print()

    # Quantum sampling sweep (serial per grid point)
    results = qdg.quantum_parallel_grid_sampling(
        shots_per_point=args.shots,
        means=means,
        variances=variances,
    )
    qdg.plot_results(results, means, variances, filename=args.output)
    
    print(f"Analysis complete! Check '{args.output}' for visualizations.")

if __name__ == "__main__":
    main()