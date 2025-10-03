"""
Quantum implementation for discrete Gaussian distribution on 1D grid
with sine wave variations in mean (u) and variance (T).

Grid: 10 points
Mean: u(x) = 0.1 * sin(2π * x/10)  
Variance: T(x) = T0 + 0.05 * sin(2π * x/10), where T0 = 1/3
Outcomes: {-1, 0, 1}
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import math
from typing import List, Tuple, Dict

class QuantumDiscreteGaussian:
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.T0 = 1/3  # Base variance
        self.outcomes = [-1, 0, 1]
        
    def compute_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and variance for each grid point"""
        x_points = np.arange(self.grid_size)
        
        # Mean: u(x) = 0.1 * sin(2π * x/10)
        means = 0.1 * np.sin(2 * np.pi * x_points / self.grid_size)
        
        # Variance: T(x) = T0 + 0.05 * sin(2π * x/10)
        variances = self.T0 + 0.05 * np.sin(2 * np.pi * x_points / self.grid_size)
        
        return means, variances
    
    def classical_discrete_gaussian_probs(self, mu: float, sigma_sq: float) -> np.ndarray:
        """Compute discrete Gaussian probabilities classically"""
        # Compute unnormalized probabilities
        unnorm_probs = np.array([
            np.exp(-(k - mu)**2 / (2 * sigma_sq)) for k in self.outcomes
        ])
        
        # Normalize
        Z = np.sum(unnorm_probs)
        return unnorm_probs / Z
    

    
    def create_quantum_circuit(self, probs: np.ndarray) -> QuantumCircuit:
        """Create quantum circuit for single grid point discrete Gaussian"""
        # Normalize probabilities
        probs = probs / np.sum(probs)
        p_minus1, p_0, p_plus1 = probs
        
        # Create circuit to generate: |ψ⟩ = √p₋₁|00⟩ + √p₀|01⟩ + √p₊₁|10⟩
        qc = QuantumCircuit(2, 2)
        
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
    
    def quantum_sample_grid_point(self, mu: float, sigma_sq: float, shots: int = 1000) -> Dict[int, int]:
        """Sample from discrete Gaussian at a single grid point using quantum circuit"""
        # Compute classical probabilities
        probs = self.classical_discrete_gaussian_probs(mu, sigma_sq)
        
        # Create and run quantum circuit
        qc = self.create_quantum_circuit(probs)
        
        # Simulate
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(1, simulator)
        qc_compiled = pass_manager.run(qc)
        
        # Run the circuit
        job = simulator.run(qc_compiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert bit strings to outcomes
        # Format from Qiskit: 'b1b0' where b1 is qubit 1, b0 is qubit 0
        outcome_counts = {-1: 0, 0: 0, 1: 0}
        for bitstring, count in counts.items():
            if len(bitstring) >= 2:
                qubit1_bit = bitstring[0]  # Second qubit (qubit 1)
                qubit0_bit = bitstring[1]  # First qubit (qubit 0)
                
                # Our encoding: |q1q0⟩ → outcome
                if qubit0_bit == '0' and qubit1_bit == '0':
                    outcome_counts[-1] += count  # |00⟩ → -1
                elif qubit0_bit == '0' and qubit1_bit == '1':
                    outcome_counts[0] += count   # |01⟩ → 0
                elif qubit0_bit == '1' and qubit1_bit == '0':
                    outcome_counts[1] += count   # |10⟩ → +1
                # |11⟩ is unused in our encoding
        
        return outcome_counts
    
    def quantum_parallel_grid_sampling(self, shots_per_point: int = 1000) -> Dict[int, Dict[int, int]]:
        """Sample from all grid points (simulating quantum parallelization)"""
        means, variances = self.compute_parameters()
        results = {}
        
        print("Quantum sampling across 1D grid...")
        print(f"Grid points: {self.grid_size}")
        print(f"Shots per point: {shots_per_point}")
        print("-" * 50)
        
        for i in range(self.grid_size):
            mu = means[i]
            sigma_sq = variances[i]
            
            print(f"Point {i}: μ={mu:.4f}, σ²={sigma_sq:.4f}")
            
            # Sample using quantum circuit
            outcome_counts = self.quantum_sample_grid_point(mu, sigma_sq, shots_per_point)
            results[i] = outcome_counts
            
            # Print results for this point
            total_shots = sum(outcome_counts.values())
            probs_empirical = {k: v/total_shots for k, v in outcome_counts.items()}
            probs_theoretical = self.classical_discrete_gaussian_probs(mu, sigma_sq)
            
            print(f"  Empirical:   P(-1)={probs_empirical[-1]:.3f}, P(0)={probs_empirical[0]:.3f}, P(1)={probs_empirical[1]:.3f}")
            print(f"  Theoretical: P(-1)={probs_theoretical[0]:.3f}, P(0)={probs_theoretical[1]:.3f}, P(1)={probs_theoretical[2]:.3f}")
            print()
        
        return results
    
    def plot_results(self, results: Dict[int, Dict[int, int]]):
        """Plot the results showing parameter variations and probability distributions"""
        means, variances = self.compute_parameters()
        
        # Create subplots - 2x2 layout
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
            prob_matrix_quantum[i, 0] = results[i][-1] / total_shots  # P(-1)
            prob_matrix_quantum[i, 1] = results[i][0] / total_shots   # P(0)
            prob_matrix_quantum[i, 2] = results[i][1] / total_shots   # P(1)
            
            # Theoretical probabilities
            theoretical_probs = self.classical_discrete_gaussian_probs(means[i], variances[i])
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
        
        # Plot 3: Heatmap of quantum empirical probabilities
        im = ax3.imshow(prob_matrix_quantum.T, aspect='auto', cmap='viridis', 
                       origin='lower', vmin=0, vmax=1)
        ax3.set_xlabel('Grid Point')
        ax3.set_ylabel('Outcome')
        ax3.set_title('Quantum Empirical Probability Heatmap')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['-1', '0', '+1'])
        plt.colorbar(im, ax=ax3, label='Probability')
        
        # Plot 4: Error analysis across grid
        errors = np.zeros(self.grid_size)
        for i in range(self.grid_size):
            # Calculate total variation distance at each point
            error = 0.5 * np.sum(np.abs(prob_matrix_theoretical[i, :] - prob_matrix_quantum[i, :]))
            errors[i] = error
        
        ax4.plot(x_points, errors, 'mo-', linewidth=2, markersize=8, 
                label=f'TVD Error (Mean: {np.mean(errors):.4f})')
        ax4.set_xlabel('Grid Point')
        ax4.set_ylabel('Total Variation Distance')
        ax4.set_title('Quantum vs Theoretical Error Across Grid')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add overall statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        ax4.axhline(y=mean_error, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Mean Error: {mean_error:.4f}')
        ax4.text(0.02, 0.95, f'Max Error: {max_error:.4f}\nMean Error: {mean_error:.4f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('results_improved.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    print("Quantum Discrete Gaussian Distribution on 1D Grid")
    print("=" * 55)
    
    # Initialize quantum discrete Gaussian
    qdg = QuantumDiscreteGaussian(grid_size=10)
    
    # Display parameter setup
    means, variances = qdg.compute_parameters()
    print(f"Grid size: {qdg.grid_size}")
    print(f"Base variance T₀: {qdg.T0}")
    print(f"Mean range: [{means.min():.4f}, {means.max():.4f}]")
    print(f"Variance range: [{variances.min():.4f}, {variances.max():.4f}]")
    print(f"Outcomes: {qdg.outcomes}")
    print()
    
    # Run quantum simulation
    results = qdg.quantum_parallel_grid_sampling(shots_per_point=2000)
    
    # Plot and analyze results
    qdg.plot_results(results)
    
    print("Analysis complete! Check 'results.png' for visualizations.")

if __name__ == "__main__":
    main()