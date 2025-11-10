# Quantum Discrete Gaussian Distribution for lbmQC

This module implements quantum computing algorithms for generating discrete Gaussian distributions on 1D grids with spatially varying parameters.

## What This Adds to lbmQC

- **Quantum discrete Gaussian sampling** over outcomes {-1, 0, 1}
- **Spatially varying parameters**: Mean and variance with sine wave modulation
- **2-qubit quantum circuits** using amplitude encoding
- **Grid-based probability distributions** for LBM applications

## Problem Setup

- **Grid**: 10 points in 1D
- **Mean variation**: μ(x) = 0.1 × sin(2π × x/10)  
- **Variance variation**: T(x) = T₀ + 0.05 × sin(2π × x/10), where T₀ = 1/3
- **Outcomes**: {-1, 0, 1}

## Quantum Algorithm

### Approach
Uses a 2-qubit quantum circuit to encode the discrete Gaussian distribution:

- |00⟩ → outcome -1
- |01⟩ → outcome 0  
- |10⟩ → outcome +1
- |11⟩ → unused

## Implementation Overview

### Grid Model

- 10 lattice points sampled individually on a classical loop
- Outcomes {-1, 0, 1} amplitude-encoded on two qubits
- Mean μ(x) = 0.1 × sin(2π × x/10)
- Variance T(x) = 1/3 + 0.05 × sin(2π × x/10)

### Core Components

- `compute_parameters(grid_size)`: returns arrays of means and variances for each point
- `classical_discrete_gaussian_probs(mu, sigma_sq)`: normalized probabilities for {-1, 0, 1}
- `create_quantum_circuit(probs)`: 2-qubit amplitude-encoding circuit with controlled rotations and measurement
- `_decode_quantum_counts(counts)`: maps measurement bitstrings to outcome counts
- `quantum_sample_grid_point(mu, sigma_sq, shots)`: compiles, runs, decodes one grid point on `AerSimulator`
- `quantum_parallel_grid_sampling(shots_per_point, means, variances)`: batches all circuits, compares empirical vs theoretical probabilities, reports total variation distance
- `plot_results(results, means, variances, filename)`: four-panel matplotlib visualization
- `main()`: orchestrates the workflow and saves `results_quantum_sampling.png`

### Execution Flow

1. Compute spatially varying means and variances.
2. Build the per-point circuits from the classical probabilities.
3. Run `AerSimulator.run([...], shots_per_point)` once with all compiled circuits.
4. Decode measurement counts and compare to classical probabilities.
5. Plot parameter profiles, empirical vs theoretical probabilities, the heatmap, and total variation distance.

### Outputs

- Console summary per grid point with empirical vs theoretical probabilities plus total variation distance
- Plot saved as `results_quantum_sampling.png`

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the quantum simulation
python quantum_discrete_gaussian.py
```

## Output

The program generates:

1. **Console output**: Parameter values and probability comparisons for each grid point.
2. **Visualization**: Four-panel plot showing:

    - Mean parameter variation across the grid
    - Variance parameter variation across the grid
    - 2D heatmap of empirical probability distributions
    - Theoretical vs quantum empirical comparison at each grid point

## Algorithm Complexity

- **Qubits required**: 2 per grid point
- **Gate depth**: O(1) per point (constant depth)
- **Classical preprocessing**: O(N) where N = grid size
- **Current workflow**: Host executes one circuit per grid point (simulator batching reduces classical overhead but is not quantum parallelism)

## Theoretical Background

For discrete Gaussian distribution over {-1, 0, 1}:

$$P(X = k) = \frac{e^{-\frac{(k-\mu)^2}{2\sigma^2}}}{\sum_{j \in \{-1,0,1\}} e^{-\frac{(j-\mu)^2}{2\sigma^2}}}$$

The quantum circuit implements this by:

1. Computing rotation angles from the classical probability values
2. Applying parameterized quantum gates to create the desired amplitude distribution
3. Measuring to sample from the encoded distribution

## Extensions

This implementation can be extended to:

- Larger grids (requires more qubits for grid encoding)
- More outcome values (requires additional qubits or qutrits)
- 2D/3D grids with spatial correlations
- Real quantum hardware execution (currently uses Aer simulator)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{sawant2025_quantum_discrete_gaussian,
  title = {Quantum Discrete Gaussian Distribution for lbmQC},
  author = {Sawant, Nilesh}, 
  year = {2025},
  url = {https://github.com/nileshsawant/lbmQC},
  version={0.1},
  month={10},
  keywords = {quantum computing, discrete Gaussian distribution, amplitude encoding, quantum circuits, lattice Boltzmann method, quantum sampling, probabilistic modeling, qiskit, quantum simulation, spatial probability distributions}
}
```
