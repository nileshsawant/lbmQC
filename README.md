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

### Circuit Design
1. **RY rotation** on first qubit to control P(first bit = 0) vs P(first bit = 1)
2. **Controlled RY rotation** on second qubit to fine-tune the probability distribution
3. **Measurement** to sample outcomes

### Key Features
- **Amplitude encoding**: Direct encoding of probabilities as quantum amplitudes
- **Parallelization**: Each grid point processed with independent quantum circuits
- **High fidelity**: Exact probability encoding without approximation errors

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the quantum simulation
python quantum_discrete_gaussian.py
```

## Output

The program generates:
1. **Console output**: Parameter values and probability comparisons for each grid point
2. **Visualization**: Four-panel plot showing:
   - Mean parameter variation across grid
   - Variance parameter variation across grid  
   - 2D heatmap of empirical probability distributions
   - Theoretical vs quantum empirical comparison at middle grid point

## Algorithm Complexity

- **Qubits required**: 2 per grid point
- **Gate depth**: O(1) per point (constant depth)
- **Classical preprocessing**: O(N) where N = grid size
- **Quantum advantage**: Potential for true parallelization across grid points on quantum hardware

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