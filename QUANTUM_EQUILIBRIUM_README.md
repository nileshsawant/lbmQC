# Quantum Equilibrium Distribution for LBM

Implementation of quantum sampling-based equilibrium distributions for the Lattice Boltzmann Method (LBM) using GPU-accelerated quantum circuits.

## Overview

This implementation replaces the classical Maxwell-Boltzmann equilibrium distribution in LBM with quantum circuit-based sampling. The quantum approach samples from a discrete Gaussian distribution over the D3Q27 lattice velocities using 6-qubit quantum circuits executed on GPU.

## Files Created

### 1. `quantum_lbm_equilibrium.py`
Main module containing the quantum equilibrium distribution implementation.

**Key Functions:**
- `quantumEqDistribution(ux, uy, uz, T, shots=20000)`: Main interface for computing quantum equilibrium
- `convert_quantum_samples_to_d3q27()`: Maps quantum velocities to D3Q27 lattice order
- `execute_quantum_circuits_batch_gpu()`: GPU-accelerated batch circuit execution
- `test_quantum_equilibrium()`: Built-in test function

### 2. `test_quantum_equilibrium_quick.py`
Standalone test script for validation.

**Usage:**
```bash
# Quick test on login node (2×2×2 grid, 1000 shots)
python test_quantum_equilibrium_quick.py

# Full test on HPC node (4×4×4 grid, 20000 shots)
python test_quantum_equilibrium_quick.py --full
```

### 3. Updated Notebook
- Added import cell for quantum module
- Added detailed usage comments in collision function

## Installation & Environment

### Activate Qiskit GPU Environment
```bash
source /kfs2/projects/hpcapps/nsawant/qcBac/lbmQC/envs/qiskit_gpu_fresh/bin/activate
```

### Required Packages
- `qiskit` (Qiskit SDK)
- `qiskit-aer` (with GPU support)
- `numpy`
- `cupy` (optional, for CuPy array compatibility)

## Quick Start

### 1. Login Node Test (Before HPC Allocation)

Test with small grid to verify installation:

```bash
# Activate environment
source envs/qiskit_gpu_fresh/bin/activate

# Run quick test (8 grid points, 1000 shots)
python test_quantum_equilibrium_quick.py
```

Expected output:
```
Grid: 2 × 2 × 2 = 8 points
Shots: 1000
Execution time: ~10-20s
Status: ✓ PASS
```

### 2. HPC Compute Node Test

Request GPU allocation and run full test:

```bash
# Request interactive GPU node
salloc -N 1 -p gpu --gres=gpu:1 --time=01:00:00

# Activate environment
source envs/qiskit_gpu_fresh/bin/activate

# Run full test (64 grid points, 20000 shots)
python test_quantum_equilibrium_quick.py --full
```

Expected output:
```
Grid: 4 × 4 × 4 = 64 points
Shots: 20000
Total measurements: 1,280,000
Execution time: ~2-5 minutes on H100
Status: ✓ PASS
```

## Integration with LBM Notebook

### Step 1: Import Module

In the notebook, locate cell 8 (after GPU/CPU selection) and uncomment:

```python
from quantum_lbm_equilibrium import quantumEqDistribution
```

### Step 2: Use in Collision Function

In cell 37 (collision function `collideMultiphaseTwoPopStandardLatticeAndPSWithReturn`), around line 2056, replace the classical equilibrium with:

```python
# OPTION A: Quantum equilibrium (slower, for validation/research)
quantum_probs = quantumEqDistribution(ux, uy, uz, T, shots=20000, verbose=False)
grid_fEq_a = Y_a * moments[:,:,:,m.rho,np.newaxis] * quantum_probs

# OPTION B: Classical equilibrium (original, fast)
# grid_fEq_a = Y_a * moments[:,:,:,m.rho,np.newaxis] * classical_product_form(...)
```

### Step 3: Adjust Parameters

**For Production (H100 GPU):**
```python
probs = quantumEqDistribution(ux, uy, uz, T, shots=20000, batch_size=512)
```

**For Testing (login node):**
```python
probs = quantumEqDistribution(ux, uy, uz, T, shots=1000, batch_size=128)
```

**For Debugging:**
```python
probs = quantumEqDistribution(ux, uy, uz, T, shots=5000, verbose=True)
```

## Performance Characteristics

### Execution Time (NVIDIA H100)

| Grid Size | Points | Shots | Time | Measurements/sec |
|-----------|--------|-------|------|------------------|
| 2×2×2 | 8 | 1,000 | ~10s | ~800 |
| 2×2×2 | 8 | 20,000 | ~30s | ~5,300 |
| 4×4×4 | 64 | 20,000 | ~3min | ~7,000 |
| 10×10×10 | 1,000 | 20,000 | ~45min | ~7,400 |

**Note:** Quantum approach is **~1000× slower** than classical equilibrium. Use for:
- Validation of classical implementations
- Research on quantum effects in fluid dynamics
- Benchmarking quantum algorithms

### Memory Requirements

- **CPU Memory:** ~100 MB for circuits + ~1 GB for results (per 1000 points)
- **GPU Memory:** ~2-4 GB for AerSimulator (H100 has 80GB)
- **Peak Memory:** During transpilation (~2× circuit memory)

## Technical Details

### D3Q27 Lattice Mapping

The quantum circuits output 3D velocities as tuples (vx, vy, vz) where each component is in {-1, 0, 1}. These are mapped to the D3Q27 lattice velocity ordering:

```python
# Example: quantum output (1, -1, 0) maps to D3Q27 index based on:
D3Q27_VX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, ...]  # 27 components
D3Q27_VY = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, ...]
D3Q27_VZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, ...]
```

### Quantum Circuit Structure

Each grid point uses a 6-qubit circuit:
- **Qubits 0-1:** vx component (x-velocity)
- **Qubits 2-3:** vy component (y-velocity)
- **Qubits 4-5:** vz component (z-velocity)

Circuit depth: 4 layers (constant, independent of grid size)

### GPU Acceleration

Circuits are executed in batches on GPU:
1. **Generate circuits:** All grid points (parallelizable)
2. **Transpile:** Optimize for GPU backend (batched)
3. **Execute:** Run batches on AerSimulator with GPU device
4. **Process results:** Convert counts to D3Q27 probabilities

Batch size of 512 optimized for H100 (80GB VRAM).

## Validation

### Moment Conservation

The quantum equilibrium should conserve moments:

```python
# Normalization
assert np.allclose(probs.sum(axis=-1), 1.0)

# Mean velocity
mean_vx = np.sum(probs * D3Q27_VX, axis=-1)
assert np.allclose(mean_vx, ux, atol=0.01)  # Statistical error depends on shots
```

### Classical vs Quantum Comparison

Create comparison test:

```python
# Compute both equilibria
probs_classical = classical_equilibrium(ux, uy, uz, T)
probs_quantum = quantumEqDistribution(ux, uy, uz, T, shots=20000)

# Compare moments
moments_classical = compute_moments(probs_classical)
moments_quantum = compute_moments(probs_quantum)

# Check agreement
assert np.allclose(moments_classical['mean_vx'], moments_quantum['mean_vx'], atol=0.01)
```

## Troubleshooting

### GPU Not Detected

**Error:** `AerSimulator` falls back to CPU

**Solution:**
```bash
# Check CUDA availability
python -c "import qiskit_aer; print(qiskit_aer.AerSimulator().available_devices())"

# Should show: ['CPU', 'GPU']

# If only CPU, reinstall qiskit-aer-gpu:
pip install qiskit-aer-gpu
```

### Out of Memory

**Error:** CUDA out of memory during execution

**Solution:** Reduce batch size
```python
probs = quantumEqDistribution(ux, uy, uz, T, batch_size=256)  # Default: 512
```

### Numerical Instabilities

**Error:** `ValueError: Invalid parameters for X: p_x = mu_x² + sigma_sq must be < 1`

**Cause:** Velocity or temperature values outside quantum circuit constraints

**Solution:** Parameters are automatically clamped, but you can preprocess:
```python
# Clamp velocities
ux = np.clip(ux, -0.5, 0.5)
uy = np.clip(uy, -0.5, 0.5)
uz = np.clip(uz, -0.5, 0.5)

# Ensure temperature in valid range
T = np.clip(T, 0.01, 0.3)
```

### Slow Execution on Login Node

**Expected:** Login nodes have limited resources

**Solution:**
- Reduce grid size: `--Nx 2 --Ny 2 --Nz 2`
- Reduce shots: `--shots 500`
- Request compute node for full tests

## Future Optimizations

### Potential Improvements

1. **Circuit Caching:** Cache compiled circuits for repeated parameter values
2. **Adaptive Shots:** Use fewer shots where distribution is smooth
3. **Sparse Sampling:** Sample subset of grid points, interpolate rest
4. **Multi-GPU:** Distribute batches across multiple GPUs
5. **Circuit Compression:** Reduce circuit depth for faster execution

### Research Directions

1. **Quantum Effects:** Investigate quantum corrections to Navier-Stokes
2. **Benchmark Studies:** Compare quantum vs classical accuracy
3. **Hardware Testing:** Profile on different quantum hardware
4. **Hybrid Approaches:** Combine quantum and classical equilibria

## Citation

If you use this code in research, please cite:

```
@software{quantum_lbm_equilibrium,
  author = {Sawant, Nilesh},
  title = {Quantum Equilibrium Distribution for Lattice Boltzmann Method},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nileshsawant/lbmQC}
}
```

## License

Same as parent repository (see main README.md)

## Contact

- **Author:** Nilesh Sawant
- **Date:** November 22, 2025
- **GPU:** NVIDIA H100
- **Institution:** HPC Applications

## Appendix: Parameter Constraints

### Quantum Circuit Constraints

For successful circuit generation, parameters must satisfy:

```
μx² + σ² < 1
μy² + σ² < 1  
μz² + σ² < 1
σ² > 0
```

Where:
- μx, μy, μz are mean velocities (ux, uy, uz in LBM)
- σ² is temperature (T in LBM)

**Typical LBM Values:**
- Velocities: |ux|, |uy|, |uz| < 0.1 (subsonic flow)
- Temperature: T = 1/3 (standard lattice temperature)
- Satisfies constraints: (0.1)² + 1/3 ≈ 0.34 < 1 ✓

**Maximum Allowed:**
- If T = 1/3, then |ux|, |uy|, |uz| < 0.82
- If |u| = 0.1, then T < 0.99

### D3Q27 Velocity Range

All quantum velocities are in {-1, 0, 1}, matching D3Q27 exactly:
- No velocity rescaling needed
- Direct mapping to lattice indices
- Perfect alignment with LBM formulation
