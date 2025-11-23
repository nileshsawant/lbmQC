# Quantum LBM Integration Guide

This document describes how to integrate `QuantumLBMInterface` from
`quantum_lbm_interface.py` into a classical Lattice Boltzmann Method (LBM)
time-stepping driver. It covers the API contract, recommended usage
patterns, performance tuning, validation checks, and examples you can copy
into your code.

## Overview

`QuantumLBMInterface` converts macroscopic fields — velocities `(ux,uy,uz)` and
temperature `T` — into D3Q27 equilibrium probability distributions using a
quantum-sampling backend (Qiskit Aer). It supports batching, adaptive shot
count, caching of results, and caching transpiled circuits for repeat runs.

Key features:
- `compute(ux,uy,uz,T)` — compute per-grid-point quantum probabilities (batched)
- `precompile_grid(...)` and `run_precompiled_grid(...)` — transpile then run
- Caching and `round_digits` parameter to quantize keys and increase hit rates

## API Contract

- Constructor: `QuantumLBMInterface(grid_shape=(nZ,nY,nX), use_gpu=True, backend='cupy'|'numpy', round_digits=4, cache_size=10000, min_shots=100, max_shots=5000)`
- Inputs for compute:
  - `ux,uy,uz`: arrays shaped `(nZ,nY,nX)` (NumPy or CuPy depending on `backend`)
  - `T`: array shaped `(nZ,nY,nX)` or `(nZ,nY,nX,1)`.
- Output: `probs` shaped `(nZ,nY,nX,27)` (D3Q27 ordering). Multiply by density `rho[...,None]` to get `feq`.

Error modes:
- `ImportError` if `backend='cupy'` requested but `cupy` not installed.
- `RuntimeError` from `run_precompiled_grid` when precompiled circuits are missing.
- Shape mismatch errors if inputs don't match `grid_shape` supplied at construction.

## Minimal integration recipe

1. Instantiate once at startup:

```python
from quantum_lbm_interface import QuantumLBMInterface

qlbm = QuantumLBMInterface(grid_shape=(nZ,nY,nX), use_gpu=True, backend='cupy', round_digits=4)
```

2. Ensure your velocity and temperature arrays are on the same array backend used
   by the interface (use `qlbm.xp.asarray(...)` to convert if needed).

3. Each timestep call:

```python
probs = qlbm.compute(ux,uy,uz,T)  # shape (nZ,nY,nX,27)
feq = rho[..., qlbm.xp.newaxis] * probs
```

## Precompile pattern

For stable / repeating parameter sets:

```python
qlbm.precompile_grid(ux_initial, uy_initial, uz_initial, T_initial)
for t in range(n_steps):
    probs = qlbm.run_precompiled_grid(ux,uy,uz,T)
    feq = rho[..., qlbm.xp.newaxis] * probs
```

`precompile_grid` transpiles unique circuits without running them; `run_precompiled_grid`
then executes those cached circuits. Use this pattern when the set of unique
parameter keys is known in advance.

## Chunking for large grids

If too many unique circuits are produced at once, compute by slabs/tiles:

```python
def compute_in_chunks(qlbm, ux, uy, uz, T, z_chunk=8):
    nZ = ux.shape[0]
    out = qlbm.xp.zeros((nZ, ux.shape[1], ux.shape[2], 27))
    for z0 in range(0, nZ, z_chunk):
        z1 = min(nZ, z0 + z_chunk)
        out[z0:z1] = qlbm.compute(ux[z0:z1], uy[z0:z1], uz[z0:z1], T[z0:z1])
    return out
```

## Tuning tips

- `round_digits`: reduce to increase cache hit rate; raise for higher resolution.
- Use `backend='cupy'` and feed CuPy arrays to avoid host-device transfers.
- `precompile_grid` helps avoid repeated transpilation overhead.
- Monitor `qlbm.get_cache_statistics()` to tune `cache_size` and `round_digits`.

## Tests to add

- Shape/consistency test: verify `compute` returns `(nZ,nY,nX,27)` and last-axis sums ~1.
- Cache test: repeated `compute` calls should increase cache hits.
- Precompile test: `precompile_grid` + `run_precompiled_grid` match `compute` results.

## Troubleshooting

- If `cupy` import fails, use `backend='numpy'`.
- If `run_precompiled_grid` fails, call `precompile_grid` again including new parameters.
- If compute is slow for many unique params, increase `round_digits` or use chunking.

## Adding a convenience method

If desired, a convenience helper `compute_step(ux,uy,uz,T,rho=None,return_feq=True)` can be
added to `QuantumLBMInterface` to auto-convert backends and optionally multiply by `rho`.

---

This file was added to `docs/` to avoid modifying the repository README.
