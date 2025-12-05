"""
Interface for Quantum LBM GPU Batch Processing.

This module provides a simple interface to the QuantumLBMGPUBatch class,
allowing it to be easily used from the LBM notebook.
"""

import cupy as cp
import numpy as np
from quantum_lbm_gpu_batch import QuantumLBMGPUBatch

# Global instance to persist across calls
_BATCH_PROCESSOR = None

def quantumEqDistribution(ux, uy, uz, T, shots=1000, n_bins=200, batch_size=1000):
    """
    Compute quantum equilibrium distribution probabilities.
    
    Args:
        ux, uy, uz: Velocity components (CuPy arrays)
        T: Temperature (CuPy array)
        shots: Number of quantum shots (optional, passed to init if new)
        n_bins: Number of bins for parameter quantization (default: 100)
        batch_size: Number of circuits to run in parallel on GPU (default: 500)
        
    Returns:
        probs: Probability distribution (CuPy array, shape [Nz, Ny, Nx, 27])
    """
    global _BATCH_PROCESSOR
    
    # Get grid dimensions
    # Handle T shape: (Nz, Ny, Nx, 1) or (Nz, Ny, Nx)
    if T.ndim == 4:
        nz, ny, nx, _ = T.shape
    else:
        nz, ny, nx = T.shape
        
    current_shape = (nz, ny, nx)
    
    # Initialize or re-initialize if shape or parameters change
    # We check if processor exists, and if grid shape, shots, or n_bins match
    needs_init = (_BATCH_PROCESSOR is None)
    
    if not needs_init:
        if _BATCH_PROCESSOR.grid_shape != current_shape:
            print(f"Re-initializing: Grid shape changed {_BATCH_PROCESSOR.grid_shape} -> {current_shape}")
            needs_init = True
        elif _BATCH_PROCESSOR.shots != shots:
            print(f"Re-initializing: Shots changed {_BATCH_PROCESSOR.shots} -> {shots}")
            needs_init = True
        elif _BATCH_PROCESSOR.n_bins != n_bins:
            print(f"Re-initializing: n_bins changed {_BATCH_PROCESSOR.n_bins} -> {n_bins}")
            needs_init = True
        elif _BATCH_PROCESSOR.batch_size != batch_size:
            print(f"Re-initializing: batch_size changed {_BATCH_PROCESSOR.batch_size} -> {batch_size}")
            needs_init = True
            
    if needs_init:
        print(f"Initializing QuantumLBMGPUBatch for grid {nx}x{ny}x{nz}, shots={shots}, bins={n_bins}, batch={batch_size}")
        _BATCH_PROCESSOR = QuantumLBMGPUBatch(
            grid_shape=current_shape,
            n_bins=n_bins,
            shots_per_circuit=shots,
            use_gpu=True,
            batch_size=batch_size
        )
    
    # The notebook expects probabilities, but compute_quantum_feq calculates
    # f_eq = Y_a * rho * probs.
    # We pass rho=1 and Y_a=1 to get just the probabilities.
    
    # Create dummy rho on GPU (CuPy)
    rho_dummy = cp.ones(current_shape, dtype=cp.float64)
    Y_a_dummy = 1.0
    
    # Call the batch processor
    # compute_quantum_feq returns a CuPy array
    probs_gpu = _BATCH_PROCESSOR.compute_quantum_feq(
        ux, uy, uz, T, rho_dummy, Y_a_dummy
    )
    
    return probs_gpu
