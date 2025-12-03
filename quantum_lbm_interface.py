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

def quantumEqDistribution(ux, uy, uz, T, shots=1000):
    """
    Compute quantum equilibrium distribution probabilities.
    
    Args:
        ux, uy, uz: Velocity components (CuPy arrays)
        T: Temperature (CuPy array)
        shots: Number of quantum shots (optional, passed to init if new)
        
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
    
    # Initialize or re-initialize if shape changes
    if _BATCH_PROCESSOR is None or _BATCH_PROCESSOR.grid_shape != current_shape:
        print(f"Initializing QuantumLBMGPUBatch for grid {nx}x{ny}x{nz}")
        _BATCH_PROCESSOR = QuantumLBMGPUBatch(
            grid_shape=current_shape,
            shots_per_circuit=shots,
            use_gpu=True
        )
    
    # The notebook expects probabilities, but compute_quantum_feq calculates
    # f_eq = Y_a * rho * probs.
    # We pass rho=1 and Y_a=1 to get just the probabilities.
    
    # Create dummy rho on GPU (CuPy)
    rho_dummy = cp.ones(current_shape, dtype=cp.float64)
    Y_a_dummy = 1.0
    
    # Call the batch processor
    # compute_quantum_feq returns a Host NumPy array
    probs_cpu = _BATCH_PROCESSOR.compute_quantum_feq(
        ux, uy, uz, T, rho_dummy, Y_a_dummy
    )
    
    # Convert back to CuPy for the notebook
    probs_gpu = cp.asarray(probs_cpu)
    
    return probs_gpu
