"""
GPU-Parallelized Quantum LBM Integration

Leverages H100 GPU to run multiple quantum circuits in parallel.

KEY STRATEGY:
- Component-wise decomposition: P(vx, vy, vz) = P(vx) * P(vy) * P(vz)
- 2D Binning: Bin (u, T) pairs instead of (ux, uy, uz, T) tuples
- 1D Quantum Circuits: Run 2-qubit circuits for 1D distributions
- Reconstruction: Combine 1D results to form 3D distribution

ADVANTAGES:
- Massive reduction in unique circuits (from ~N to ~sqrt(N))
- Isotropic quantization (same bins for x, y, z)
- Higher efficiency allows more shots/bins
"""

import numpy as np
import cupy as cp
from typing import Dict, Tuple, List
from collections import defaultdict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from quantum_discrete_gaussian import QuantumDiscreteGaussian
import time


class QuantumLBMGPUBatch:
    """
    GPU-accelerated batch quantum circuit execution for LBM.
    
    OPTIMIZATIONS:
    1. Component-wise decomposition: Decompose 3D problem into 1D problems
    2. Parameter binning: Bin (u, T) pairs for high efficiency
    3. Circuit batching: Run all unique 1D circuits in single GPU call
    4. Result reconstruction: Reconstruct 3D distribution on GPU
    """
    
    def __init__(self,
                 grid_shape: Tuple[int, int, int],
                 n_bins: int = 401,
                 shots_per_circuit: int = 1000,
                 use_gpu: bool = True,
                 batch_size: int = 500,
                 enable_binning: bool = True):
        """
        Initialize GPU batch processor.
        
        PARAMETERS:
        - grid_shape: (nZ, nY, nX)
        - n_bins: Binning resolution (odd number recommended for symmetry)
        - shots_per_circuit: Quantum shots per unique parameter set
        - use_gpu: Enable GPU acceleration
        - batch_size: Max circuits to run in single GPU batch
        - enable_binning: If False, runs a unique circuit for every grid point (very slow)
        """
        self.grid_shape = grid_shape
        self.nZ, self.nY, self.nX = grid_shape
        self.n_bins = n_bins
        self.shots = shots_per_circuit
        self.batch_size = batch_size
        self.enable_binning = enable_binning
        
        # Initialize quantum discrete Gaussian
        self.qdg = QuantumDiscreteGaussian(
            grid_size=max(grid_shape),
            circuit_type='symmetric',
            grid_3d=grid_shape
        )
        
        # Setup GPU simulator
        self.use_gpu = use_gpu
        self._setup_gpu_simulator()
        
        # PRE-COMPILATION STRATEGY:
        # Create one parameterized template circuit (1D) and transpile it once.
        print("  Initializing parameterized quantum circuit template (1D)...")
        self.qc_template, self.qc_parameters = self.qdg.create_quantum_circuit_1d_parametric_template()
        
        # Transpile once for the target backend
        print("  Transpiling template circuit...")
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.simulator)
        self.transpiled_qc = pass_manager.run(self.qc_template)
        
        # Statistics
        self.total_points = self.nX * self.nY * self.nZ
        
        print(f"QuantumLBMGPUBatch initialized:")
        print(f"  Grid: {self.nX}×{self.nY}×{self.nZ} = {self.total_points:,} points")
        print(f"  Parameter bins: {n_bins}^2 (2D decomposition)")
        print(f"  Shots per circuit: {shots_per_circuit}")
        print(f"  Batch size: {batch_size} circuits/GPU call")
        print(f"  GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"  Binning: {'Enabled' if self.enable_binning else 'Disabled'}")
    
    def _setup_gpu_simulator(self):
        """Setup Qiskit GPU simulator."""
        if self.use_gpu:
            try:
                # Try GPU backend
                self.simulator = AerSimulator(method='statevector', device='GPU')
                print("  Qiskit GPU: Available ✓")
            except Exception as e:
                print(f"  Qiskit GPU: Failed ({e})")
                print("  Falling back to CPU")
                self.simulator = AerSimulator(method='statevector')
                self.use_gpu = False
        else:
            self.simulator = AerSimulator(method='statevector')
    
    def _create_component_bins_gpu(self,
                                  ux: cp.ndarray,
                                  uy: cp.ndarray,
                                  uz: cp.ndarray,
                                  T: cp.ndarray) -> Tuple[List[Tuple], cp.ndarray]:
        """
        Vectorized component-wise binning on GPU.
        
        STRATEGY:
        1. Flatten ux, uy, uz into a single 'u' array (3 * N_points)
        2. Repeat T array 3 times (3 * N_points)
        3. Bin (u, T) pairs
        
        RETURNS:
        - params_list: List of (u, T) tuples for unique bins
        - inverse_indices: CuPy array mapping (3*N) points to params_list indices
        """
        print("\nBinning parameters (Component-wise 2D)...")
        start = time.time()
        
        # 1. Flatten and concatenate
        # ux, uy, uz are (Nz, Ny, Nx)
        # We want a single long array of all velocities
        u_all = cp.concatenate([ux.ravel(), uy.ravel(), uz.ravel()])
        
        # T is (Nz, Ny, Nx)
        # We repeat it 3 times to match u_all
        T_flat = T.ravel()
        T_all = cp.concatenate([T_flat, T_flat, T_flat])
        
        if not self.enable_binning:
            print("  Binning DISABLED: Using all points as unique circuits")
            # Convert to CPU list directly
            # This will be slow for large grids!
            u_cpu = cp.asnumpy(u_all)
            T_cpu = cp.asnumpy(T_all)
            params_list = list(zip(u_cpu, T_cpu))
            inverse_indices = cp.arange(len(params_list), dtype=cp.int32)
            return params_list, inverse_indices

        # Helper to get bins and range
        def get_bins(arr, n_bins, vmin=None, vmax=None):
            if vmin is None: vmin = float(cp.min(arr))
            if vmax is None: vmax = float(cp.max(arr))
            
            if vmax == vmin:
                return cp.full_like(arr, n_bins // 2, dtype=cp.int32), vmin, vmax
            
            # Normalize to 0..1
            norm = (arr - vmin) / (vmax - vmin)
            bins = cp.floor(norm * n_bins).astype(cp.int32)
            bins = cp.clip(bins, 0, n_bins - 1)
            return bins, vmin, vmax

        # 2. Compute bins
        # Use global min/max for u to ensure consistent quantization
        b_u, min_u, max_u = get_bins(u_all, self.n_bins)
        b_T, min_T, max_T = get_bins(T_all, self.n_bins)
        
        print(f"  Parameter ranges:")
        print(f"    u: [{min_u:+.6f}, {max_u:+.6f}]")
        print(f"    T: [{min_T:+.6f}, {max_T:+.6f}]")
        
        # 3. Pack bins into single integer
        # packed = b_u + b_T * N
        N = self.n_bins
        packed_bins = b_u + b_T * N
        
        # 4. Find unique bins
        unique_packed, inverse_indices = cp.unique(packed_bins, return_inverse=True)
        
        # 5. Unpack unique bins
        u_b_T = unique_packed // N
        u_b_u = unique_packed % N
        
        # Convert to CPU
        u_b_u = cp.asnumpy(u_b_u)
        u_b_T = cp.asnumpy(u_b_T)
        
        def bin_to_val(b_idx, vmin, vmax, n_bins):
            if vmax == vmin: return vmin
            step = (vmax - vmin) / n_bins
            return vmin + (b_idx + 0.5) * step
            
        params_list = []
        for i in range(len(unique_packed)):
            p_u = bin_to_val(u_b_u[i], min_u, max_u, N)
            p_T = bin_to_val(u_b_T[i], min_T, max_T, N)
            params_list.append((p_u, p_T))
            
        elapsed = time.time() - start
        n_unique = len(params_list)
        total_components = u_all.size
        compression = total_components / n_unique if n_unique > 0 else 0
        
        print(f"  Unique bins: {n_unique:,}")
        print(f"  Compression: {compression:.1f}x")
        print(f"  Time: {elapsed:.2f}s")
        
        return params_list, inverse_indices
    
    def _run_circuits_batched_1d(self, params_list: List[Tuple]) -> np.ndarray:
        """
        Run 1D quantum circuits in GPU-batched mode.
        
        RETURNS:
        - probs_array: NumPy array [n_unique, 3] (P(-1), P(0), P(1))
        """
        print("\nRunning 1D quantum circuits on GPU...")
        start = time.time()
        
        n_circuits = len(params_list)
        
        # 1. Prepare parameter bindings
        print(f"  Preparing parameters for {n_circuits:,} instances...")
        parameter_binds = []
        
        for params in params_list:
            u, T = params
            
            # Compute angles for 1D circuit
            theta1, theta2 = self.qdg.compute_angles(u, T)
            
            # Create binding dictionary
            # self.qc_parameters is [theta1, theta2]
            bind = {
                self.qc_parameters[0]: theta1,
                self.qc_parameters[1]: theta2
            }
            parameter_binds.append(bind)
        
        # 2. Run in batches
        print(f"  Executing in batches of {self.batch_size}...")
        
        # Pre-allocate result array (3 outcomes: -1, 0, 1)
        probs_array = np.zeros((n_circuits, 3), dtype=np.float64)
        
        n_batches = (n_circuits + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_circuits)
            
            batch_binds = parameter_binds[start_idx:end_idx]
            
            # Vectorize bindings
            vectorized_bind = {param: [] for param in self.qc_parameters}
            for bind in batch_binds:
                for param, value in bind.items():
                    vectorized_bind[param].append(value)
            
            # Run batch
            job = self.simulator.run(self.transpiled_qc, parameter_binds=[vectorized_bind], shots=self.shots)
            result = job.result()
            
            # Process results
            for i in range(len(batch_binds)):
                counts = result.get_counts(i)
                # Decode 2-qubit measurement to {-1, 0, 1}
                # Using symmetric decoder
                outcome_counts = self.qdg._decode_quantum_counts_symmetric(counts)
                
                total = sum(outcome_counts.values())
                if total > 0:
                    probs_array[start_idx + i, 0] = outcome_counts.get(-1, 0) / total
                    probs_array[start_idx + i, 1] = outcome_counts.get(0, 0) / total
                    probs_array[start_idx + i, 2] = outcome_counts.get(1, 0) / total
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                progress = 100 * (batch_idx + 1) / n_batches
                elapsed_batch = time.time() - start
                rate = (end_idx) / elapsed_batch
                eta = (n_circuits - end_idx) / rate if rate > 0 else 0
                print(f"    Batch {batch_idx+1}/{n_batches} ({progress:.0f}%) | "
                      f"{rate:.1f} circuits/s | ETA: {eta:.1f}s")
        
        total_time = time.time() - start
        print(f"  Quantum execution complete: {total_time:.2f}s")
        print(f"  Throughput: {n_circuits/total_time:.1f} circuits/s")
        
        return probs_array
    
    def compute_quantum_feq(self,
                           ux: cp.ndarray,
                           uy: cp.ndarray,
                           uz: cp.ndarray,
                           T: cp.ndarray,
                           rho: cp.ndarray,
                           Y_a: float) -> cp.ndarray:
        """
        Main interface: Compute quantum equilibrium distribution.
        """
        print("\n" + "="*80)
        print("QUANTUM EQUILIBRIUM DISTRIBUTION (COMPONENT-WISE)")
        print("="*80)
        
        overall_start = time.time()
        
        # Handle T dimension
        if T.ndim == 4:
            T_3d = T[:,:,:,0]
        else:
            T_3d = T
        
        # Step 1: Component-wise Binning
        params_list, inverse_indices = self._create_component_bins_gpu(
            ux, uy, uz, T_3d
        )
        
        # Step 2: Run 1D circuits
        # Returns [n_unique, 3]
        unique_probs_1d_cpu = self._run_circuits_batched_1d(params_list)
        
        # Step 3: Reconstruct 3D distribution on GPU
        print("\nReconstructing 3D distribution on GPU...")
        reconstruct_start = time.time()
        
        # Transfer unique 1D probabilities to GPU
        unique_probs_1d_gpu = cp.asarray(unique_probs_1d_cpu)
        
        # Broadcast back to all components
        # inverse_indices maps to [ux_flat, uy_flat, uz_flat]
        all_probs_1d = unique_probs_1d_gpu[inverse_indices] # Shape: [3*N_points, 3]
        
        # Split into x, y, z components
        n_points = self.total_points
        probs_x = all_probs_1d[0:n_points]          # [N_points, 3]
        probs_y = all_probs_1d[n_points:2*n_points] # [N_points, 3]
        probs_z = all_probs_1d[2*n_points:]         # [N_points, 3]
        
        # Get D3Q27 lattice vectors
        # We need to construct the 27 probabilities
        # P_i = P_x(cx_i) * P_y(cy_i) * P_z(cz_i)
        
        # Get lattice vectors (on CPU first)
        vX, vY, vZ = self.qdg.get_d3q27_velocity_ordering()
        
        # Map velocity values {-1, 0, 1} to indices {0, 1, 2}
        # P(-1) is at index 0, P(0) at 1, P(1) at 2
        # Map: -1->0, 0->1, 1->2  => index = val + 1
        
        # Create indices for all 27 directions
        idx_x = cp.asarray(vX + 1)
        idx_y = cp.asarray(vY + 1)
        idx_z = cp.asarray(vZ + 1)
        
        # We need to compute P_i for each of the 27 directions for all grid points
        # Result shape: [N_points, 27]
        
        # Use advanced indexing to gather probabilities
        # probs_x is [N_points, 3]
        # We want [N_points, 27] where col i uses probs_x[:, idx_x[i]]
        
        # Gather probabilities for all 27 directions
        # This is a bit tricky to vectorize efficiently without a loop over 27
        # But 27 is small, so a loop is fine
        
        grid_probs = cp.zeros((n_points, 27), dtype=cp.float64)
        
        for i in range(27):
            # P_x component for direction i
            px = probs_x[:, idx_x[i]]
            py = probs_y[:, idx_y[i]]
            pz = probs_z[:, idx_z[i]]
            
            grid_probs[:, i] = px * py * pz
            
        # Reshape to grid [nZ, nY, nX, 27]
        grid_probs = grid_probs.reshape(self.nZ, self.nY, self.nX, 27)
        
        # Final calculation: f_eq = Y_a * rho * probs
        grid_feq_gpu = grid_probs * (Y_a * rho[..., cp.newaxis])
        
        reconstruct_time = time.time() - reconstruct_start
        print(f"  Reconstruction time: {reconstruct_time:.2f}s")
        
        total_time = time.time() - overall_start
        
        print("\n" + "="*80)
        print(f"✓ QUANTUM COMPUTATION COMPLETE")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Unique 1D circuits: {len(params_list):,}")
        print(f"  Speedup vs serial: ~{self.total_points*3/max(1, len(params_list)):.0f}x")
        print("="*80 + "\n")
        
        return grid_feq_gpu
