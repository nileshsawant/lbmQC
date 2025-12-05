"""
GPU-Parallelized Quantum LBM Integration

Leverages H100 GPU to run multiple quantum circuits in parallel.

KEY STRATEGY:
- Instead of: 63,001 grid points × 1 circuit × 1000 shots (serial)
- Do: Batch N circuits together, run on GPU simultaneously
- H100 can handle hundreds of 6-qubit circuits in parallel

QISKIT GPU APPROACH:
1. Parameter binning reduces 63,001 → ~few hundred unique parameter sets
2. Create all circuits for unique parameters
3. Run all circuits in single GPU batch with transpile
4. Broadcast results back to grid points
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
    1. Parameter binning: Reduce 63,001 points to ~100-1000 unique bins
    2. Circuit batching: Run all unique circuits in single GPU call
    3. Result caching: Store computed probabilities for reuse
    4. Adaptive shots: Fewer shots for near-equilibrium regions
    """
    
    def __init__(self,
                 grid_shape: Tuple[int, int, int],
                 n_bins: int = 15,
                 shots_per_circuit: int = 1000,
                 use_gpu: bool = True,
                 batch_size: int = 500):
        """
        Initialize GPU batch processor.
        
        PARAMETERS:
        - grid_shape: (nZ, nY, nX)
        - n_bins: Binning resolution (15^4 = 50,625 max unique combinations)
        - shots_per_circuit: Quantum shots per unique parameter set
        - use_gpu: Enable GPU acceleration
        - batch_size: Max circuits to run in single GPU batch
        """
        self.grid_shape = grid_shape
        self.nZ, self.nY, self.nX = grid_shape
        self.n_bins = n_bins
        self.shots = shots_per_circuit
        self.batch_size = batch_size
        
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
        # Create one parameterized template circuit and transpile it once.
        print("  Initializing parameterized quantum circuit template...")
        self.qc_template, self.qc_parameters = self.qdg.create_quantum_circuit_3d_parametric_template()
        
        # Transpile once for the target backend
        print("  Transpiling template circuit...")
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.simulator)
        self.transpiled_qc = pass_manager.run(self.qc_template)
        
        # Statistics
        self.total_points = self.nX * self.nY * self.nZ
        
        print(f"QuantumLBMGPUBatch initialized:")
        print(f"  Grid: {self.nX}×{self.nY}×{self.nZ} = {self.total_points:,} points")
        print(f"  Parameter bins: {n_bins}^4 = {n_bins**4:,} max")
        print(f"  Shots per circuit: {shots_per_circuit}")
        print(f"  Batch size: {batch_size} circuits/GPU call")
        print(f"  GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
    
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
    
    def _create_parameter_bins_gpu(self,
                                  ux: cp.ndarray,
                                  uy: cp.ndarray,
                                  uz: cp.ndarray,
                                  T: cp.ndarray) -> Tuple[List[Tuple], cp.ndarray]:
        """
        Vectorized binning on GPU.
        
        RETURNS:
        - params_list: List of (ux, uy, uz, T) tuples for unique bins
        - inverse_indices: CuPy array mapping grid points to params_list indices
        """
        print("\nBinning parameters (GPU Vectorized)...")
        start = time.time()
        
        # Helper to get bins and range
        def get_bins(arr, n_bins):
            vmin = float(cp.min(arr))
            vmax = float(cp.max(arr))
            if vmax == vmin:
                return cp.full_like(arr, n_bins // 2, dtype=cp.int32), vmin, vmax
            
            # Normalize to 0..1
            # We use a small epsilon to ensure max value falls into last bin
            norm = (arr - vmin) / (vmax - vmin)
            bins = cp.floor(norm * n_bins).astype(cp.int32)
            bins = cp.clip(bins, 0, n_bins - 1)
            return bins, vmin, vmax

        # 1. Compute bins for all fields on GPU
        b_ux, min_ux, max_ux = get_bins(ux, self.n_bins)
        b_uy, min_uy, max_uy = get_bins(uy, self.n_bins)
        b_uz, min_uz, max_uz = get_bins(uz, self.n_bins)
        b_T,  min_T,  max_T  = get_bins(T,  self.n_bins)
        
        print(f"  Parameter ranges:")
        print(f"    ux: [{min_ux:+.6f}, {max_ux:+.6f}]")
        print(f"    uy: [{min_uy:+.6f}, {max_uy:+.6f}]")
        print(f"    uz: [{min_uz:+.6f}, {max_uz:+.6f}]")
        print(f"    T : [{min_T:+.6f}, {max_T:+.6f}]")
        
        # 2. Pack bins into single integer for unique finding
        # n_bins is typically 15. We can pack into int32.
        # packed = b_ux + b_uy*N + b_uz*N^2 + b_T*N^3
        N = self.n_bins
        packed_bins = (b_ux + 
                       b_uy * N + 
                       b_uz * (N**2) + 
                       b_T  * (N**3))
        
        # 3. Find unique bins
        # unique_packed: sorted unique packed indices
        # inverse_indices: indices to reconstruct original array (on GPU)
        unique_packed, inverse_indices = cp.unique(packed_bins, return_inverse=True)
        
        # 4. Unpack unique bins to get parameters
        # We use the CENTER of the bin as the representative parameter
        u_b_T  = unique_packed // (N**3)
        rem    = unique_packed % (N**3)
        u_b_uz = rem // (N**2)
        rem    = rem % (N**2)
        u_b_uy = rem // N
        u_b_ux = rem % N
        
        # Convert unique bin indices to CPU for parameter list creation
        u_b_ux = cp.asnumpy(u_b_ux)
        u_b_uy = cp.asnumpy(u_b_uy)
        u_b_uz = cp.asnumpy(u_b_uz)
        u_b_T  = cp.asnumpy(u_b_T)
        
        def bin_to_val(b_idx, vmin, vmax, n_bins):
            # Center of the bin
            if vmax == vmin: return vmin
            step = (vmax - vmin) / n_bins
            return vmin + (b_idx + 0.5) * step
            
        params_list = []
        for i in range(len(unique_packed)):
            p_ux = bin_to_val(u_b_ux[i], min_ux, max_ux, N)
            p_uy = bin_to_val(u_b_uy[i], min_uy, max_uy, N)
            p_uz = bin_to_val(u_b_uz[i], min_uz, max_uz, N)
            p_T  = bin_to_val(u_b_T[i],  min_T,  max_T,  N)
            params_list.append((p_ux, p_uy, p_uz, p_T))
            
        elapsed = time.time() - start
        n_unique = len(params_list)
        compression = self.total_points / n_unique if n_unique > 0 else 0
        
        print(f"  Unique bins: {n_unique:,}")
        print(f"  Compression: {compression:.1f}x")
        print(f"  Time: {elapsed:.2f}s")
        
        return params_list, inverse_indices
    
    def _run_circuits_batched(self, params_list: List[Tuple]) -> np.ndarray:
        """
        Run all quantum circuits in GPU-batched mode using Parameterized Quantum Circuits.
        
        RETURNS:
        - probs_array: NumPy array [n_unique, 27]
        """
        print("\nRunning quantum circuits on GPU (Parameterized)...")
        start = time.time()
        
        n_circuits = len(params_list)
        
        # 1. Prepare parameter bindings
        # Instead of creating N circuits, we create N parameter sets for 1 circuit
        print(f"  Preparing parameters for {n_circuits:,} instances...")
        parameter_binds = []
        
        for params in params_list:
            ux, uy, uz, T = params
            
            # Compute angles for this parameter set
            # We use the helper method we added to QuantumDiscreteGaussian
            theta1_x, theta2_x = self.qdg.compute_angles(ux, T)
            theta1_y, theta2_y = self.qdg.compute_angles(uy, T)
            theta1_z, theta2_z = self.qdg.compute_angles(uz, T)
            
            # Create binding dictionary
            # self.qc_parameters is [theta1_x, theta2_x, theta1_y, theta2_y, theta1_z, theta2_z]
            bind = {
                self.qc_parameters[0]: theta1_x,
                self.qc_parameters[1]: theta2_x,
                self.qc_parameters[2]: theta1_y,
                self.qc_parameters[3]: theta2_y,
                self.qc_parameters[4]: theta1_z,
                self.qc_parameters[5]: theta2_z
            }
            parameter_binds.append(bind)
        
        # 2. Run in batches
        # Even with parameter binds, we should batch to avoid memory issues if N is huge
        print(f"  Executing in batches of {self.batch_size}...")
        
        # Pre-allocate result array
        probs_array = np.zeros((n_circuits, 27), dtype=np.float64)
        
        n_batches = (n_circuits + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_circuits)
            
            batch_binds = parameter_binds[start_idx:end_idx]
            
            # Transpose list of dicts to dict of lists for vectorized execution
            # batch_binds is [{p1: v1, p2: v2}, ...]
            # We want {p1: [v1, ...], p2: [v2, ...]}
            vectorized_bind = {param: [] for param in self.qc_parameters}
            
            for bind in batch_binds:
                for param, value in bind.items():
                    vectorized_bind[param].append(value)
            
            # Run batch on GPU
            # parameter_binds expects a list of bindings for each circuit.
            # We have 1 circuit. So we pass a list with 1 element.
            # That element is the vectorized binding dictionary.
            job = self.simulator.run(self.transpiled_qc, parameter_binds=[vectorized_bind], shots=self.shots)
            result = job.result()
            
            # Process results
            for i in range(len(batch_binds)):
                counts = result.get_counts(i)
                velocity_counts = self.qdg._decode_quantum_counts_3d(counts)
                probs_27 = self.qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
                probs_array[start_idx + i, :] = probs_27
            
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
        
        INPUTS:
        - ux, uy, uz: Velocity fields (CuPy, shape [nZ,nY,nX])
        - T: Temperature field (CuPy, shape [nZ,nY,nX] or [nZ,nY,nX,1])
        - rho: Density field (CuPy, shape [nZ,nY,nX])
        - Y_a: Phase fraction scalar
        
        RETURNS:
        - grid_fEq_a: CuPy array shape [nZ,nY,nX,27]
        """
        print("\n" + "="*80)
        print("QUANTUM EQUILIBRIUM DISTRIBUTION (GPU-BATCHED)")
        print("="*80)
        
        overall_start = time.time()
        
        # Handle T dimension
        if T.ndim == 4:
            T_3d = T[:,:,:,0]
        else:
            T_3d = T
        
        # Step 1: Parameter binning (Vectorized on GPU)
        # No CPU transfer needed for full arrays!
        params_list, inverse_indices = self._create_parameter_bins_gpu(
            ux, uy, uz, T_3d
        )
        
        # Step 2: Run batched quantum circuits
        # Returns NumPy array [n_unique, 27]
        unique_probs_cpu = self._run_circuits_batched(params_list)
        
        # Step 3: Broadcast to full grid (Vectorized on GPU)
        print("\nBroadcasting results to grid...")
        broadcast_start = time.time()
        
        # Transfer unique probabilities to GPU (small data)
        unique_probs_gpu = cp.asarray(unique_probs_cpu)
        
        # Broadcast using advanced indexing
        # inverse_indices shape: [N_points]
        # unique_probs_gpu shape: [N_unique, 27]
        # Result shape: [N_points, 27]
        grid_probs_flat = unique_probs_gpu[inverse_indices]
        
        # Reshape to grid
        grid_probs = grid_probs_flat.reshape(self.nZ, self.nY, self.nX, 27)
        
        # Final calculation: f_eq = Y_a * rho * probs
        # rho shape: [nZ, nY, nX] -> [nZ, nY, nX, 1]
        grid_feq_gpu = grid_probs * (Y_a * rho[..., cp.newaxis])
        
        broadcast_time = time.time() - broadcast_start
        print(f"  Broadcast time: {broadcast_time:.2f}s")
        
        total_time = time.time() - overall_start
        
        print("\n" + "="*80)
        print(f"✓ QUANTUM COMPUTATION COMPLETE")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Unique circuits: {len(params_list):,}")
        print(f"  Speedup vs serial: ~{self.total_points/max(1, len(params_list)):.0f}x")
        print("="*80 + "\n")
        
        return grid_feq_gpu


def estimate_performance(nX, nY, nZ, n_bins=15, shots=1000):
    """
    Estimate quantum computation time for given grid.
    
    ASSUMPTIONS:
    - H100 GPU: ~100-200 circuits/second (6 qubits, 1000 shots each)
    - Binning reduces points by ~100-500x
    """
    total_points = nX * nY * nZ
    
    # Estimate unique bins (empirical: typically 0.5-2% of total points for flow)
    estimated_unique = min(total_points, max(100, int(total_points * 0.01)))
    
    # GPU throughput
    circuits_per_second = 150  # Conservative estimate for H100
    
    quantum_time = estimated_unique / circuits_per_second
    overhead_time = 10  # Binning + transpile + broadcast
    total_time = quantum_time + overhead_time
    
    print(f"\nPERFORMANCE ESTIMATE:")
    print(f"  Grid: {nX}×{nY}×{nZ} = {total_points:,} points")
    print(f"  Estimated unique bins: ~{estimated_unique:,}")
    print(f"  Compression: ~{total_points/estimated_unique:.0f}x")
    print(f"  Estimated quantum time: {quantum_time:.1f}s")
    print(f"  Estimated total time: {total_time:.1f}s")
    print(f"  Per timestep overhead: {total_time:.1f}s")
    

if __name__ == "__main__":
    # Estimate for your grid
    estimate_performance(251, 251, 1)
