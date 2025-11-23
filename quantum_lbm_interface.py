"""
GPU-Accelerated Quantum LBM Interface

Bridges quantum_discrete_gaussian.py with LBM simulation for computing
equilibrium distributions on H100 GPU.

KEY FEATURES:
- Batch quantum circuit execution for multiple grid points
- GPU-accelerated Qiskit operations
- Adaptive sampling based on local flow conditions
- Caching frequently-used velocity distributions
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from quantum_discrete_gaussian import QuantumDiscreteGaussian
import hashlib
import pickle


class QuantumLBMInterface:
    """
    Interface between quantum circuit sampling and LBM simulation.
    
    STRATEGY FOR LARGE GRIDS:
    1. Batch processing: Group similar (ux,uy,uz,T) parameters
    2. Adaptive shots: Fewer shots for low-velocity regions
    3. Caching: Store frequently-used distributions
    4. GPU backend: Use Qiskit GPU simulator if available
    """
    
    def __init__(self, 
                 grid_shape: Tuple[int, int, int],
                 use_gpu: bool = True,
                 backend: str = 'cupy',
                 cache_size: int = 10000,
                 min_shots: int = 100,
                 max_shots: int = 5000,
                 round_digits: Optional[int] = 4):
        """
        Initialize quantum-LBM interface.
        
        PARAMETERS:
        - grid_shape: (nZ, nY, nX) grid dimensions
        - use_gpu: Enable GPU acceleration for Qiskit
        - backend: Array library to use ('cupy' or 'numpy')
        - cache_size: Maximum cached distributions
        - min_shots: Minimum quantum samples (low-velocity regions)
        - max_shots: Maximum quantum samples (high-velocity regions)
        """
        self.grid_shape = grid_shape
        self.nZ, self.nY, self.nX = grid_shape
        self.use_gpu = use_gpu
        self.min_shots = min_shots
        self.max_shots = max_shots
        
        # Setup array backend based on user choice and availability
        self.backend = backend
        if self.backend == 'cupy':
            try:
                import cupy as cp
                self.xp = cp
                self._asnumpy = self.xp.asnumpy
                self._asarray = self.xp.asarray
                print("Array backend: CuPy")
            except ImportError:
                raise ImportError("CuPy backend was requested, but the 'cupy' package is not installed.")
        else:
            self.xp = np
            self._asnumpy = lambda x: x  # Identity function if already numpy
            self._asarray = self.xp.asarray
            print("Array backend: NumPy")

        # Initialize quantum discrete Gaussian
        self.qdg = QuantumDiscreteGaussian(
            grid_size=max(grid_shape), 
            circuit_type='symmetric',
            grid_3d=grid_shape
        )
        
        # Distribution cache: hash(ux,uy,uz,T) -> 27 probabilities
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        # Cache for transpiled/compiled circuits to avoid re-transpilation
        # cache key -> transpiled QuantumCircuit
        self.transpiled_cache = {}
        # Number of decimal digits to round parameters for hashing/quantization.
        # If None, use full double precision formatting (float-level precision).
        if round_digits is None:
            self.round_digits = None
        else:
            self.round_digits = int(round_digits)
        
        # Setup GPU-accelerated simulator and pass manager
        self._setup_simulator_and_pass_manager()
        
        print(f"QuantumLBMInterface initialized:")
        print(f"  Grid: {self.nX}×{self.nY}×{self.nZ} = {self.nX*self.nY*self.nZ} points")
        print(f"  GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"  Cache size: {cache_size}")
        print(f"  Shots range: [{min_shots}, {max_shots}]")
    
    def _setup_simulator_and_pass_manager(self):
        """Setup Qiskit simulator and pass manager with GPU support if available."""
        try:
            # Try to use GPU backend
            if self.use_gpu:
                # Prefer default GPU device selection (let Aer decide method)
                self.simulator = AerSimulator(device='GPU')
                print("  Qiskit GPU backend: Available")
            else:
                self.simulator = AerSimulator()
                print("  Qiskit GPU backend: Disabled (using CPU)")
        except Exception as e:
            print(f"  Qiskit GPU backend: Not available ({e})")
            print("  Falling back to CPU simulator")
            self.simulator = AerSimulator()
            self.use_gpu = False
        
        # Create a single pass manager for transpilation
        self.pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.simulator)

    def _hash_parameters(self, ux: float, uy: float, uz: float, T: float) -> str:
        """Create hash key for caching. Round to 4 decimals for grouping."""
        # Apply quantization/rounding to group similar parameters together.
        if self.round_digits is None:
            # Use full double precision significant digits (approx 17) to preserve float-level uniqueness
            fmt = "{:.17g}"
        else:
            fmt = f"{{:.{self.round_digits}f}}"
        key = f"{fmt.format(ux)}_{fmt.format(uy)}_{fmt.format(uz)}_{fmt.format(T)}"
        return key

    def unique_parameter_stats(self, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, T: np.ndarray) -> Dict:
        """
        Return statistics on unique parameter keys for a given grid.

        Useful to estimate how many unique circuits will be produced and to
        decide on a `round_digits` setting before running a full workload.
        """
        T_3d = T[:,:,:,0] if T.ndim == 4 else T
        ux_cpu = self._asnumpy(ux)
        uy_cpu = self._asnumpy(uy)
        uz_cpu = self._asnumpy(uz)
        T_cpu = self._asnumpy(T_3d)

        total = 0
        keys = []
        for iz in range(self.nZ):
            for iy in range(self.nY):
                for ix in range(self.nX):
                    ux_local = ux_cpu[iz, iy, ix]
                    uy_local = uy_cpu[iz, iy, ix]
                    uz_local = uz_cpu[iz, iy, ix]
                    T_local = T_cpu[iz, iy, ix]
                    keys.append(self._hash_parameters(ux_local, uy_local, uz_local, T_local))
                    total += 1

        uniq = set(keys)
        counts = len(uniq)
        return {
            'total_points': total,
            'unique_keys': counts,
            'fraction_unique': counts/total if total>0 else 0
        }

    def compute(self, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Simple single-call API for repeated use in an LBM timestep loop.

        This method performs the batched quantum computation and reuses any
        transpiled circuits in `self.transpiled_cache`. It's a convenience
        wrapper around `compute_grid_probabilities_quantum`.
        """
        return self.compute_grid_probabilities_quantum(ux, uy, uz, T)
    
    def _compute_adaptive_shots(self, ux: float, uy: float, uz: float) -> int:
        """
        Compute number of shots based on flow velocity.
        
        STRATEGY:
        - High velocity: More shots needed (sharper distribution)
        - Low velocity: Fewer shots sufficient (broader distribution)
        """
        velocity_magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        
        # Linear scaling between min and max shots
        # velocity 0.0 -> min_shots
        # velocity 0.5+ -> max_shots
        if velocity_magnitude < 0.01:
            return self.min_shots
        elif velocity_magnitude > 0.5:
            return self.max_shots
        else:
            fraction = velocity_magnitude / 0.5
            return int(self.min_shots + fraction * (self.max_shots - self.min_shots))
    
    def get_single_point_probabilities(self,
                                       ux: float,
                                       uy: float,
                                       uz: float,
                                       T: float,
                                       use_cache: bool = True) -> np.ndarray:
        """
        Get 27 probabilities for single grid point using quantum sampling.
        
        RETURNS:
        np.ndarray shape (27,): Probabilities in D3Q27 LBM ordering
        """
        # Check cache first
        if use_cache:
            cache_key = self._hash_parameters(ux, uy, uz, T)
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            self.cache_misses += 1
        
        # Determine number of shots
        shots = self._compute_adaptive_shots(ux, uy, uz)
        
        # Quantum sampling
        velocity_counts = self.qdg.quantum_sample_grid_point_3d_parametric(
            ux, uy, uz, T, shots=shots,
            simulator=self.simulator,
            pass_manager=self.pass_manager
        )
        
        # Convert to LBM ordering
        probs_27 = self.qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
        
        # Cache result
        if use_cache and len(self.cache) < self.cache_size:
            self.cache[cache_key] = probs_27
        
        return probs_27
    
    def compute_grid_probabilities_quantum(self,
                                 ux: np.ndarray,  # Can be CuPy or NumPy array
                                 uy: np.ndarray,
                                 uz: np.ndarray,
                                 T: np.ndarray) -> np.ndarray:
        """
        Compute quantum equilibrium probability distributions for the entire grid
        using efficient batch processing.
        
        THIS IS THE MAIN INTERFACE FUNCTION FOR YOUR LBM CODE.
        
        STRATEGY:
        1. Identify unique, uncached grid points.
        2. Generate a list of quantum circuits for these unique points.
        3. Transpile all circuits in a single batch.
        4. Execute all transpiled circuits in a single job on the simulator.
        5. Process the results, update the cache, and populate the final grid.
        """
        print("\n" + "="*70)
        print("QUANTUM PROBABILITY DISTRIBUTION COMPUTATION (BATCHED)")
        print("="*70)
        
        # Handle T dimension
        T_3d = T[:,:,:,0] if T.ndim == 4 else T
        
        # Transfer to CPU for parameter processing
        ux_cpu = self._asnumpy(ux)
        uy_cpu = self._asnumpy(uy)
        uz_cpu = self._asnumpy(uz)
        T_cpu = self._asnumpy(T_3d)
        
        grid_probabilities = np.zeros((self.nZ, self.nY, self.nX, 27))
        
        # --- BATCH PREPARATION ---
        circuits_by_shots = {} # Group circuits by shot count for batching
        
        print("\n1. Identifying unique computations and checking cache...")
        for iz in range(self.nZ):
            for iy in range(self.nY):
                for ix in range(self.nX):
                    ux_local, uy_local, uz_local, T_local = ux_cpu[iz, iy, ix], uy_cpu[iz, iy, ix], uz_cpu[iz, iy, ix], T_cpu[iz, iy, ix]
                    cache_key = self._hash_parameters(ux_local, uy_local, uz_local, T_local)
                    
                    if cache_key in self.cache:
                        grid_probabilities[iz, iy, ix, :] = self.cache[cache_key]
                        self.cache_hits += 1
                    else:
                        self.cache_misses += 1
                        shots = self._compute_adaptive_shots(ux_local, uy_local, uz_local)
                        
                        # Add circuit to the correct shot group
                        if shots not in circuits_by_shots:
                            circuits_by_shots[shots] = []
                        
                        # Avoid adding duplicate circuits within the same batch
                        is_duplicate_in_batch = any(meta['cache_key'] == cache_key for meta in circuits_by_shots[shots])
                        if not is_duplicate_in_batch:
                            # If we previously transpiled this circuit (from earlier timestep),
                            # reuse the transpiled circuit to avoid re-transpilation.
                            if cache_key in self.transpiled_cache:
                                tc = self.transpiled_cache[cache_key]
                                circuits_by_shots[shots].append({'circuit': tc, 'cache_key': cache_key, 'compiled': True})
                            else:
                                qc = self.qdg.create_quantum_circuit_3d_parametric(ux_local, uy_local, uz_local, T_local)
                                qc.name = cache_key
                                circuits_by_shots[shots].append({'circuit': qc, 'cache_key': cache_key, 'compiled': False})

        total_points = self.nX * self.nY * self.nZ
        unique_circuits_count = sum(len(group) for group in circuits_by_shots.values())
        print(f"   Found {total_points - unique_circuits_count} cached points.")
        print(f"   Need to run {unique_circuits_count} unique quantum circuits in {len(circuits_by_shots)} batches.")

        if unique_circuits_count > 0:
            # --- BATCH EXECUTION (per shot group) ---
            print("\n2. Transpiling and executing batches...")
            all_results = {} # Store results from all batches
            for shots, group in circuits_by_shots.items():
                # Separate already-transpiled circuits from raw circuits so we
                # only transpile new circuits once and store them for reuse.
                precompiled = [item for item in group if item.get('compiled', False)]
                raw = [item for item in group if not item.get('compiled', False)]

                # Build ordered lists for execution and mapping keys
                compiled_batch = []
                batch_keys = []

                # Add precompiled circuits (from cache)
                for item in precompiled:
                    compiled_batch.append(item['circuit'])
                    batch_keys.append(item['cache_key'])

                # Transpile raw circuits (only these) and cache the transpiled circuits
                if raw:
                    raw_circuits = [item['circuit'] for item in raw]
                    raw_keys = [item['cache_key'] for item in raw]
                    print(f"   - Transpiling {len(raw_circuits)} new circuits for {shots} shots...")
                    start_time = time.time()
                    transpiled_raw = self.pass_manager.run(raw_circuits)
                    transpile_time = time.time() - start_time
                    print(f"     ✓ Transpiled in {transpile_time:.2f}s")

                    # Cache transpiled circuits and add to batch
                    for key, tc in zip(raw_keys, transpiled_raw):
                        # Store transpiled circuit for reuse in later timesteps
                        self.transpiled_cache[key] = tc
                        compiled_batch.append(tc)
                        batch_keys.append(key)

                print(f"   - Running batch of {len(compiled_batch)} circuits with {shots} shots...")
                start_time = time.time()
                job = self.simulator.run(compiled_batch, shots=shots)
                result = job.result()

                # Map results back to cache keys (try index then name)
                for i, cache_key in enumerate(batch_keys):
                    counts = {}
                    try:
                        counts = result.get_counts(i)
                    except Exception:
                        try:
                            counts = result.get_counts(cache_key)
                        except Exception:
                            # Give up and use empty counts
                            counts = {}
                    all_results[cache_key] = counts

                print(f"     ✓ Batch finished in {time.time() - start_time:.2f}s")

            # --- PROCESS RESULTS ---
            print("\n3. Processing results and updating cache...")
            for cache_key, counts in all_results.items():
                velocity_counts = self.qdg._decode_quantum_counts_3d(counts)
                probs_27 = self.qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
                
                # Store in cache for future use
                if len(self.cache) < self.cache_size:
                    self.cache[cache_key] = probs_27
            
            # Now that the cache is populated, fill the grid for the points we just computed
            for iz in range(self.nZ):
                for iy in range(self.nY):
                    for ix in range(self.nX):
                        # Check if this point was part of the batch
                        if np.all(grid_probabilities[iz, iy, ix, :] == 0):
                            ux_local, uy_local, uz_local, T_local = ux_cpu[iz, iy, ix], uy_cpu[iz, iy, ix], uz_cpu[iz, iy, ix], T_cpu[iz, iy, ix]
                            cache_key = self._hash_parameters(ux_local, uy_local, uz_local, T_local)
                            if cache_key in self.cache:
                                grid_probabilities[iz, iy, ix, :] = self.cache[cache_key]

        print(f"\n✓ Quantum computation complete!")
        cache_efficiency = 100*self.cache_hits/(self.cache_hits+self.cache_misses) if (self.cache_hits+self.cache_misses) > 0 else 0
        print(f"  Cache efficiency: {cache_efficiency:.1f}%")
        print("="*70 + "\n")
        
        # Transfer back to the original backend
        return self._asarray(grid_probabilities)

    def precompile_grid(self, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, T: np.ndarray, use_cache: bool = True):
        """
        Precompile (transpile) all unique circuits for the provided grid and store them
        in `self.transpiled_cache`. This allows subsequent calls to execute quickly
        without re-transpilation.

        This will mimic the same grouping by adaptive shot count as `compute_grid_probabilities_quantum`,
        but only perform transpilation and caching.
        """
        # Handle T dimension
        T_3d = T[:,:,:,0] if T.ndim == 4 else T

        ux_cpu = self._asnumpy(ux)
        uy_cpu = self._asnumpy(uy)
        uz_cpu = self._asnumpy(uz)
        T_cpu = self._asnumpy(T_3d)

        circuits_to_transpile = []
        circuit_keys = []

        for iz in range(self.nZ):
            for iy in range(self.nY):
                for ix in range(self.nX):
                    ux_local = ux_cpu[iz, iy, ix]
                    uy_local = uy_cpu[iz, iy, ix]
                    uz_local = uz_cpu[iz, iy, ix]
                    T_local = T_cpu[iz, iy, ix]
                    cache_key = self._hash_parameters(ux_local, uy_local, uz_local, T_local)
                    if use_cache and cache_key in self.cache:
                        continue
                    if cache_key in self.transpiled_cache:
                        continue
                    qc = self.qdg.create_quantum_circuit_3d_parametric(ux_local, uy_local, uz_local, T_local)
                    qc.name = cache_key
                    circuits_to_transpile.append(qc)
                    circuit_keys.append(cache_key)

        if circuits_to_transpile:
            print(f"Precompiling {len(circuits_to_transpile)} unique circuits...")
            start = time.time()
            transpiled = self.pass_manager.run(circuits_to_transpile)
            dt = time.time() - start
            print(f"   Transpiled in {dt:.2f}s")
            for key, tc in zip(circuit_keys, transpiled):
                self.transpiled_cache[key] = tc
        else:
            print("No circuits to precompile (all cached/transpiled)")

    def run_precompiled_grid(self, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Execute precompiled/transpiled circuits for the grid. Assumes that
        `precompile_grid` has already been called and that `self.transpiled_cache`
        contains all necessary circuits. This performs only execution + result processing.
        """
        T_3d = T[:,:,:,0] if T.ndim == 4 else T
        ux_cpu = self._asnumpy(ux)
        uy_cpu = self._asnumpy(uy)
        uz_cpu = self._asnumpy(uz)
        T_cpu = self._asnumpy(T_3d)

        grid_probabilities = np.zeros((self.nZ, self.nY, self.nX, 27))

        # Group precompiled circuits by shots so they can be run efficiently.
        circuits_by_shots = {}
        for iz in range(self.nZ):
            for iy in range(self.nY):
                for ix in range(self.nX):
                    ux_local = ux_cpu[iz, iy, ix]
                    uy_local = uy_cpu[iz, iy, ix]
                    uz_local = uz_cpu[iz, iy, ix]
                    T_local = T_cpu[iz, iy, ix]
                    cache_key = self._hash_parameters(ux_local, uy_local, uz_local, T_local)
                    if cache_key in self.cache:
                        grid_probabilities[iz, iy, ix, :] = self.cache[cache_key]
                        continue
                    if cache_key not in self.transpiled_cache:
                        raise RuntimeError(f"Circuit for {cache_key} not precompiled. Call precompile_grid first.")

                    # use adaptive shots to determine grouping
                    shots = self._compute_adaptive_shots(ux_local, uy_local, uz_local)
                    if shots not in circuits_by_shots:
                        circuits_by_shots[shots] = []
                    circuits_by_shots[shots].append((cache_key, self.transpiled_cache[cache_key]))

        all_results = {}
        for shots, group in circuits_by_shots.items():
            batch = [tc for _, tc in group]
            keys = [k for k, _ in group]
            print(f"Running batch of {len(batch)} precompiled circuits with {shots} shots...")
            job = self.simulator.run(batch, shots=shots)
            result = job.result()
            for i, key in enumerate(keys):
                counts = {}
                try:
                    counts = result.get_counts(i)
                except Exception:
                    try:
                        counts = result.get_counts(key)
                    except Exception:
                        counts = {}
                all_results[key] = counts

        # Process results
        for cache_key, counts in all_results.items():
            velocity_counts = self.qdg._decode_quantum_counts_3d(counts)
            probs_27 = self.qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = probs_27

        # Fill grid from cache
        for iz in range(self.nZ):
            for iy in range(self.nY):
                for ix in range(self.nX):
                    ux_local = ux_cpu[iz, iy, ix]
                    uy_local = uy_cpu[iz, iy, ix]
                    uz_local = uz_cpu[iz, iy, ix]
                    T_local = T_cpu[iz, iy, ix]
                    cache_key = self._hash_parameters(ux_local, uy_local, uz_local, T_local)
                    grid_probabilities[iz, iy, ix, :] = self.cache.get(cache_key, np.zeros(27))

        return self._asarray(grid_probabilities)
    
    def compute_grid_probabilities_analytical(self,
                                    ux: np.ndarray,
                                    uy: np.ndarray,
                                    uz: np.ndarray,
                                    T: np.ndarray) -> np.ndarray:
        """
        Compute analytical equilibrium probability distributions (classical).
        
        Uses compute_3d_probability_distribution_lbm_order for exact calculation.
        This is for comparison and validation purposes.
        """
        print("\n" + "="*70)
        print("ANALYTICAL PROBABILITY DISTRIBUTION COMPUTATION")
        print("="*70)
        
        # Handle T dimension
        if T.ndim == 4:
            T_3d = T[:,:,:,0]
        else:
            T_3d = T
        
        # Transfer to CPU if needed
        ux_cpu = self._asnumpy(ux)
        uy_cpu = self._asnumpy(uy)
        uz_cpu = self._asnumpy(uz)
        T_cpu = self._asnumpy(T_3d)
        
        # Initialize output on CPU
        grid_probabilities = np.zeros((self.nZ, self.nY, self.nX, 27))
        
        total_points = self.nX * self.nY * self.nZ
        print(f"\nProcessing {total_points} grid points...")
        
        processed = 0
        for iz in range(self.nZ):
            for iy in range(self.nY):
                for ix in range(self.nX):
                    ux_local = ux_cpu[iz, iy, ix]
                    uy_local = uy_cpu[iz, iy, ix]
                    uz_local = uz_cpu[iz, iy, ix]
                    T_local = T_cpu[iz, iy, ix]
                    
                    # Analytical probabilities
                    probs_27 = self.qdg.compute_3d_probability_distribution_lbm_order(
                        ux_local, uy_local, uz_local, T_local
                    )
                    
                    grid_probabilities[iz, iy, ix, :] = probs_27
                    
                    processed += 1
                    if processed % 10000 == 0 or processed == total_points:
                        print(f"  Progress: {processed}/{total_points} ({100*processed/total_points:.1f}%)")
        
        print(f"\n✓ Analytical computation complete!")
        print("="*70 + "\n")
        
        # Transfer back to the original backend
        return self._asarray(grid_probabilities)
    
    def clear_cache(self):
        """Clear probability cache."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        print("Cache cleared.")
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = 100 * self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate_percent': hit_rate
        }

    def compute_step(self,
                     ux,
                     uy,
                     uz,
                     T,
                     rho=None,
                     return_feq: bool = False) -> np.ndarray:
        """
        Convenience wrapper for a single LBM timestep.

        This method will:
        - Convert inputs to the interface's backend (`self.xp`) if needed.
        - Call `self.compute(...)` to get probabilities (shape: nZ,nY,nX,27).
        - If `rho` is provided and `return_feq` is True, return `feq = rho[..., None] * probs`.

        Returns either `probs` or `feq` depending on `return_feq`.

        Notes:
        - This wrapper does not change caching or transpilation behavior; it simply
          makes calling code slightly cleaner and ensures backend consistency.
        - Input shapes must match `self.grid_shape`.
        """
        # Convert to configured backend arrays (no-op if already correct)
        try:
            ux_b = self._asarray(ux)
            uy_b = self._asarray(uy)
            uz_b = self._asarray(uz)
            T_b = self._asarray(T)
        except Exception:
            # If backend is cupy, ensure we use xp.asarray to convert
            ux_b = self.xp.asarray(ux)
            uy_b = self.xp.asarray(uy)
            uz_b = self.xp.asarray(uz)
            T_b = self.xp.asarray(T)

        probs = self.compute(ux_b, uy_b, uz_b, T_b)

        if return_feq:
            if rho is None:
                raise ValueError("rho must be provided when return_feq=True")
            rho_b = self.xp.asarray(rho)
            feq = rho_b[..., self.xp.newaxis] * probs
            return feq

        return probs


def demo_usage():
    """Demonstrate usage with small test case."""
    print("DEMO: QuantumLBMInterface Usage\n")
    
    # --- Backend Configuration ---
    # Change this to 'numpy' to test without a GPU
    backend_to_use = 'cupy' 
    
    # Small test grid
    nX, nY, nZ = 5, 5, 1
    
    # Create interface
    qlbm = QuantumLBMInterface(
        grid_shape=(nZ, nY, nX),
        use_gpu=True,
        backend=backend_to_use,
        cache_size=100,
        min_shots=100,
        max_shots=1000
    )
    
    xp = qlbm.xp # Use the backend-specific array module

    # Create test velocity/temperature fields
    ux = xp.random.uniform(-0.1, 0.1, (nZ, nY, nX))
    uy = xp.random.uniform(-0.1, 0.1, (nZ, nY, nX))
    uz = xp.random.uniform(-0.1, 0.1, (nZ, nY, nX))
    T = xp.ones((nZ, nY, nX, 1)) * 0.33
    
    # Compute quantum equilibrium probabilities
    print("\n1. Quantum computation:")
    grid_probs_quantum = qlbm.compute_grid_probabilities_quantum(ux, uy, uz, T)
    
    # Compute analytical for comparison
    print("\n2. Analytical computation:")
    grid_probs_analytical = qlbm.compute_grid_probabilities_analytical(ux, uy, uz, T)
    
    # In a real LBM code, you would now multiply by rho:
    # rho = xp.ones((nZ, nY, nX))
    # grid_feq_quantum = rho[..., xp.newaxis] * grid_probs_quantum

    # Compare
    print("\n3. Comparison:")
    diff = xp.abs(grid_probs_quantum - grid_probs_analytical)
    print(f"  Max difference: {xp.max(diff):.6f}")
    print(f"  Mean difference: {xp.mean(diff):.6f}")
    print(f"  RMS difference: {xp.sqrt(xp.mean(diff**2)):.6f}")
    
    # Cache statistics
    stats = qlbm.get_cache_statistics()
    print("\n4. Cache statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_usage()
