#!/usr/bin/env python3
"""
Benchmark CPU vs GPU performance for quantum circuit execution.
Tests the quantum_sample_grid_point_3d_parametric function with different backends.
"""

import time
import numpy as np
from quantum_discrete_gaussian import QuantumDiscreteGaussian

def benchmark_device(device: str, num_circuits: int = 20, shots: int = 1000):
    """Benchmark circuit execution on specified device."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {device} device")
    print(f"{'='*60}")
    
    # Initialize quantum sampler
    qvd = QuantumDiscreteGaussian()
    
    # Test parameters (typical LBM values)
    mu_x, mu_y, mu_z = 0.0, 0.0, 0.0  # Zero bulk velocity
    sigma_sq = 0.33  # Temperature
    
    print(f"Parameters: μ=({mu_x}, {mu_y}, {mu_z}), σ²={sigma_sq}")
    print(f"Circuits: {num_circuits}, Shots per circuit: {shots}")
    print(f"\nStarting benchmark...")
    
    # Warm-up run
    _ = qvd.quantum_sample_grid_point_3d_parametric(
        mu_x, mu_y, mu_z, sigma_sq, shots=100, device=device
    )
    
    # Timed runs
    start_time = time.time()
    for i in range(num_circuits):
        result = qvd.quantum_sample_grid_point_3d_parametric(
            mu_x, mu_y, mu_z, sigma_sq, shots=shots, device=device
        )
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{num_circuits} circuits, Rate: {rate:.2f} circuits/sec")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Results
    circuits_per_sec = num_circuits / elapsed_time
    time_per_circuit = elapsed_time / num_circuits
    
    print(f"\n{device} Results:")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Circuits/sec: {circuits_per_sec:.2f}")
    print(f"  Time/circuit: {time_per_circuit:.4f} seconds")
    print(f"  Total shots: {num_circuits * shots}")
    print(f"  Shots/sec: {num_circuits * shots / elapsed_time:.0f}")
    
    return {
        'device': device,
        'num_circuits': num_circuits,
        'shots': shots,
        'total_time': elapsed_time,
        'circuits_per_sec': circuits_per_sec,
        'time_per_circuit': time_per_circuit
    }

def main():
    print("Quantum Circuit Performance Benchmark")
    print(f"Testing quantum_sample_grid_point_3d_parametric with CPU and GPU backends")
    
    # Benchmark CPU
    cpu_results = benchmark_device('CPU', num_circuits=20, shots=1000)
    
    # Benchmark GPU
    gpu_results = benchmark_device('GPU', num_circuits=20, shots=1000)
    
    # Comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    speedup = gpu_results['circuits_per_sec'] / cpu_results['circuits_per_sec']
    print(f"CPU: {cpu_results['circuits_per_sec']:.2f} circuits/sec")
    print(f"GPU: {gpu_results['circuits_per_sec']:.2f} circuits/sec")
    print(f"Speedup: {speedup:.2f}x")
    
    # Grid calculation estimate
    grid_points = 251 * 251 * 1  # 63,001 points
    print(f"\n{'='*60}")
    print(f"FULL GRID ESTIMATE (251×251×1 = {grid_points:,} points)")
    print(f"{'='*60}")
    cpu_time = grid_points / cpu_results['circuits_per_sec']
    gpu_time = grid_points / gpu_results['circuits_per_sec']
    print(f"CPU time: {cpu_time:.1f} seconds ({cpu_time/60:.1f} minutes)")
    print(f"GPU time: {gpu_time:.1f} seconds ({gpu_time/60:.1f} minutes)")
    print(f"Time saved: {(cpu_time - gpu_time)/60:.1f} minutes")

if __name__ == '__main__':
    main()
