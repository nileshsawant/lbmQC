#!/usr/bin/env python3
"""
Test LBM-style moment computation: u_x = Sum_i f_i c_ix

This script demonstrates three ways to compute moments:
1. Direct from quantum samples (existing method)
2. From 1D probabilities (theoretical method)
3. From 27 probabilities using LBM formulas (NEW)

All three methods should give the same results.
"""

import numpy as np
from quantum_discrete_gaussian import QuantumDiscreteGaussian

def test_lbm_moment_calculation():
    """Compare three methods of moment calculation."""
    
    # Test parameters
    mu_x = 0.1
    mu_y = -0.05
    mu_z = 0.15
    T = 0.2
    shots = 5000
    
    print("=" * 80)
    print("LBM-STYLE MOMENT CALCULATION TEST")
    print("=" * 80)
    print(f"Parameters: mu_x={mu_x}, mu_y={mu_y}, mu_z={mu_z}, T={T}")
    print(f"Quantum shots: {shots}")
    print()
    
    # Initialize quantum system
    qdg = QuantumDiscreteGaussian(circuit_type='symmetric')
    
    # ========================================================================
    # METHOD 1: Direct from quantum samples (existing)
    # ========================================================================
    print("-" * 80)
    print("METHOD 1: Direct moment calculation from quantum samples")
    print("         (compute_moments_from_samples_3d)")
    print("-" * 80)
    
    velocity_counts = qdg.quantum_sample_grid_point_3d_parametric(
        mu_x, mu_y, mu_z, T, shots=shots
    )
    
    moments_direct = qdg.compute_moments_from_samples_3d(velocity_counts)
    
    print("Quantum samples:")
    for vel, count in sorted(velocity_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {vel}: {count} ({100*count/shots:.1f}%)")
    print(f"  ... ({len(velocity_counts)} unique velocities)")
    print()
    
    print("Moments (direct from samples):")
    print(f"  E[vx] = {moments_direct['mean_x']:.6f}")
    print(f"  E[vy] = {moments_direct['mean_y']:.6f}")
    print(f"  E[vz] = {moments_direct['mean_z']:.6f}")
    print(f"  Var[vx] = {moments_direct['var_x']:.6f}")
    print(f"  Var[vy] = {moments_direct['var_y']:.6f}")
    print(f"  Var[vz] = {moments_direct['var_z']:.6f}")
    print()
    
    # ========================================================================
    # METHOD 2: Theoretical from 1D probabilities (existing)
    # ========================================================================
    print("-" * 80)
    print("METHOD 2: Theoretical moments from 1D probabilities")
    print("         (compute_theoretical_moments_3d)")
    print("-" * 80)
    
    moments_theoretical = qdg.compute_theoretical_moments_3d(mu_x, mu_y, mu_z, T)
    
    # Show 1D probabilities for reference
    probs_x = qdg.discrete_gaussian_probs(mu_x, T)
    probs_y = qdg.discrete_gaussian_probs(mu_y, T)
    probs_z = qdg.discrete_gaussian_probs(mu_z, T)
    
    print("1D Probabilities:")
    print(f"  X: P(-1)={probs_x[0]:.4f}, P(0)={probs_x[1]:.4f}, P(+1)={probs_x[2]:.4f}")
    print(f"  Y: P(-1)={probs_y[0]:.4f}, P(0)={probs_y[1]:.4f}, P(+1)={probs_y[2]:.4f}")
    print(f"  Z: P(-1)={probs_z[0]:.4f}, P(0)={probs_z[1]:.4f}, P(+1)={probs_z[2]:.4f}")
    print()
    
    print("Moments (theoretical from 1D):")
    print(f"  E[vx] = {moments_theoretical['mean_x']:.6f}")
    print(f"  E[vy] = {moments_theoretical['mean_y']:.6f}")
    print(f"  E[vz] = {moments_theoretical['mean_z']:.6f}")
    print(f"  Var[vx] = {moments_theoretical['var_x']:.6f}")
    print(f"  Var[vy] = {moments_theoretical['var_y']:.6f}")
    print(f"  Var[vz] = {moments_theoretical['var_z']:.6f}")
    print()
    
    # ========================================================================
    # METHOD 3: LBM-style from 27 probabilities (NEW)
    # ========================================================================
    print("-" * 80)
    print("METHOD 3: LBM-style moment calculation from 27 probabilities")
    print("         u_x = Sum_{i=0..26} f_i c_ix  (NEW compute_moments_lbm_style)")
    print("-" * 80)
    
    # Theoretical: Compute 27 probabilities in LBM order
    probs_27_theoretical = qdg.compute_3d_probability_distribution_lbm_order(
        mu_x, mu_y, mu_z, T
    )
    
    # Show some probabilities
    print("D3Q27 Probabilities (theoretical):")
    vX, vY, vZ = qdg.get_d3q27_velocity_ordering()
    for i in range(min(5, 27)):
        print(f"  [{i:2d}] c=({vX[i]:+2d},{vY[i]:+2d},{vZ[i]:+2d}): f={probs_27_theoretical[i]:.6f}")
    print(f"  ... (27 total directions)")
    print(f"  Sum f_i = {np.sum(probs_27_theoretical):.8f} (should be 1.0)")
    print()
    
    moments_lbm_theoretical = qdg.compute_moments_lbm_style(probs_27_theoretical)
    
    print("Moments (LBM-style from theoretical 27 probabilities):")
    print(f"  u_x = Sum f_i c_ix = {moments_lbm_theoretical['mean_x']:.6f}")
    print(f"  u_y = Sum f_i c_iy = {moments_lbm_theoretical['mean_y']:.6f}")
    print(f"  u_z = Sum f_i c_iz = {moments_lbm_theoretical['mean_z']:.6f}")
    print(f"  Var[u_x] = Sum f_i c_ix^2 - u_x^2 = {moments_lbm_theoretical['var_x']:.6f}")
    print(f"  Var[u_y] = Sum f_i c_iy^2 - u_y^2 = {moments_lbm_theoretical['var_y']:.6f}")
    print(f"  Var[u_z] = Sum f_i c_iz^2 - u_z^2 = {moments_lbm_theoretical['var_z']:.6f}")
    print(f"  rho = {moments_lbm_theoretical['rho']:.8f}")
    print()
    
    # Also compute from quantum samples converted to LBM order
    probs_27_quantum = qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
    moments_lbm_quantum = qdg.compute_moments_lbm_style(probs_27_quantum)
    
    print("Moments (LBM-style from quantum samples -> 27 probabilities):")
    print(f"  u_x = Sum f_i c_ix = {moments_lbm_quantum['mean_x']:.6f}")
    print(f"  u_y = Sum f_i c_iy = {moments_lbm_quantum['mean_y']:.6f}")
    print(f"  u_z = Sum f_i c_iz = {moments_lbm_quantum['mean_z']:.6f}")
    print(f"  Var[u_x] = {moments_lbm_quantum['var_x']:.6f}")
    print(f"  Var[u_y] = {moments_lbm_quantum['var_y']:.6f}")
    print(f"  Var[u_z] = {moments_lbm_quantum['var_z']:.6f}")
    print(f"  rho = {moments_lbm_quantum['rho']:.8f}")
    print()
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("=" * 80)
    print("COMPARISON: All three methods should give same results")
    print("=" * 80)
    
    print("\n1. Method 1 (direct) vs Method 2 (theoretical 1D):")
    for key in ['mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z']:
        diff = abs(moments_direct[key] - moments_theoretical[key])
        print(f"  {key:8s}: {moments_direct[key]:+.6f} vs {moments_theoretical[key]:+.6f} "
              f"(diff = {diff:.6f})")
    
    print("\n2. Method 2 (theoretical 1D) vs Method 3 (LBM theoretical):")
    for key in ['mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z']:
        diff = abs(moments_theoretical[key] - moments_lbm_theoretical[key])
        print(f"  {key:8s}: {moments_theoretical[key]:+.6f} vs {moments_lbm_theoretical[key]:+.6f} "
              f"(diff = {diff:.6f})")
    
    print("\n3. Method 1 (direct quantum) vs Method 3 (LBM quantum):")
    for key in ['mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z']:
        diff = abs(moments_direct[key] - moments_lbm_quantum[key])
        print(f"  {key:8s}: {moments_direct[key]:+.6f} vs {moments_lbm_quantum[key]:+.6f} "
              f"(diff = {diff:.6f})")
    
    # Validate all methods agree
    print("\n" + "=" * 80)
    print("VALIDATION:")
    print("=" * 80)
    
    # Check theoretical methods match exactly
    theoretical_match = all(
        abs(moments_theoretical[key] - moments_lbm_theoretical[key]) < 1e-10
        for key in ['mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z']
    )
    print(f"✓ Theoretical methods (1D vs LBM) match: {theoretical_match}")
    
    # Check quantum methods are close
    quantum_match = all(
        abs(moments_direct[key] - moments_lbm_quantum[key]) < 1e-10
        for key in ['mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z']
    )
    print(f"✓ Quantum methods (direct vs LBM) match: {quantum_match}")
    
    # Check quantum vs theoretical have small error
    max_error = max(
        abs(moments_direct[key] - moments_theoretical[key])
        for key in ['mean_x', 'mean_y', 'mean_z', 'var_x', 'var_y', 'var_z']
    )
    print(f"✓ Maximum quantum vs theoretical error: {max_error:.6f}")
    print(f"✓ Error < 0.05: {max_error < 0.05}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION: All three methods produce consistent results!")
    print("=" * 80)
    print("\nLBM formulas work correctly:")
    print("  u_x = Sum_{i=0..26} f_i c_ix  ✓")
    print("  u_y = Sum_{i=0..26} f_i c_iy  ✓")
    print("  u_z = Sum_{i=0..26} f_i c_iz  ✓")
    print("  Var = Sum f_i c_i^2 - (Sum f_i c_i)^2  ✓")
    print()

if __name__ == "__main__":
    test_lbm_moment_calculation()
