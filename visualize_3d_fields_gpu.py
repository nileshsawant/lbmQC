"""
Quick 2D Field Visualization for 3D Quantum LBM - GPU PARALLEL VERSION

This script generates side-by-side comparisons of:
- Input fields: mu_x, mu_y, mu_z, T (theoretical parameters)
- Output fields: E[vx], E[vy], E[vz], Var (from quantum sampling)

Shows 2D slices through the 3D grid to visualize spatial variations.

GPU ACCELERATION:
This version batches all quantum circuits for all grid points and runs them
in a single parallel job on a GPU-accelerated AerSimulator.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_discrete_gaussian import QuantumDiscreteGaussian
import argparse
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import time


def visualize_fields_gpu(Nx=10, Ny=6, Nz=4, slice_index=2, shots=3000, save_only=False, use_lbm_moments=False, batch_size=1024):
    """
    Generate field visualization comparing input and quantum output using GPU acceleration.
    
    Parameters:
    - Nx, Ny, Nz: Grid dimensions
    - slice_index: Which z-slice to visualize
    - shots: Number of quantum samples per grid point
    - save_only: If True, save plot without showing
    - use_lbm_moments: If True, use LBM-style calculation (uₓ = Σ fᵢ cᵢₓ)
    - batch_size: Number of circuits to run in each GPU batch.
    """
    print("=" * 70)
    print("3D QUANTUM LBM FIELD VISUALIZATION - GPU PARALLEL")
    print("=" * 70)
    print(f"\nGrid: {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz} points")
    print(f"Visualizing z-slice: {slice_index} (of {Nz})")
    print(f"Shots per point: {shots}")
    print(f"Moment calculation: {'LBM-style (Σ fᵢ cᵢₓ)' if use_lbm_moments else 'Direct from samples'}")
    print(f"GPU batch size: {batch_size}")
    print()
    
    # Initialize quantum sampler
    qdg = QuantumDiscreteGaussian(grid_size=Nx, circuit_type='symmetric', 
                                   grid_3d=(Nx, Ny, Nz))
    
    # Compute 3D input parameters
    print("Computing input fields...")
    means_x, means_y, means_z, temperatures = qdg.compute_parameters_3d()
    
    print(f"\nInput field ranges:")
    print(f"  mu_x: [{means_x.min():.4f}, {means_x.max():.4f}]")
    print(f"  mu_y: [{means_y.min():.4f}, {means_y.max():.4f}]")
    print(f"  mu_z: [{means_z.min():.4f}, {means_z.max():.4f}]")
    print(f"  T:  [{temperatures.min():.4f}, {temperatures.max():.4f}]")
    print()
    
    # Prepare all circuits for batch execution
    total_points = Nx * Ny * Nz
    grid_points = [(i, j, k) for i in range(Nx) for j in range(Ny) for k in range(Nz)]
    
    print(f"Generating {total_points} quantum circuits...")
    circuits = []
    for i, j, k in grid_points:
        mu_x = means_x[i, j, k]
        mu_y = means_y[i, j, k]
        mu_z = means_z[i, j, k]
        T = temperatures[i, j, k]
        qc = qdg.create_quantum_circuit_3d_parametric(mu_x, mu_y, mu_z, T)
        qc.name = f"point_{i}_{j}_{k}"
        circuits.append(qc)
    print("Circuit generation complete.")

    # Set up GPU simulator
    print("\nConfiguring GPU simulator...")
    try:
        simulator = AerSimulator(device='GPU')
        print("✓ AerSimulator configured for GPU.")
    except Exception as e:
        print(f"✗ GPU simulator failed: {e}")
        print("Falling back to CPU simulator.")
        simulator = AerSimulator()

    # Compile circuits
    print("Compiling circuits (transpilation)...")
    start_time = time.time()
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=simulator)
    compiled_circuits = pass_manager.run(circuits)
    compile_time = time.time() - start_time
    print(f"✓ Circuits compiled in {compile_time:.2f} seconds.")

    # Run circuits in batches
    print(f"\nExecuting {total_points} circuits in batches of {batch_size}...")
    all_counts = {}
    start_time = time.time()
    
    for i in range(0, total_points, batch_size):
        batch = compiled_circuits[i:i+batch_size]
        print(f"  Running batch {i//batch_size + 1}/{-(-total_points//batch_size)} (circuits {i} to {i+len(batch)-1})")
        job = simulator.run(batch, shots=shots)
        result = job.result()
        
        for circ in batch:
            try:
                counts = result.get_counts(circ.name)
                all_counts[circ.name] = counts
            except Exception:
                print(f"    Warning: No counts found for circuit {circ.name}")
                all_counts[circ.name] = {}

    exec_time = time.time() - start_time
    print(f"✓ All batches executed in {exec_time:.2f} seconds.")
    print("\nSampling complete!")

    # Storage for quantum output moments
    quantum_means_x = np.zeros((Nx, Ny, Nz))
    quantum_means_y = np.zeros((Nx, Ny, Nz))
    quantum_means_z = np.zeros((Nx, Ny, Nz))
    quantum_vars_x = np.zeros((Nx, Ny, Nz))
    quantum_vars_y = np.zeros((Nx, Ny, Nz))
    quantum_vars_z = np.zeros((Nx, Ny, Nz))

    # Process results
    print("\nProcessing results...")
    for idx, (i, j, k) in enumerate(grid_points):
        circuit_name = f"point_{i}_{j}_{k}"
        counts = all_counts.get(circuit_name, {})
        
        velocity_counts = qdg._decode_quantum_counts_3d(counts)

        if use_lbm_moments:
            probs_27 = qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
            moments = qdg.compute_moments_lbm_style(probs_27)
        else:
            moments = qdg.compute_moments_from_samples_3d(velocity_counts)
        
        quantum_means_x[i, j, k] = moments.get('mean_x', 0)
        quantum_means_y[i, j, k] = moments.get('mean_y', 0)
        quantum_means_z[i, j, k] = moments.get('mean_z', 0)
        quantum_vars_x[i, j, k] = moments.get('var_x', 0)
        quantum_vars_y[i, j, k] = moments.get('var_y', 0)
        quantum_vars_z[i, j, k] = moments.get('var_z', 0)
    print("✓ Results processed.")

    # Extract 2D slices
    mu_x_slice = means_x[:, :, slice_index]
    mu_y_slice = means_y[:, :, slice_index]
    mu_z_slice = means_z[:, :, slice_index]
    T_slice = temperatures[:, :, slice_index]
    
    qmean_x_slice = quantum_means_x[:, :, slice_index]
    qmean_y_slice = quantum_means_y[:, :, slice_index]
    qmean_z_slice = quantum_means_z[:, :, slice_index]
    qvar_avg_slice = (quantum_vars_x[:, :, slice_index] + 
                      quantum_vars_y[:, :, slice_index] + 
                      quantum_vars_z[:, :, slice_index]) / 3.0
    
    # Create visualization
    print("\nGenerating plots...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    extent = [0, Nx - 1, 0, Ny - 1]

    # Row 1: uₓ (x-velocity)
    im0 = axes[0, 0].imshow(mu_x_slice.T, aspect='auto', cmap='RdBu_r', origin='lower', extent=extent)
    axes[0, 0].set_title(f'Input: ux (Mean X-Velocity) [z={slice_index}]', fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], label='ux')
    
    im1 = axes[0, 1].imshow(qmean_x_slice.T, aspect='auto', cmap='RdBu_r', origin='lower', extent=extent, vmin=im0.get_clim()[0], vmax=im0.get_clim()[1])
    axes[0, 1].set_title(f'Quantum: E[vx] from Samples [z={slice_index}]', fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1], label='E[vx]')
    
    # Row 2: uᵧ (y-velocity)
    im2 = axes[1, 0].imshow(mu_y_slice.T, aspect='auto', cmap='RdBu_r', origin='lower', extent=extent)
    axes[1, 0].set_title(f'Input: uy (Mean Y-Velocity) [z={slice_index}]', fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0], label='uy')
    
    im3 = axes[1, 1].imshow(qmean_y_slice.T, aspect='auto', cmap='RdBu_r', origin='lower', extent=extent, vmin=im2.get_clim()[0], vmax=im2.get_clim()[1])
    axes[1, 1].set_title(f'Quantum: E[vy] from Samples [z={slice_index}]', fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1], label='E[vy]')
    
    # Row 3: uz (z-velocity)
    im4 = axes[2, 0].imshow(mu_z_slice.T, aspect='auto', cmap='RdBu_r', origin='lower', extent=extent)
    axes[2, 0].set_title(f'Input: uz (Mean Z-Velocity) [z={slice_index}]', fontweight='bold')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[2, 0], label='uz')
    
    im5 = axes[2, 1].imshow(qmean_z_slice.T, aspect='auto', cmap='RdBu_r', origin='lower', extent=extent, vmin=im4.get_clim()[0], vmax=im4.get_clim()[1])
    axes[2, 1].set_title(f'Quantum: E[vz] from Samples [z={slice_index}]', fontweight='bold')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[2, 1], label='E[vz]')
    
    # Row 4: Temperature / Variance
    im6 = axes[3, 0].imshow(T_slice.T, aspect='auto', cmap='hot', origin='lower', extent=extent)
    axes[3, 0].set_title(f'Input: T (Temperature) [z={slice_index}]', fontweight='bold')
    axes[3, 0].set_xlabel('x')
    axes[3, 0].set_ylabel('y')
    plt.colorbar(im6, ax=axes[3, 0], label='T')
    
    im7 = axes[3, 1].imshow(qvar_avg_slice.T, aspect='auto', cmap='hot', origin='lower', extent=extent, vmin=im6.get_clim()[0], vmax=im6.get_clim()[1])
    axes[3, 1].set_title(f'Quantum: Avg(Var[vx,vy,vz]) [z={slice_index}]', fontweight='bold')
    axes[3, 1].set_xlabel('x')
    axes[3, 1].set_ylabel('y')
    plt.colorbar(im7, ax=axes[3, 1], label='Variance')
    
    plt.tight_layout()
    
    filename = f'field_comparison_gpu_{Nx}x{Ny}x{Nz}_z{slice_index}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {filename}")
    
    if not save_only:
        plt.show()
    else:
        plt.close()
    
    # Compute and print statistics
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    error_mean_x = np.abs(qmean_x_slice - mu_x_slice)
    error_mean_y = np.abs(qmean_y_slice - mu_y_slice)
    error_mean_z = np.abs(qmean_z_slice - mu_z_slice)
    error_var = np.abs(qvar_avg_slice - T_slice)
    
    print(f"\nMean Velocity Errors (z-slice {slice_index}):")
    print(f"  E[vx]: mean={np.mean(error_mean_x):.6f}, max={np.max(error_mean_x):.6f}, std={np.std(error_mean_x):.6f}")
    print(f"  E[vy]: mean={np.mean(error_mean_y):.6f}, max={np.max(error_mean_y):.6f}, std={np.std(error_mean_y):.6f}")
    print(f"  E[vz]: mean={np.mean(error_mean_z):.6f}, max={np.max(error_mean_z):.6f}, std={np.std(error_mean_z):.6f}")
    
    print(f"\nVariance Error:")
    print(f"  Var: mean={np.mean(error_var):.6f}, max={np.max(error_var):.6f}, std={np.std(error_var):.6f}")
    
    print(f"\nField Ranges (z-slice {slice_index}):")
    print(f"  Input mu_x: [{mu_x_slice.min():.4f}, {mu_x_slice.max():.4f}]")
    print(f"  Output E[vx]: [{qmean_x_slice.min():.4f}, {qmean_x_slice.max():.4f}]")
    print(f"  Input mu_y: [{mu_y_slice.min():.4f}, {mu_y_slice.max():.4f}]")
    print(f"  Output E[vy]: [{qmean_y_slice.min():.4f}, {qmean_y_slice.max():.4f}]")
    print(f"  Input mu_z: [{mu_z_slice.min():.4f}, {mu_z_slice.max():.4f}]")
    print(f"  Output E[vz]: [{qmean_z_slice.min():.4f}, {qmean_z_slice.max():.4f}]")
    print(f"  Input T: [{T_slice.min():.4f}, {T_slice.max():.4f}]")
    print(f"  Output Var: [{qvar_avg_slice.min():.4f}, {qvar_avg_slice.max():.4f}]")
    
    threshold = 0.05
    mean_errors = [error_mean_x, error_mean_y, error_mean_z]
    all_pass = all(np.max(err) < threshold for err in mean_errors) and np.max(error_var) < threshold
    
    print(f"\nValidation (threshold: {threshold}):")
    print(f"  Mean errors: {'✓ PASS' if all(np.max(err) < threshold for err in mean_errors) else '✗ FAIL'}")
    print(f"  Variance error: {'✓ PASS' if np.max(error_var) < threshold else '✗ FAIL'}")
    print(f"  Overall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    print("=" * 70)
    print()


def main():
    """
    Main execution with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Visualize 3D quantum LBM fields on GPU',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--Nx', type=int, default=10, help='Grid size in X direction')
    parser.add_argument('--Ny', type=int, default=6, help='Grid size in Y direction')
    parser.add_argument('--Nz', type=int, default=4, help='Grid size in Z direction')
    parser.add_argument('--slice', type=int, default=2, help='Z-slice index to visualize')
    parser.add_argument('--shots', type=int, default=3000, help='Quantum shots per grid point')
    parser.add_argument('--save-only', action='store_true', help='Save plot without displaying')
    parser.add_argument('--lbm-moments', action='store_true', 
                        help='Use LBM-style moment calculation (uₓ = Σ fᵢ cᵢₓ)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Number of circuits per GPU batch')
    
    args = parser.parse_args()
    
    visualize_fields_gpu(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        slice_index=args.slice,
        shots=args.shots,
        save_only=args.save_only,
        use_lbm_moments=args.lbm_moments,
        batch_size=args.batch_size
    )
    
    print("✓ GPU field visualization complete!")
    print()


if __name__ == "__main__":
    main()
