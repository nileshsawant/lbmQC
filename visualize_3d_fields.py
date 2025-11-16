"""
Quick 2D Field Visualization for 3D Quantum LBM

This script generates side-by-side comparisons of:
- Input fields: mu_x, mu_y, mu_z, T (theoretical parameters)
- Output fields: E[vx], E[vy], E[vz], Var (from quantum sampling)

Shows 2D slices through the 3D grid to visualize spatial variations.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_discrete_gaussian import QuantumDiscreteGaussian
import argparse


def visualize_fields(Nx=10, Ny=6, Nz=4, slice_index=2, shots=3000, save_only=False, use_lbm_moments=False):
    """
    Generate field visualization comparing input and quantum output.
    
    Parameters:
    - Nx, Ny, Nz: Grid dimensions
    - slice_index: Which z-slice to visualize
    - shots: Number of quantum samples per grid point
    - save_only: If True, save plot without showing
    - use_lbm_moments: If True, use LBM-style calculation (uₓ = Σ fᵢ cᵢₓ)
    """
    print("=" * 70)
    print("3D QUANTUM LBM FIELD VISUALIZATION")
    print("=" * 70)
    print(f"\nGrid: {Nx} × {Ny} × {Nz} = {Nx*Ny*Nz} points")
    print(f"Visualizing z-slice: {slice_index} (of {Nz})")
    print(f"Shots per point: {shots}")
    print(f"Moment calculation: {'LBM-style (Σ fᵢ cᵢₓ)' if use_lbm_moments else 'Direct from samples'}")
    print()
    
    # Initialize quantum sampler
    # Initialize QuantumDiscreteGaussian with matching 1D grid_size for clarity
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
    print("Note: mu_z varies primarily with z-coordinate")
    print("      A 2D slice at constant z shows secondary x-variation")
    print()
    
    # Storage for quantum output moments
    quantum_means_x = np.zeros((Nx, Ny, Nz))
    quantum_means_y = np.zeros((Nx, Ny, Nz))
    quantum_means_z = np.zeros((Nx, Ny, Nz))
    quantum_vars_x = np.zeros((Nx, Ny, Nz))
    quantum_vars_y = np.zeros((Nx, Ny, Nz))
    quantum_vars_z = np.zeros((Nx, Ny, Nz))
    
    # Sample all grid points
    print(f"Quantum sampling {Nx*Ny*Nz} grid points...")
    print("(This may take several minutes)")
    print()
    
    total_points = Nx * Ny * Nz
    for idx, (i, j, k) in enumerate([(i, j, k) for i in range(Nx) for j in range(Ny) for k in range(Nz)]):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Progress: {idx+1}/{total_points} points ({100*(idx+1)/total_points:.1f}%)")
        
        mu_x = means_x[i, j, k]
        mu_y = means_y[i, j, k]
        mu_z = means_z[i, j, k]
        T = temperatures[i, j, k]
        
        # Quantum sampling
        velocity_counts = qdg.quantum_sample_grid_point_3d_parametric(mu_x, mu_y, mu_z, T, shots)
        
        # Compute moments (two methods available)
        if use_lbm_moments:
            # LBM-style: Convert to 27 probabilities, then compute moments
            probs_27 = qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
            moments = qdg.compute_moments_lbm_style(probs_27)
        else:
            # Direct: Compute moments directly from velocity samples
            moments = qdg.compute_moments_from_samples_3d(velocity_counts)
        
        quantum_means_x[i, j, k] = moments['mean_x']
        quantum_means_y[i, j, k] = moments['mean_y']
        quantum_means_z[i, j, k] = moments['mean_z']
        quantum_vars_x[i, j, k] = moments['var_x']
        quantum_vars_y[i, j, k] = moments['var_y']
        quantum_vars_z[i, j, k] = moments['var_z']
    
    print(f"  Progress: {total_points}/{total_points} points (100.0%)")
    print("\nSampling complete!")
    print()
    
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
    print("Generating plots...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    # Use conventional image orientation: x -> columns, y -> rows
    # mu_x_slice has shape (Nx, Ny). Matplotlib imshow expects (rows=y, cols=x),
    # so transpose the slice arrays before plotting and set extent accordingly.
    extent = [0, Nx - 1, 0, Ny - 1]

    # Transpose slices so axis 0 -> y (rows) and axis 1 -> x (cols)
    mu_x_plot = mu_x_slice.T

    # Row 1: uₓ (x-velocity)
    im0 = axes[0, 0].imshow(mu_x_plot, aspect='auto', cmap='RdBu_r', 
                            origin='lower', extent=extent)
    axes[0, 0].set_title(f'Input: ux (Mean X-Velocity) [z={slice_index}]', fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], label='ux')
    
    im1 = axes[0, 1].imshow(qmean_x_slice.T, aspect='auto', cmap='RdBu_r', 
                            origin='lower', extent=extent, vmin=im0.get_clim()[0], vmax=im0.get_clim()[1])
    axes[0, 1].set_title(f'Quantum: E[vx] from Samples [z={slice_index}]', fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1], label='E[vx]')
    
    # Row 2: uᵧ (y-velocity)
    im2 = axes[1, 0].imshow(mu_y_slice.T, aspect='auto', cmap='RdBu_r', 
                            origin='lower', extent=extent)
    axes[1, 0].set_title(f'Input: uy (Mean Y-Velocity) [z={slice_index}]', fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0], label='uy')
    
    im3 = axes[1, 1].imshow(qmean_y_slice.T, aspect='auto', cmap='RdBu_r', 
                            origin='lower', extent=extent, vmin=im2.get_clim()[0], vmax=im2.get_clim()[1])
    axes[1, 1].set_title(f'Quantum: E[vy] from Samples [z={slice_index}]', fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1], label='E[vy]')
    
    # Row 3: uz (z-velocity)
    im4 = axes[2, 0].imshow(mu_z_slice.T, aspect='auto', cmap='RdBu_r', 
                            origin='lower', extent=extent)
    axes[2, 0].set_title(f'Input: uz (Mean Z-Velocity) [z={slice_index}]', fontweight='bold')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[2, 0], label='uz')
    
    im5 = axes[2, 1].imshow(qmean_z_slice.T, aspect='auto', cmap='RdBu_r', 
                            origin='lower', extent=extent, vmin=im4.get_clim()[0], vmax=im4.get_clim()[1])
    axes[2, 1].set_title(f'Quantum: E[vz] from Samples [z={slice_index}]', fontweight='bold')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[2, 1], label='E[vz]')
    
    # Row 4: Temperature / Variance
    im6 = axes[3, 0].imshow(T_slice.T, aspect='auto', cmap='hot', 
                            origin='lower', extent=extent)
    axes[3, 0].set_title(f'Input: T (Temperature) [z={slice_index}]', fontweight='bold')
    axes[3, 0].set_xlabel('x')
    axes[3, 0].set_ylabel('y')
    plt.colorbar(im6, ax=axes[3, 0], label='T')
    
    im7 = axes[3, 1].imshow(qvar_avg_slice.T, aspect='auto', cmap='hot', 
                            origin='lower', extent=extent, vmin=im6.get_clim()[0], vmax=im6.get_clim()[1])
    axes[3, 1].set_title(f'Quantum: Avg(Var[vx,vy,vz]) [z={slice_index}]', fontweight='bold')
    axes[3, 1].set_xlabel('x')
    axes[3, 1].set_ylabel('y')
    plt.colorbar(im7, ax=axes[3, 1], label='Variance')
    
    plt.tight_layout()
    
    filename = f'field_comparison_{Nx}x{Ny}x{Nz}_z{slice_index}.png'
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
    
    # Validation
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
        description='Visualize 3D quantum LBM fields',
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
    
    args = parser.parse_args()
    
    visualize_fields(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        slice_index=args.slice,
        shots=args.shots,
        save_only=args.save_only,
        use_lbm_moments=args.lbm_moments
    )
    
    print("✓ Field visualization complete!")
    print()


if __name__ == "__main__":
    main()
