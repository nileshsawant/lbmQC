#!/bin/bash

# Load the required modules for the HPC environment
echo "Loading modules..."
module load anaconda3/2024.06.1
module load cuda/12.4

# Activate the conda environment
echo "Activating conda environment..."
source activate /projects/hpcapps/nsawant/qcBac/lbmQC/envs/qiskit_gpu_fresh

# Set environment variables for CUDA if needed by qiskit-aer
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Navigate to the script directory
cd /projects/hpcapps/nsawant/qcBac/lbmQC

echo "================================================="
echo "BENCHMARKING: 10x6x4 grid, 40000 shots"
echo "================================================="
echo

# --- CPU Benchmark ---
echo "--- Running CPU Benchmark ---"
# The 'time' command will measure the execution time
time python visualize_3d_fields.py --Nx 10 --Ny 6 --Nz 4 --shots 40000 --save-only
echo "--- CPU Benchmark Finished ---"
echo
echo

# --- GPU Benchmark ---
echo "--- Running GPU Benchmark ---"
# The 'time' command will measure the execution time
time python visualize_3d_fields_gpu.py --Nx 10 --Ny 6 --Nz 4 --shots 40000 --save-only
echo "--- GPU Benchmark Finished ---"
echo
echo

# --- GPU Interface Benchmark ---
echo "--- Running GPU Interface Benchmark ---"
time python benchmark_interface.py --Nx 10 --Ny 6 --Nz 4 --shots 40000
echo "--- GPU Interface Benchmark Finished ---"
echo

echo "Benchmark complete. Compare the 'real' time from the outputs above."
echo "Also compare the 'STATISTICAL ANALYSIS' sections from each run."
