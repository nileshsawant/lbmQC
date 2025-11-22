#!/bin/bash

# Load the required modules for the HPC environment
echo "Loading modules..."
module load anaconda3/2024.06.1
module load cuda/12.4

# Activate the conda environment
echo "Activating conda environment..."
source activate /projects/hpcapps/nsawant/qcBac/lbmQC/envs/qiskit_gpu_fresh

# Set environment variables for CUDA
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install qiskit-aer with GPU support from the local directory
echo "Installing qiskit-aer with GPU support..."
cd /projects/hpcapps/nsawant/qcBac/lbmQC/qiskit-aer

# Clean previous builds
rm -rf _skbuild

# Install with GPU support. This command assumes the qiskit-aer build system
# will detect the CUDA environment.
pip install . --config-settings=cmake.define.CMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc

# Navigate back to the script directory
cd /projects/hpcapps/nsawant/qcBac/lbmQC

# Run the GPU-accelerated visualization script
echo "Running the GPU visualization script..."
python visualize_3d_fields_gpu.py --Nx 32 --Ny 32 --Nz 32 --shots 4096 --batch-size 2048

echo "Script execution finished."
