# Quantum Computing for Lattice Boltzmann Method (LBM)

A quantum computing approach to generate Maxwell-Boltzmann distributions for Computational Fluid Dynamics applications.

## 🎯 Overview

This project demonstrates how quantum computing can be applied to generate discrete velocity probability distributions for the Lattice Boltzmann Method (LBM) in computational fluid dynamics. The approach converts fluid grid data (velocity, temperature) into discrete 3-velocity probability distributions f(c=-1,0,+1) using quantum circuits.

## 🔬 Key Features

- **Quantum Maxwell-Boltzmann Generation**: Uses quantum superposition and controlled rotations to create physically realistic velocity distributions
- **Discrete Velocity Conversion**: Maps continuous quantum distributions to discrete velocity states for LBM applications
- **CFD Grid Processing**: Processes entire fluid grids with varying velocity and temperature fields
- **Visualization Tools**: Comprehensive plotting functions for analyzing probability distributions

## 📊 Functions

### Core Functions:
- \`quantum_discrete_velocity_cfd()\`: Main function that converts (velocity, temperature) grids to discrete probability distributions
- \`quantum_maxwell_boltzmann_discrete()\`: Generates Maxwell-Boltzmann distributions using quantum circuits
- \`convert_to_discrete_velocities()\`: Maps continuous velocities to discrete states {-1, 0, +1}
- \`visualize_discrete_cfd_grid()\`: Creates 4-panel visualization of probability distributions

## 🚀 Applications

- **Lattice Boltzmann Method (LBM)**: Direct initialization of velocity distributions
- **Discrete Velocity Models**: Gas kinetics simulations with quantum-enhanced distributions
- **Turbulence Modeling**: Quantum-enhanced probability distributions for turbulent flows
- **Real-time CFD**: Reduced computational cost through quantum parallelism

## 🛠️ Requirements

- Python 3.11+
- Qiskit 2.1.2+
- NumPy
- Matplotlib
- Jupyter Notebook

## 🔧 Installation

\`\`\`bash
# Create conda environment
conda create -n quantum-computing python=3.11
conda activate quantum-computing

# Install packages
pip install qiskit matplotlib numpy jupyter
\`\`\`

## 📚 Theory

The Maxwell-Boltzmann distribution describes the velocity distribution of particles in thermal equilibrium:

\`\`\`
f(v) ∝ exp(-(v-v₀)²/(2σ²))
\`\`\`

Where:
- v₀ is the mean velocity (bulk fluid motion)
- σ² = k_B*T/m relates to temperature
- The quantum circuit approximates this exponential decay through controlled rotations

## 📄 License

MIT License

---

*Bridging quantum computing and computational fluid dynamics for next-generation simulation capabilities.*
