# PVA/Na-MMT interface: MD + surrogate-assisted (qEHVI) optimization

Data and code supporting the manuscript:

**"Interfacial adhesion, hydrogen bonding and thermal behavior of poly(vinyl alcohol)/sodium montmorillonite composites: a molecular dynamics study with a surrogate-assisted optimization framework"**
M. Z. Koç, Karabuk University. Submitted to *Computational Materials Science*.

## Overview

This repository contains the LAMMPS input files, the equilibrated interfacial configuration, and the Python scripts used to (i) characterize the all-atom PVA/Na-montmorillonite interface and (ii) run the molecular-dynamics-in-the-loop multi-objective Bayesian optimization (qEHVI) over clay loading.

## Contents

### MD input and configuration
- `in.interface.txt` — main LAMMPS input script (equilibration + interfacial property measurement)
- `interface.data` — equilibrated PVA/Na-MMT interfacial configuration (9074 atoms)

### Optimization and analysis (Python)
- `bridge_level2.py` — MD-in-the-loop qEHVI over clay loading (adhesion vs matrix mobility)
- `error_bars.py` — repeatability study (3 systems x 3 random seeds), mean +/- std
- `qehvi_vs_random_realMD.py` — qEHVI vs random search on the real-MD-derived response surface
- `plot_*.py` — figure-generation scripts (density profile, Tg, Pareto front, etc.)

## Requirements

- **LAMMPS** (22 Jul 2025 or compatible), built with the CLASS2/KSPACE/MOLECULE packages
- **Python 3.11** with: `botorch`, `torch`, `gpytorch`, `numpy`, `matplotlib`

## How to reproduce

1. Run the interfacial characterization:
   `mpirun -np 10 lmp -in in.interface.txt`
2. Run the clay-loading optimization loop:
   `python bridge_level2.py`
3. Generate figures with the corresponding `plot_*.py` scripts.

## Force fields

- PVA: CHARMM-compatible (CGenFF) parameterization
- Na-montmorillonite: INTERFACE force field

## Citation

If you use this code or data, please cite the associated article (details to be updated upon publication).

## Contact

Muhammed Zahid Koç — mzkoc@karabuk.edu.tr
Department of Industrial Engineering, Karabuk University, Turkey
