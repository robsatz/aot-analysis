# aot-analysis

Data analysis code for the Arrow of Time project, an initiative to produce a large-scale movie-watching fMRI dataset.

The work in this repo was produced in the scope of my master's thesis project at the Spinoza Centre for Neuroimaging under the supervision of Dr. Tomas Knapen.

## Setup

- Clone the repository
- Open a terminal and navigate to the root of this repository
- Run `conda env create -f environment.yml`
- The code of this repo will be installed in your conda environment as `aot_analysis`
- Activate the environment via `conda activate aot_analysis`
- Change the paths in `config.yml`

## Structure

The package is structured by model type.
The three implemented model types are described below.
After fitting, the results are analyzed and visualized in [notebooks](./notebooks).

### glmsingle

Fitting single-trial response amplitudes using the [GLMsingle](https://github.com/cvnlab/GLMsingle) package.

Intended execution order: 
- [fit](./aot_analysis/glmsingle/fit.py)
- [visualize](./aot_analysis/glmsingle/visualize.py)

### prf

Fitting population receptive fields using the [prfpy](https://github.com/VU-Cog-Sci/prfpy/) package.

Intended execution order:
- [create design matrices](./aot_analysis/prf/design_matrix.py)
- [fit](./aot_analysis/prf/fit.py)
- [visualize](./aot_analysis/prf/visualize.py)

### motion_energy

Fitting a motion energy model using the [pymoten](https://github.com/gallantlab/pymoten) package.

Intended execution order:
- [extract motion energy features](./aot_analysis/motion_energy/features.py)
- [fit](./aot_analysis/motion_energy/fit.py)
- [visualize](./aot_analysis/motion_energy/visualize.py)

