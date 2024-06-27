
import numpy as np
import matplotlib.pyplot as pl
from prfpy import stimulus, model, fit
from pathlib import Path
import numpy as np
import nibabel as nib
import argparse
import nibabel as nib
from copy import deepcopy
import os

from src.prf.logger import FitLogger
from src import io_utils

from src.prf.logger import FitLogger
from src import io_utils


def load_data(subject):
    data = nib.load(
        DIR_DATA / f'sub-{str(subject).zfill(3)}_ses_pRF_filtered_psc_averageallruns_psc_func.nii.gz').get_fdata()
    design_matrix = np.load(
        DIR_DESIGN / f'sub-{str(subject).zfill(2)}_run-01_design_matrix_output.npy')
    design_matrix = np.moveaxis(design_matrix, 0, -1)  # pixels x pixels x time
    data = data[:, :, :, :339]  # TODO: Remove
    design_matrix = design_matrix[:, :, :339]  # TODO: Remove

    assert data.shape[-1] == design_matrix.shape[-1], "Data and design matrix do not have the same number of timepoints."

    return data, design_matrix


def select_voxels(data, n_slices, slice_nr, subject):

    # flatten to n_voxels x n_timepoints
    data = data.reshape(-1, data.shape[-1])
    print('Data shape at import:', data.shape, flush=True)

    # filter nan values
    brain_mask = ~np.isnan(data).all(axis=1)
    print(brain_mask.shape, flush=True)
    brain_vertices = np.where(brain_mask)[0]
    print('Number of valid vertices:', brain_vertices.shape, flush=True)

    # get slice
    slice_vertices = np.array_split(brain_vertices, n_slices, axis=0)[slice_nr]
    data = data[slice_vertices]
    print('Shape of slice:', data.shape, flush=True)

    # save vertices
    subject = str(subject).zfill(3)
    out_path = DIR_DERIVATIVES / f'sub-{subject}' / 'prf_slices'
    os.makedirs(out_path, exist_ok=True)
    np.save(
        out_path / f'vertices_slice_{str(slice_nr).zfill(4)}.npy', slice_vertices)

    return data


def split_timepoints(design_matrix, data):
    n_timepoints = data.shape[-1]
    split_idx = n_timepoints // 2
    return design_matrix[:, :, :split_idx], design_matrix[:, :, split_idx:], data[:, :split_idx], data[:, split_idx:]


def create_stimulus(design_matrix, params):
    return stimulus.PRFStimulus2D(
        screen_size_cm=params['fit']['grid']['screen_size_cm'],
        screen_distance_cm=params['fit']['grid']['screen_distance_cm'],
        design_matrix=design_matrix,
        TR=params['design_matrix']['tr_output']
    )


def define_search_space_gauss(stim, params):
    screen_size = stim.screen_size_degrees
    max_ecc_size = screen_size / 2.0
    n_gridpoints = params['fit']['grid']['n_gridpoints']

    sizes = max_ecc_size * np.linspace(0.25, 1, n_gridpoints)**2
    eccs = max_ecc_size * np.linspace(0.1, 1, n_gridpoints)**2
    polars = np.linspace(0, 2*np.pi, n_gridpoints)

    grid_fit_params = {
        'ecc_grid': eccs,
        'polar_grid': polars,
        'size_grid': sizes,
        'n_batches': n_jobs,
        'fixed_grid_baseline': params['fit']['grid']['fixed_grid_baseline'],
        'grid_bounds': [tuple(params['fit']['amplitude']['prf_ampl_gauss'])],
        'verbose': True
    }

    bounds = [
        (-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
        (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
        (0.2, 1.5*screen_size),  # prf size
        # prf amplitude
        tuple(params['fit']['amplitude']['prf_ampl_gauss']),
        # bold baseline SHOULD THIS BE 0 OR 1000?
        tuple(params['fit']['amplitude']['bold_bsl']),
        tuple(params['fit']['hrf']['deriv_bound']),
        tuple(params['fit']['hrf']['disp_bound'])]

    iterative_fit_params = {
        'rsq_threshold': params['fit']['rsq_threshold'],
        'bounds': bounds,
        'constraints': [],
        'xtol': params['fit']['xtol'],
        'ftol': params['fit']['ftol'],
    }

    return grid_fit_params, iterative_fit_params


def define_search_space_norm(stim, params):
    screen_size = stim.screen_size_degrees
    max_ecc_size = screen_size / 2.0

    grid_fit_params = {
        'surround_amplitude_grid': params['fit']['norm']['surround_amplitude_grid'],
        'surround_size_grid': params['fit']['norm']['surround_size_grid'],
        'neural_baseline_grid': params['fit']['norm']['neural_baseline_grid'],
        'surround_baseline_grid': params['fit']['norm']['surround_baseline_grid'],
        'n_batches': n_jobs,
        'rsq_threshold': params['fit']['rsq_threshold'],
        'fixed_grid_baseline': params['fit']['grid']['fixed_grid_baseline'],
        'grid_bounds': [tuple(params['fit']['amplitude']['prf_ampl_norm']), tuple(params['fit']['norm']['neural_baseline_bound'])],
        'verbose': True
    }

    bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
              (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
              (0.2, 1.5*screen_size),  # prf size
              tuple(params['fit']['amplitude']
                    ['prf_ampl_norm']),  # prf amplitude
              # bold baseline SHOULD THIS BE 0 OR 1000?
              tuple(params['fit']['amplitude']['bold_bsl']),
              # surround amplitude
              tuple(params['fit']['norm']['surround_amplitude_bound']),
              (0.1, 3*screen_size),  # surround size
              # neural baseline
              tuple(params['fit']['norm']['neural_baseline_bound']),
              tuple([float(item) for item in params['fit']['norm']
                     ['surround_baseline_bound']]),  # surround baseline
              tuple(params['fit']['hrf']['deriv_bound']),  # hrf derivative
              tuple(params['fit']['hrf']['disp_bound'])]  # hrf dispersion

    iterative_fit_params = {
        'rsq_threshold': params['fit']['rsq_threshold'],
        'bounds': bounds,
        'constraints': [],
        'xtol': params['fit']['xtol'],
        'ftol': params['fit']['ftol'],
    }
    return grid_fit_params, iterative_fit_params


def gauss_fit(logger, stim, data, n_jobs, params):
    grid_fit_params, iterative_fit_params = define_search_space_gauss(
        stim, params)
    gauss_model = model.Iso2DGaussianModel(
        stimulus=stim, hrf=params['fit']['hrf']['default'])
    gauss_fitter = fit.Iso2DGaussianFitter(
        data=data, model=gauss_model, n_jobs=n_jobs, fit_hrf=True)

    logger.attach_fitter('gauss', gauss_fitter)
    # runs grid search while logging to file
    logger.grid_fit(
        grid_fit_params, params['fit']['rsq_threshold'], params['fit']['filter_positive'])
    # runs iterative search while logging to file
    logger.iterative_fit(iterative_fit_params)


def norm_fit(logger, stim, data, n_jobs, params):
    grid_fit_params, iterative_fit_params = define_search_space_norm(
        stim, params)
    norm_model = model.Norm_Iso2DGaussianModel(
        stimulus=stim,
        hrf=params['fit']['hrf']['default']
    )
    norm_fitter = fit.Norm_Iso2DGaussianFitter(
        data=data,
        model=norm_model,
        n_jobs=n_jobs,
        previous_gaussian_fitter=deepcopy(logger.fitter),
        use_previous_gaussian_fitter_hrf=params['fit']['norm']['use_previous_gaussian_fitter_hrf']
    )

    logger.attach_fitter('norm', norm_fitter)
    # runs grid search while logging to file
    logger.grid_fit(
        grid_fit_params, params['fit']['rsq_threshold'], params['fit']['filter_positive'])
    # runs iterative search while logging to file
    logger.iterative_fit(iterative_fit_params)


def run_pipeline(subject, n_slices, slice_nr, n_jobs, params):
    # prepare inputs
    data, design_matrix = load_data(subject)
    data = select_voxels(data, n_slices, slice_nr, subject)
    design_matrix_train, design_matrix_test, data_train, data_test = split_timepoints(
        design_matrix, data)
    print('Shapes after splitting', [arr.shape for arr in (design_matrix_train,
          design_matrix_test, data_train, data_test)], flush=True)
    stim_train = create_stimulus(design_matrix_train, params)
    stim_test = create_stimulus(design_matrix_test, params)

    # fit and evaluate models
    logger = FitLogger(subject, slice_nr)
    gauss_fit(logger, stim_train, data_train, n_jobs, params)
    if params['fit']['filter_positive']:
        logger.filter_positive_prfs()
    # apply DN model using search results from gaussian fit
    norm_fit(logger, stim_train, data_train, n_jobs, params)
    logger.crossvalidate_fit(stim_test, data_test)


config = io_utils.load_config()
config = io_utils.load_config()
params = config['prf']

DIR_DERIVATIVES = Path(config['paths']['derivatives'])

paths = config['paths']['prf_experiment']
DIR_BASE = Path(paths['base'])
DIR_DESIGN = DIR_BASE / paths['design']
DIR_DATA = Path(paths['bold'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    parser.add_argument(
        "-slice", "--slice_nr", type=int, default=0)
    parser.add_argument(
        "--n_slices", type=int, default=1)
    parser.add_argument(
        "--n_jobs", type=int, default=2)
    args = parser.parse_args()

    subject = args.subject
    n_slices = args.n_slices
    slice_nr = args.slice_nr
    n_jobs = args.n_jobs

    run_pipeline(
        subject, n_slices, slice_nr, n_jobs, params)
