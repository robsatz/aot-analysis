
import numpy as np
import matplotlib.pyplot as pl
from prfpy import stimulus, model, fit
from prf_logger import FitLogger
import yaml
from pathlib import Path
import numpy as np
import nibabel as nib
import argparse
import nibabel as nib


def load_data(subject, slice_nr):
    data = nib.load(
        DIR_DATA / f'sub-{str(subject).zfill(3)}_ses_pRF_filtered_psc_averageallruns_psc_func.nii.gz').get_fdata()
    design_matrix = np.load(
        DIR_DESIGN / f'sub-{str(subject).zfill(2)}_run-01_design_matrix_output.npy')
    design_matrix = np.moveaxis(design_matrix, 0, -1)  # pixels x pixels x time
    data = data[:, :, :, :339]  # TODO: Remove
    design_matrix = design_matrix[:, :, :339]  # TODO: Remove

    assert data.shape[-1] == design_matrix.shape[-1], "Data and design matrix do not have the same number of timepoints."

    return data, design_matrix


def select_voxels(data, subject):
    subject = str(subject).zfill(3)
    glmsingle_results = nib.load(
        f'derivatives/sub-{subject}/GLMsingle_analysis/sub-{subject}_R2_aot_mean.nii.gz')
    r2 = glmsingle_results.get_fdata().flatten()
    best_voxel = np.nanargmax(r2)
    best_voxel, r2[best_voxel]

    # get all timepoints for selected voxel
    data = data.reshape(-1, data.shape[-1])[best_voxel]

    # need to preserve first dimension for prfpy
    return data[np.newaxis, :]


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
        'fixed_grid_baseline': params['fit']['norm']['fixed_grid_baseline'],
        'grid_bounds': [tuple(params['fit']['amplitude']['prf_ampl_norm']), tuple(params['fit']['norm']['neural_baseline_bound'])],
        'verbose': True
    }

    bounds = [
        (-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
        (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
        (0.2, 1.5*screen_size),  # prf size
        tuple(params['fit']['amplitude']['prf_ampl_norm']),  # prf amplitude
        # bold baseline SHOULD THIS BE 0 OR 1000?
        tuple(params['fit']['amplitude']['bold_bsl']),
        # surround amplitude
        tuple(params['fit']['norm']['surround_amplitude_bound']),
        tuple(params['fit']['norm']['surround_size_bound']
              [0], 3*screen_size),  # surround size
        # neural baseline
        tuple(params['fit']['norm']['neural_baseline_bound']),
        # surround baseline
        tuple(params['fit']['norm']['surround_baseline_bound']),
        # hrf derivative
        tuple(params['fit']['hrf']['deriv_bound']),
        tuple(params['fit']['hrf']['disp_bound'])  # hrf dispersion
    ]

    iterative_fit_params = {
        'rsq_threshold': params['fit']['rsq_threshold'],
        'bounds': bounds,
        'constraints': [],
        'xtol': params['fit']['xtol'],
        'ftol': params['fit']['ftol'],
    }
    return grid_fit_params, iterative_fit_params


def gauss_fit(logger, data, stim, n_jobs, params):
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


# def select_filters(logger, data, stim, params, gauss_fitter):
#     norm_grid = np.nan_to_num(norm_fitter.gridsearch_params)
#     nr = np.sum(norm_grid[:, -1] > rsq_threshold)
#     return


# def norm_fit(logger, data, stim, params):
#     grid_fit_params, iterative_fit_params = define_search_space_norm(
#         stim, params)
#     norm_model = model.Norm_Iso2DGaussianModel(
#         stimulus=stim,
#         hrf=params['hrf']['default'],
#         filter_predictions=filter_predictions,
#         filter_type=filter_type,
#         filter_params=filter_params,
#         normalize_RFs=normalize_RFs,
#         hrf_basis=hrf_basis,
#         normalize_hrf=normalize_hrf
#     )
#     norm_fitter = fit.Norm_Iso2DGaussianFitter(
#         data=data,
#         model=norm_model,
#         n_jobs=n_jobs,
#         previous_gaussian_fitter=deepcopy(logger.fitter),
#         use_previous_gaussian_fitter_hrf=use_previous_gaussian_fitter_hrf
#     )

    logger.attach_fitter('norm', gauss_fitter)
    # runs grid search while logging to file
    logger.grid_fit(
        grid_fit_params, params['fit']['rsq_threshold'], params['fit']['filter_positive'])
    # runs iterative search while logging to file
    logger.iterative_fit(iterative_fit_params)


def batch():
    pass


def fit_prfs(subject, slice_nr, n_jobs, params):
    data, design_matrix = load_data(subject, slice_nr)
    data = select_voxels(data, subject)
    stim = create_stimulus(design_matrix, params)

    logger = FitLogger(subject)
    gauss_fit(logger, data, stim, n_jobs, params)
    # threshold_filters(logger, data, stim, params)
    # # apply DN model using search results from gaussian fit
    # norm_fit(logger, data, stim, params)


# TODO: Batch data
config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
params = config['prf']

paths = config['paths']['prf_experiment']
DIR_BASE = Path(paths['base'])
DIR_DESIGN = DIR_BASE / paths['design']
DIR_DATA = Path(paths['bold'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    parser.add_argument(
        "-slice", "--slice_nr", type=int, default=1)
    parser.add_argument(
        "-n_jobs", "--n_jobs", type=int, default=2)
    args = parser.parse_args()

    subject = args.subject
    slice_nr = args.slice_nr
    n_jobs = args.n_jobs

    fit_prfs(
        subject, slice_nr, n_jobs, params)
