import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from copy import deepcopy
from src.prf.parameters import Parameters
from src import io_utils
from src.prf import fit as prf_fit


def load_vertices(slice_nr, subject):
    try:
        return np.load(DIR_DERIVATIVES / f'sub-{str(subject).zfill(3)}' / 'prf_slices' /
                       f'vertices_slice_{str(slice_nr).zfill(4)}.npy')
    except FileNotFoundError:
        print(f'Slice vertices not found for slice {slice_nr}', flush=True)


def load_params(model_type, search_type, slice_nr, subject):
    try:
        return np.load(
            DIR_DERIVATIVES / f'sub-{str(subject).zfill(3)}' / 'prf_fits' / f'sub-{str(subject).zfill(3)}_{str(slice_nr).zfill(5)}_{model_type}_{search_type}_fit.npy')
    except FileNotFoundError:
        print(
            f'Params not found for {model_type}_{search_type}_fit, slice {slice_nr}', flush=True)
        return False

    # norm_iter_params = np.load(DIR_DERIVATIVES / 'prf_fits' / f'sub-002_{str(i).zfill(5)}_norm_iter_fit.npy')
    # norm_iter_fit[vertices] = norm_iter_params


def concat_slices(n_voxels, n_slices, subject):
    # initialize empty arrays
    placeholder = np.empty((n_voxels, 12))
    placeholder.fill(np.nan)

    gauss_params = placeholder[:, :8]
    norm_params = placeholder[:, :]

    params_dict = {
        'gauss': {
            'grid': deepcopy(gauss_params),
            'iter': deepcopy(gauss_params)
        },
        'norm': {
            'grid': deepcopy(norm_params),
            'iter': deepcopy(norm_params)}}

    # retrieve params
    for slice_nr in range(n_slices):
        vertices = load_vertices(slice_nr, subject)
        if vertices is None:
            continue
        for model in params_dict.keys():
            for search in params_dict[model].keys():
                params_dict[model][search][vertices] = load_params(
                    model, search, slice_nr, subject)

    return params_dict


def save_csv(params, out_path):
    params.to_csv(out_path)
    print(f'CSV saved to {out_path}', flush=True)


def save_params(params_dict, volume_shape, subject):
    # specifying file containing affine transform/header metadata
    metadata_path = DIR_DATA / f'sub-{str(subject).zfill(3)}_ses_pRF_filtered_psc_averageallruns_psc_func.nii.gz'
    out_path = DIR_DERIVATIVES / \
        f'sub-{str(subject).zfill(3)}' / 'prf_analysis'
    os.makedirs(out_path, exist_ok=True)

    for model in params_dict.keys():
        for search in params_dict[model].keys():
            filename_base = f'sub-{str(subject).zfill(3)}_{model}_{search}_fit'
            params = Parameters(
                params_dict[model][search], model=model).to_df()
            save_csv(params, out_path / f'{filename_base}.csv')
            for param in params.columns:
                param_volume = params[param].values.reshape(volume_shape)
                io_utils.save_nifti(
                    param_volume,
                    out_path / f'{filename_base}_{param}.nii.gz',
                    subject,
                    metadata_path=metadata_path)


config = io_utils.load_config()

DIR_DERIVATIVES = Path(config['paths']['derivatives'])

paths = config['paths']['prf_experiment']
DIR_BASE = Path(paths['base'])
DIR_DESIGN = DIR_BASE / paths['design']
DIR_DATA = Path(paths['bold'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()

    subject = args.subject
    n_slices = config['prf']['fit']['n_slices']

    # retrieve volumetric shape for initialization of arrays
    data, _ = prf_fit.load_data(subject)
    volume_shape = data.shape[:3]
    n_voxels = volume_shape[0] * volume_shape[1] * volume_shape[2]

    params_dict = concat_slices(
        n_voxels, n_slices, subject)

    save_params(params_dict, volume_shape, subject)
