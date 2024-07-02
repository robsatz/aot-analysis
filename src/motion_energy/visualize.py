import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
from copy import deepcopy

from src import io_utils
from src.motion_energy import fit as moten_fit


def load_vertices(slice_nr, subject):
    try:
        return np.load(DIR_DERIVATIVES
                       / f'sub-{str(subject).zfill(3)}'
                       / 'moten_slices'
                       / f'vertices_slice_{str(slice_nr).zfill(4)}.npy')
    except FileNotFoundError:
        print(f'Slice vertices not found for slice {slice_nr}', flush=True)


def load_betas(segmentation, aot_condition, slice_nr, subject):
    try:
        return np.load(
            DIR_DERIVATIVES
            / f'sub-{str(subject).zfill(3)}'
            / f'fracridge_{segmentation}_{aot_condition}'
            / f'sub-{str(subject).zfill(3)}_slice-{str(slice_nr).zfill(4)}_betas.npy')
    except FileNotFoundError:
        print(
            f'Betas not found for {segmentation}, slice {slice_nr}', flush=True)
        return False


def load_filters():
    with open(DIR_MOTION_ENERGY / 'pyramid.pkl', "rb") as f:
        pyramid = pickle.load(f)
    filters = pd.DataFrame(pyramid.filters)
    filters['ecc'] = np.sqrt(filters['centerh']**2+filters['centerv']**2)
    filters['polar'] = np.angle(filters['centerh']+filters['centerv']*1j)
    return filters


def get_volume_shape(segmentation, subject):
    amplitudes = moten_fit.load_amplitudes(
        subject, session=1, segmentation=segmentation)
    return amplitudes.shape[:-1]


def select_voxel_params(filters, betas):
    # for each voxel, choose filter with highest beta value
    best_filter_idx = np.nanargmax(betas, axis=0)
    best_params = filters.iloc[best_filter_idx]
    return best_params.values


def concat_slices(filters, segmentation, aot_condition, n_voxels, n_slices, subject):
    # initialize empty array
    n_params = filters.shape[1]
    best_params = np.zeros((n_voxels, n_params))

    # retrieve params
    for slice_nr in range(n_slices):
        vertices = load_vertices(slice_nr, subject)
        if vertices is None:
            continue
        betas = load_betas(segmentation, aot_condition, slice_nr, subject)
        best_params[vertices] = select_voxel_params(filters, betas)

    return best_params


def main(subject, segmentation, aot_condition, n_slices):
    out_path = DIR_DERIVATIVES / \
        f'sub-{str(subject).zfill(3)}' / \
        'moten_analysis'
    os.makedirs(out_path, exist_ok=True)

    filename_base = f'sub-{str(subject).zfill(3)}'

    filters = load_filters()
    param_names = filters.columns
    volume_shape = get_volume_shape(segmentation, subject)
    n_voxels = np.prod(volume_shape)
    params = concat_slices(filters, segmentation,
                           aot_condition, n_voxels, n_slices, subject)
    # unflatten to volume shape
    params = params.reshape(volume_shape + (-1,))

    for param_idx, param_label in enumerate(param_names):
        print(
            f'Storing nifti for param {param_label}', flush=True)
        io_utils.save_nifti(
            params[:, :, :, param_idx],
            out_path / f'{filename_base}_{param_label}.nii.gz',
            subject)


config = io_utils.load_config()
DIR_DERIVATIVES = Path(config['paths']['derivatives'])
DIR_MOTION_ENERGY = Path(config['paths']['motion_energy'])
DIR_DATA = Path(config['paths']['data'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()

    subject = args.subject
    segmentation = config['fracridge']['segmentation']
    aot_condition = config['fracridge']['aot_condition']
    n_slices = config['fracridge']['n_slices']

    main(subject, segmentation, aot_condition, n_slices)
