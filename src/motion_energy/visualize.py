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


def load_results(segmentation, aot_condition, slice_nr, subject, results_type):
    try:
        results = np.load(
            DIR_DERIVATIVES
            / f'sub-{str(subject).zfill(3)}'
            / f'fracridge_{segmentation}_{aot_condition}'
            / f'sub-{str(subject).zfill(3)}_slice-{str(slice_nr).zfill(4)}_{results_type}.npy')
        print(
            f'loading {results_type} for {segmentation}, slice {slice_nr} - shape: {results.shape}', flush=True)
        return results
    except FileNotFoundError:
        print(
            f'{results_type} not found for {segmentation}, slice {slice_nr}', flush=True)
        return False


def load_filters(screen_size_cm, screen_distance_cm):
    with open(DIR_MOTION_ENERGY / 'pyramid.pkl', "rb") as f:
        pyramid = pickle.load(f)

    filters = pd.DataFrame(pyramid.filters)

    # take screen_size_cm as width, calculate pixel size in cm
    vdim_px, hdim_px, _ = pyramid.definition.stimulus_vht_fov
    px_size_cm = screen_size_cm / hdim_px

    # pymoten encodes coordinates as distances from top left corner relative to pixel **height**
    # recode to absolute distances from center in cm
    centerv_px = vdim_px * filters.loc[:, 'centerv'] - vdim_px/2
    centerh_px = vdim_px * filters.loc[:, 'centerh'] - hdim_px/2

    centerv_cm = centerv_px * px_size_cm
    centerh_cm = centerh_px * px_size_cm

    # convert dims to degrees of visual angle
    centerv_deg = 2.0 * \
        np.degrees(np.arctan(centerv_cm / (2.0*screen_distance_cm)))
    centerh_deg = 2.0 * \
        np.degrees(np.arctan(centerh_cm / (2.0*screen_distance_cm)))

    complex_coords = centerh_deg + centerv_deg * 1j
    filters['ecc'] = np.abs(complex_coords)
    filters['polar'] = np.angle(complex_coords)

    return filters


def get_volume_shape(segmentation, subject):
    amplitudes = moten_fit.load_amplitudes(
        subject, session=1, segmentation=segmentation)
    return amplitudes.shape[:-1]


def select_voxel_params(filters, betas):
    # for each voxel, choose filter with highest beta value
    betas = np.nan_to_num(betas)

    best_filter_idx = np.nanargmax(betas, axis=0)
    best_params = filters.iloc[best_filter_idx]

    print(best_params.shape)
    return best_params.values


def concat_slices(filters, segmentation, aot_condition, n_voxels, n_slices, subject):
    # initialize empty array
    n_params = filters.shape[1]
    best_params = np.zeros((n_voxels, n_params))
    r2 = np.zeros((n_voxels, ))

    # retrieve params
    for slice_nr in range(n_slices):
        vertices = load_vertices(slice_nr, subject)
        if vertices is None:
            continue

        betas = load_results(segmentation, aot_condition,
                             slice_nr, subject, 'betas')
        print(best_params.shape, vertices.shape)
        best_params[vertices] = select_voxel_params(filters, betas)

        r2[vertices] = load_results(
            segmentation, aot_condition, slice_nr, subject, 'r2')

    return best_params, r2


def main(subject, segmentation, aot_condition, n_slices, screen_size_cm, screen_distance_cm):
    out_path = DIR_DERIVATIVES / \
        f'sub-{str(subject).zfill(3)}' / \
        f'moten_analysis_{segmentation}_{aot_condition}'
    os.makedirs(out_path, exist_ok=True)

    filename_base = f'sub-{str(subject).zfill(3)}'

    filters = load_filters(screen_size_cm, screen_distance_cm)
    param_names = filters.columns
    volume_shape = get_volume_shape(segmentation, subject)
    n_voxels = np.prod(volume_shape)
    params, r2 = concat_slices(filters, segmentation,
                               aot_condition, n_voxels, n_slices, subject)

    # unflatten to volume shape
    params = params.reshape(volume_shape + (-1,))
    r2 = r2.reshape(volume_shape)

    # save as nifti
    metadata_path = DIR_DATA / \
        'sub-002_ses-01_task-AOT_rec-nordicstc_run-1_space-T1w_part-mag_boldref.nii.gz'
    for param_idx, param_label in enumerate(param_names):
        print(
            f'Storing nifti for param {param_label}', flush=True)
        io_utils.save_nifti(
            params[:, :, :, param_idx],
            out_path / f'{filename_base}_{param_label}.nii.gz',
            subject,
            metadata_path=metadata_path)

    print(f'Storing nifti for r2', flush=True)
    io_utils.save_nifti(
        r2,
        out_path / f'{filename_base}_r2.nii.gz',
        subject,
        metadata_path=metadata_path)


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
    screen_size_cm = config['prf']['fit']['grid']['screen_size_cm']
    screen_distance_cm = config['prf']['fit']['grid']['screen_distance_cm']

    main(subject, segmentation, aot_condition,
         n_slices, screen_size_cm, screen_distance_cm)
