import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle

from aot_analysis import io_utils
from aot_analysis.motion_energy import fit as moten_fit


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
    filters['centerv'] = 2.0 * \
        np.degrees(np.arctan(centerv_cm / (2.0*screen_distance_cm)))
    filters['centerh'] = 2.0 * \
        np.degrees(np.arctan(centerh_cm / (2.0*screen_distance_cm)))

    # descriptive stats: nanmin and nanmax
    print(
        f'x_deg: {np.nanmin(filters["centerh"])} - {np.nanmax(filters["centerh"])}')
    print(
        f'y_deg: {np.nanmin(filters["centerv"])} - {np.nanmax(filters["centerv"])}')

    return filters


def cart2polar(x_deg, y_deg):
    # descriptive stats: nanmin and nanmax
    print(
        f'x_deg: {np.nanmin(x_deg)} - {np.nanmax(x_deg)}')
    print(
        f'y_deg: {np.nanmin(y_deg)} - {np.nanmax(y_deg)}')

    # assumes coordinates in degrees of visual angle
    ecc = np.sqrt(x_deg**2+y_deg**2)
    polar = np.angle(x_deg+y_deg*1j)
    # descriptive stats: nanmin and nanmax
    print(f'ecc: {np.nanmin(ecc)} - {np.nanmax(ecc)}')
    print(f'polar: {np.nanmin(polar)}'
          + f'- {np.nanmax(polar)}')
    return pd.DataFrame({'ecc': ecc, 'polar': polar})


def get_volume_shape(segmentation, subject):
    amplitudes = moten_fit.load_amplitudes(
        subject, session=1, segmentation=segmentation)
    return amplitudes.shape[:-1]


def average_voxel_params(filters, betas):
    filters = filters.values

    # option 1: select top N betas
    # selected_betas = np.zeros_like(betas)
    # n = 50
    # top_betas = np.argpartition(betas, -n, axis=0)[-n:]
    # selected_betas[top_betas, :] = betas[top_betas, :]

    # # option 2: select all positive betas
    # selected_betas = np.where(betas > 0, betas, 0)

    # option 3: select all betas
    selected_betas = np.nan_to_num(betas)

    # descriptive stats: min and max
    print(
        f'selected_betas: {np.min(selected_betas)} - {np.max(selected_betas)}')

    # convert betas to weights
    betas_sum = np.sum(selected_betas, axis=0)
    weights = selected_betas / betas_sum

    # descriptive stats: min and max
    print(f'weights: {np.min(weights)} - {np.max(weights)}')

    # calculate weighted average via matmul
    avg_params = weights.T @ filters

    # descriptive stats: nanmin and nanmax
    print(f'filters: {np.nanmin(filters)} - {np.nanmax(filters)}')
    print(f'avg_params: {np.nanmin(avg_params)} - {np.nanmax(avg_params)}')

    print(avg_params.shape, flush=True)
    return avg_params


def concat_slices(filters, segmentation, aot_condition, n_voxels, n_slices, subject):
    # initialize empty array
    n_params = filters.shape[1]
    avg_params = np.zeros((n_voxels, n_params))
    r2 = np.zeros((n_voxels, ))

    # retrieve params
    for slice_nr in range(n_slices):
        print(f'Loading params for slice {slice_nr}', flush=True)
        vertices = load_vertices(slice_nr, subject)
        if vertices is None:
            continue

        betas = load_results(segmentation, aot_condition,
                             slice_nr, subject, 'betas')
        r2[vertices] = load_results(
            segmentation, aot_condition, slice_nr, subject, 'r2')
        avg_params[vertices] = average_voxel_params(filters, betas)
    print(f'Concatenation complete - R2 max: {np.nanmax(r2)}')
    return avg_params, r2


def main(subject, segmentation, aot_condition, n_slices, rsq_threshold, screen_size_cm, screen_distance_cm):
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

    # save all params as csv
    df = pd.DataFrame(params, columns=param_names)
    df[['ecc', 'polar']] = cart2polar(df['centerh'], df['centerv'])
    df['R2'] = r2
    print('Dataframe R2 max:', df.R2.max())
    filepath = out_path / f'{filename_base}_params.csv'
    df.to_csv(filepath)
    print(f'Saving csv to {filepath}', flush=True)

    # filter out low-rsq vertices
    filter_vertices = np.where(r2 < rsq_threshold)
    params[filter_vertices] = 0.0

    # unflatten to volume shape
    params = params.reshape(volume_shape + (-1,))
    r2 = r2.reshape(volume_shape)

    # save niftis
    metadata_path = DIR_DATA / \
        'sub-002_ses-01_task-AOT_rec-nordicstc_run-1_space-T1w_part-mag_boldref.nii.gz'
    for param_idx, param_label in enumerate(param_names):
        print(
            f'Saving nifti for param {param_label}', flush=True)
        io_utils.save_nifti(
            params[:, :, :, param_idx],
            out_path / f'{filename_base}_{param_label}.nii.gz',
            subject,
            metadata_path=metadata_path)

    print(f'Saving nifti for r2', flush=True)
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

    fit_settings = config['motion_energy']['fit']
    segmentation = fit_settings['segmentation']
    aot_condition = fit_settings['aot_condition']
    n_slices = fit_settings['n_slices']

    rsq_threshold = config['motion_energy']['viz']['rsq_threshold']

    screen_size_cm = config['prf']['fit']['grid']['screen_size_cm']
    screen_distance_cm = config['prf']['fit']['grid']['screen_distance_cm']

    main(subject, segmentation, aot_condition,
         n_slices, rsq_threshold, screen_size_cm, screen_distance_cm)
