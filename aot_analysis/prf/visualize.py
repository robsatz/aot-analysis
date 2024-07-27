import os
import numpy as np
from pathlib import Path
import argparse
from copy import deepcopy

from aot_analysis.prf.parameters import Parameters
from aot_analysis import io_utils
from aot_analysis.prf import fit as prf_fit


def load_vertices(slice_nr, subject):
    try:
        return np.load(DIR_DERIVATIVES / f'sub-{str(subject).zfill(3)}' / 'prf_slices' /
                       f'vertices_slice_{str(slice_nr).zfill(4)}.npy')
    except FileNotFoundError:
        print(f'Slice vertices not found for slice {slice_nr}', flush=True)


def load_params(model_type, stage, slice_nr, subject):
    try:
        return np.load(
            DIR_DERIVATIVES / f'sub-{str(subject).zfill(3)}' / 'prf_fits' / f'sub-{str(subject).zfill(3)}_{str(slice_nr).zfill(5)}_{model_type}_{stage}.npy')
    except FileNotFoundError:
        print(
            f'Params not found for {model_type}_{stage}_fit, slice {slice_nr}', flush=True)
        return False


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
            'iter': deepcopy(norm_params),
            'test': deepcopy(norm_params)}}

    # retrieve params
    for slice_nr in range(n_slices):
        vertices = load_vertices(slice_nr, subject)
        if vertices is None:
            continue
        for model in params_dict.keys():
            for stage in params_dict[model].keys():
                params_dict[model][stage][vertices] = load_params(
                    model, stage, slice_nr, subject)

    return params_dict


def save_csv(params, out_path):
    params.to_csv(out_path)
    print(f'CSV saved to {out_path}', flush=True)


def save_params(params_dict, volume_shape, rsq_threshold, subject):
    # specifying file containing affine transform/header metadata
    metadata_path = DIR_DATA / \
        (f'sub-{str(subject).zfill(3)}'
         + '_ses_pRF_filtered_psc_averageallruns_psc_func.nii.gz')
    out_path = DIR_DERIVATIVES / \
        f'sub-{str(subject).zfill(3)}' / \
        'prf_analysis'
    os.makedirs(out_path, exist_ok=True)

    filename_base = f'sub-{str(subject).zfill(3)}'
    # placeholder_volume = np.zeros(volume_shape)
    search_process_by_param = {}
    # order matters: determines nifti volume order
    for model, stage in [('gauss', 'grid'),
                         ('gauss', 'iter'),
                         ('norm', 'grid'),
                         ('norm', 'iter'),
                         ('norm', 'test')]:
        params = params_dict[model][stage]
        np.save(
            str(out_path / f'{filename_base}_{model}_{stage}_fit.npy'), params)
        params = Parameters(
            params, model=model).to_df()
        save_csv(params, out_path /
                 f'{filename_base}_{model}_{stage}_fit.csv')

        # # create r2 mask for thresholded outputs
        # r2_volume = params['r2'].values.reshape(volume_shape)
        # r2_mask = r2_volume > rsq_threshold

        # iterate over params
        for param in params.columns:
            full_volume = params[param].values.reshape(volume_shape)
            # if param == 'r2':
            # some failing prf fits result in extreme, negative r2 outliers
            # full_volume[full_volume < 0] = 0
            # outputs = [full_volume]
            # else:

            #     outputs = [full_volume]
            # save rsq-thresholded version
            # thresh_volume = deepcopy(placeholder_volume)
            # thresh_volume[r2_mask] = full_volume[r2_mask]
            # outputs.append(thresh_volume)

            outputs = [full_volume]
            if param not in search_process_by_param:
                search_process_by_param[param] = outputs
            else:
                search_process_by_param[param].extend(outputs)

    for param in search_process_by_param.keys():
        print(
            f'Storing nifti for {param}: {len(search_process_by_param[param])} volumes')
        print('Shape of first volume:',
              search_process_by_param[param][0].shape)
        # save nifti as combined volume (showing change throughout search process)
        search_process = np.stack(
            search_process_by_param[param], axis=-1)
        io_utils.save_nifti(
            search_process,
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
    rsq_threshold = config['prf']['viz']['rsq_threshold']

    # retrieve volumetric shape for initialization of arrays
    data, _ = prf_fit.load_data(subject)
    volume_shape = data.shape[:3]
    n_voxels = np.prod(volume_shape)

    params_dict = concat_slices(
        n_voxels, n_slices, subject)

    save_params(params_dict, volume_shape, rsq_threshold, subject)
