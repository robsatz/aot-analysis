import os
import numpy as np
from pathlib import Path
import argparse
import yaml


def select_voxels(data, n_slices, slice_nr, subject):

    # flatten to n_voxels x n_timepoints
    data = data.reshape(-1, data.shape[-1])
    print('Data shape at import:', data.shape)

    # filter nan values
    brain_mask = ~np.isnan(data).all(axis=1)
    print(brain_mask.shape)
    brain_vertices = np.where(brain_mask)[0]
    print('Number of valid vertices:', brain_vertices.shape)

    # get slice
    slice_vertices = np.array_split(brain_vertices, n_slices, axis=0)[slice_nr]
    data = data[slice_vertices]
    print('Shape of slice:', data.shape)

    # save vertices
    subject = str(subject).zfill(3)
    out_path = DIR_DERIVATIVES / f'sub-{subject}' / 'prf_slices'
    os.makedirs(out_path, exist_ok=True)
    np.save(
        out_path / f'vertices_slice_{str(slice_nr).zfill(4)}.npy', slice_vertices)

    return data


def concat_slices(data, subject):
    subject = str(subject).zfill(3)
    out_path = DIR_DERIVATIVES / f'sub-{subject}' / 'prf_slices'
    slices = [f for f in os.listdir(out_path) if f.endswith('.npy')]
    slices.sort()
    print(slices)
    data = np.concatenate([np.load(out_path / f) for f in slices], axis=0)
    print('Data shape after concatenation:', data.shape)
    return data


config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
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
        "-slice", "--slice_nr", type=int, default=1)
    parser.add_argument(
        "-n_jobs", "--n_jobs", type=int, default=2)
    args = parser.parse_args()

    subject = args.subject
    slice_nr = args.slice_nr
    n_jobs = args.n_jobs

    concat_slices(
        subject, slice_nr, n_jobs, params)
