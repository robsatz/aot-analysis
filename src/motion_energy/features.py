import moten
from pathlib import Path
import os
import pickle
import argparse
import numpy as np
import yaml
import os

from src import io_utils


def compute_motion_energy(pyramid, input_filename, output_filename):
    luminance_images = moten.io.video2luminance(
        str(DIR_STIMULI / input_filename), size=SIZE)
    nimages, vdim, hdim = luminance_images.shape
    print('Shape of videos (spatially downsampled):', nimages, vdim, hdim)

    features = pyramid.project_stimulus(luminance_images)
    np.save(output_filename, features)
    return features


def label2idx(label):
    video_id = int(label[:4])
    is_reverse = (label[-2:] == 'rv')
    video_idx = (video_id - 1) * 2 + int(is_reverse)
    return video_idx


def get_video_average(pyramid, video_filename, recompute):
    video_condition, _ = os.path.splitext(video_filename)  # e.g. 0001_fw
    video_features_filename = DIR_MOTION_ENERGY / \
        'video_features' / (video_condition + '.npy')
    if os.path.exists(video_features_filename) and not recompute:
        print(f'Loading existing features for video {video_condition}')
        video_features = np.load(video_features_filename)
    else:
        video_features = compute_motion_energy(
            pyramid, video_filename, video_features_filename)
    video_avg = np.mean(video_features, axis=0)

    video_idx = label2idx(video_condition)
    return video_idx, video_avg


core_settings = io_utils.load_config()
DIR_STIMULI = Path(core_settings['paths']['stimuli'])
DIR_BASE = Path('.')
DIR_MOTION_ENERGY = DIR_BASE / core_settings['paths']['motion_energy']
DIR_MOTION_ENERGY_VIDEO_FEATURES = DIR_MOTION_ENERGY / 'video_features'
SIZE = core_settings['motion_energy']['design_matrix']['vhsize']
FPS = core_settings['motion_energy']['design_matrix']['fps']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-re", "--recompute", action="store_true", default=False,
                        help="Recompute features.")
    args = parser.parse_args()

    os.makedirs(DIR_MOTION_ENERGY_VIDEO_FEATURES, exist_ok=True)
    video_files = sorted(os.listdir(DIR_STIMULI))

    pyramid = moten.get_default_pyramid(vhsize=SIZE, fps=FPS)
    with open(DIR_MOTION_ENERGY / 'pyramid.pkl', 'wb') as f:
        pickle.dump(pyramid, f)

    # init array to store features for all videos
    features = np.zeros((len(video_files), pyramid.nfilters))

    for i, video_file in enumerate(video_files):
        print(f'Processing video {video_file} ({i+1}/{len(video_files)})')
        video_idx, video_avg = get_video_average(
            pyramid, video_file, args.recompute)
        features[video_idx, :] = video_avg

    # save features for all videos
    np.save(DIR_MOTION_ENERGY / 'motion_energy_all.npy', features)
