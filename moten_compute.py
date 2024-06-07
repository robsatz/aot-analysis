import moten
from pathlib import Path
import os
import pickle
import argparse
import numpy as np
import yaml


def compute_motion_energy(pyramid, input_filename, output_filename):
    luminance_images = moten.io.video2luminance(
        str(DIR_STIMULI / input_filename), size=SIZE)
    nimages, vdim, hdim = luminance_images.shape
    print('Shape of videos (spatially downsampled):', nimages, vdim, hdim)

    features = pyramid.project_stimulus(luminance_images)
    np.save(output_filename, features)
    return features


def get_motion_energy(pyramid, video_filename):
    video_id = video_filename[:4]  # e.g. 0001
    video_features_filename = DIR_MOTION_ENERGY / \
        'video_features' / (video_id + '.npy')
    if os.path.exists(video_filename) and not args.recompute:
        print(f'Loading existing features for video {video_id}')
        video_features = np.load(video_features_filename)
    else:
        video_features = compute_motion_energy(
            pyramid, video_filename, video_features_filename)

    video_avg = np.mean(video_features, axis=0)
    video_idx = (int(video_id[:4]) - 1)
    features[video_idx, :] = video_avg


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_STIMULI = Path(core_settings['paths']['stimuli'])
DIR_BASE = Path('.')
DIR_MOTION_ENERGY = DIR_BASE / core_settings['paths']['motion_energy']
DIR_MOTION_ENERGY_VIDEO_FEATURES = DIR_MOTION_ENERGY / 'video_features'
SIZE = core_settings['motion_energy']['size']
FPS = core_settings['motion_energy']['fps']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-re", "--recompute", action=argparse.BooleanOptionalAction,
                        help="Recompute features.")
    args = parser.parse_args()

    os.makedirs(DIR_MOTION_ENERGY_VIDEO_FEATURES, exist_ok=True)
    stimuli_files = os.listdir(DIR_STIMULI)
    video_files = [file for file in stimuli_files if file.endswith('_fw.mp4')]

    pyramid = moten.get_default_pyramid(vhsize=SIZE, fps=FPS)
    with open(DIR_MOTION_ENERGY / 'pyramid.pkl', 'wb') as f:
        pickle.dump(pyramid, f)

    # init array to store features for all videos
    features = np.zeros((len(video_files), pyramid.nfilters))

    for i, video_file in enumerate(video_files):
        print(f'Processing video {video_file} ({i+1}/{len(video_files)})')
        get_motion_energy(pyramid, video_file)

    # save features for all videos
    np.save(DIR_MOTION_ENERGY / 'motion_energy_all.npy', features)
