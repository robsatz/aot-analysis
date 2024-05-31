import moten
from pathlib import Path
import os
import pickle
import argparse
import numpy as np
import yaml


def compute_motion_energy(pyramid, video_file):
    luminance_images = moten.io.video2luminance(
        str(DIR_STIMULI / video_file), size=SIZE)
    nimages, vdim, hdim = luminance_images.shape
    print('Shape of videos (spatially downsampled):', nimages, vdim, hdim)

    features = pyramid.project_stimulus(luminance_images)
    video_avg = np.mean(features, axis=0)

    return video_avg


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_STIMULI = Path(core_settings['paths']['stimuli'])
DIR_BASE = Path('./')
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
    # init array to store features for all videos
    features = np.zeros((len(video_files), pyramid.nfilters))

    for i, video_file in enumerate(video_files):
        video_id, _ = os.path.splitext(video_file)
        output_file = DIR_MOTION_ENERGY / \
            'video_features' / (video_id + '.npy')
        if os.path.exists(output_file) and not args.recompute:
            print(f'Loading existing features for {video_file}')
            video_features = np.load(output_file)
        else:
            video_features = compute_motion_energy(pyramid, video_file)
            np.save(output_file, video_features)

        is_reverse = (video_id[5:7] == 'rv')
        video_idx = (int(video_id[:4]) - 1) + int(is_reverse)
        features[video_idx, :] = video_features

    # save features for all videos
    np.save(DIR_MOTION_ENERGY / 'motion_energy_all.npy', features)
