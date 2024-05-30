import moten
from pathlib import Path
import os
import pickle


def compute_motion_energy(video_file):
    luminance_images = moten.io.video2luminance(
        str(DIR_VIDEOS / video_file), size=(192, 108))
    nimages, vdim, hdim = luminance_images.shape
    print(nimages, vdim, hdim)

    pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=FPS)
    features = pyramid.project_stimulus(luminance_images)
    print(features.shape)

    return features


DIR_VIDEOS = Path(
    '/tank/shared/2024/visual/AOT/derivatives/stimuli/rescaled_final')
DIR_OUTPUT = Path('./motion_energy_features')

FPS = 24

if __name__ == '__main__':
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    video_files = os.listdir(DIR_VIDEOS)
    for video_file in video_files:
        video_id, video_ext = os.path.splitext(video_file)
        if video_ext == ".mp4":
            output_file = DIR_OUTPUT / f"{video_id}.pkl"
            if output_file.exists():
                print("Skipping - features already computed:", video_file)
            else:
                print("Computing features for:", video_file)
                features = compute_motion_energy(video_file)
                with open(output_file, "wb") as f:
                    pickle.dump(features, f)
