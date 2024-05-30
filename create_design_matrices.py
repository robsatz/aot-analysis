import os
import numpy as np
from pathlib import Path
from pprint import pprint
import yaml
import pandas as pd
import argparse


def get_tr_sequence(subject, session, run, shift=0):
    tr_sequence = yaml.load(open(
        DIR_SETTINGS / f'experiment_settings_sub_{subject}_ses_{session}_run_{run}.yml'), Loader=yaml.FullLoader)['stimuli']['movie_files']

    # extend each trial by three blank TRs; recode blanks
    tr_sequence = [
        [x, BLANK, BLANK, BLANK] if x != 'blank' else [BLANK, BLANK, BLANK, BLANK] for x in tr_sequence
    ]

    # flatten the sequence
    tr_sequence = [
        item for sublist in tr_sequence for item in sublist
    ]

    # add 16 0s to the start and end of the list
    tr_sequence = (
        [BLANK] * (16 + shift) + tr_sequence +
        [BLANK] * (16 - shift)
    )

    return tr_sequence


def get_feature_indices(tr_label, prior_videos):
    """
    Given a TR label, assigns an index for the design matrix according to video direction .
    Assumes TR labels of the format '0001_rv.mp4'.
    """
    video_id = int(tr_label[:4])
    is_reverse = (tr_label[5:7] == 'rv')
    is_repeat = (video_id in prior_videos)
    if not is_repeat:
        prior_videos.append(video_id)

    # -1 for zero-indexing, *2 for two conditions per video
    feature_idx_aot = (video_id - 1) * 2 + int(is_reverse)
    feature_idx_control = (video_id - 1) * 2 + int(is_repeat)

    return feature_idx_aot, feature_idx_control, prior_videos


def create_run_design_matrices(tr_sequence, prior_videos, subject, session, run):
    # initialize design matrix with shape: n_TRs, n_features (n_video_ids + 1 placeholder -- later filled with is_reverse or is_repeat)
    n_features = 4358
    design_matrix_aot = np.zeros(
        (len(tr_sequence), n_features), dtype=int)
    design_matrix_control = np.zeros(
        (len(tr_sequence), n_features), dtype=int)

    for tr_index, tr_label in enumerate(tr_sequence):
        if tr_label != BLANK:
            feature_idx_aot, feature_idx_control, prior_videos = get_feature_indices(
                tr_label, prior_videos)
            design_matrix_aot[tr_index][feature_idx_aot] = 1
            design_matrix_control[tr_index][feature_idx_control] = 1

    # # remove unused features from design matrices
    # used_features = np.where(np.sum(design_matrix_aot, axis=0) != 0)[0]
    # design_matrix_aot = design_matrix_aot[:, used_features]
    # design_matrix_control = design_matrix_control[:, used_features]
    return design_matrix_aot, design_matrix_control, prior_videos


def create_session_design_matrices(subject, session):
    core_settings = yaml.load(
        open(DIR_EXPERIMENT / 'core_exp_settings.yml'), Loader=yaml.FullLoader)

    design_matrices_aot = []
    design_matrices_control = []
    prior_videos = []

    # get run design matrices
    n_runs = core_settings['various']['run_number']
    for run_idx in range(n_runs):
        run = str(run_idx + 1).zfill(2)
        tr_sequence = get_tr_sequence(subject, session, run)
        design_matrix_aot, design_matrix_control, prior_videos = create_run_design_matrices(
            tr_sequence, prior_videos, subject, session, run)
        design_matrices_aot.append(design_matrix_aot)
        design_matrices_control.append(design_matrix_control)

    # remove unused features from design matrices
    used_indices_aot = used_indices_control = set()
    for run_idx in range(n_runs):
        used_indices_aot.update(
            set(np.where(np.sum(design_matrices_aot[run_idx], axis=0) != 0)[0]))
        used_indices_control.update(
            set(np.where(np.sum(design_matrices_control[run_idx], axis=0) != 0)[0]))

    for run_idx in range(n_runs):
        design_matrices_aot[run_idx] = design_matrices_aot[run_idx][:, list(
            used_indices_aot)]
        design_matrices_control[run_idx] = design_matrices_control[run_idx][:, list(
            used_indices_control)]

        # save design matrices
        subject = subject.zfill(3)
        output_path = DIR_OUTPUT / \
            f'sub-{subject}' / f'ses-{session}' / 'design_matrices'
        os.makedirs(output_path, exist_ok=True)
        np.save(output_path /
                f'sub-{subject}_ses-{session}_run-{str(run_idx+1).zfill(2)}_design_matrix_aot.npy', design_matrices_aot[run_idx])
        np.save(output_path /
                f'sub-{subject}_ses-{session}_run-{str(run_idx+1).zfill(2)}_design_matrix_control.npy', design_matrices_control[run_idx])

    return design_matrices_aot, design_matrices_control


def create_subject_design_matrices(subject):
    for session_idx in range(10):
        session = str(session_idx + 1).zfill(2)
        create_session_design_matrices(subject, session)


BLANK = -1  # encoding of blank trials

DIR_BASE = Path('arrow_of_time_experiment/aot')
DIR_EXPERIMENT = DIR_BASE / 'experiment'
DIR_VIDEOS = DIR_BASE / 'data/videos'
DIR_SETTINGS = DIR_BASE / 'data/experiment/settings/main'
DIR_OUTPUT = Path('./derivatives')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()
    subject = str(args.subject).zfill(2)
    create_subject_design_matrices(subject)
