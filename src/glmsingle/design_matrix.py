import os
import numpy as np
from pathlib import Path
import yaml
import argparse

from src import io_utils


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


def label2idx(tr_label, prior_videos):
    """
    Given a TR label, assigns an index for the design matrix according to video direction .
    Assumes TR labels of the format '0001_rv.mp4'.
    """
    video_id = int(tr_label[:4])
    is_reverse = (tr_label[5:7] == 'rv')
    is_repeat = (tr_label in prior_videos)
    xor = is_reverse ^ is_repeat
    if not is_repeat:
        prior_videos.append(tr_label)

    # -1 for zero-indexing, * 2 for two conditions per video
    # segment by arrow of time condition
    feature_idx_aot = (video_id - 1) * 2 + int(is_reverse)
    # segment by presentation index
    feature_idx_pres = (video_id - 1) * 2 + int(is_repeat)
    # scrambled, control segmentation
    feature_idx_scram = (video_id - 1) * 2 + int(xor)

    return feature_idx_aot, feature_idx_pres, feature_idx_scram, prior_videos


def idx2label(feature_idx):
    """
    Given a feature index, returns the corresponding label.
    """
    video_id = feature_idx // 2 + 1

    condition = feature_idx % 2
    aot_suffix = {0: '_fw', 1: '_rv'}
    pres_suffix = {0: '_1st', 1: '_2nd'}

    label = str(video_id).zfill(4)
    label_aot = label + aot_suffix[condition]
    label_pres = label + pres_suffix[condition]

    return label_aot, label_pres


def create_run_design_matrices(tr_sequence, n_features, prior_videos):
    # initialize design matrices with shape: number of TRs x number of all possible TR labels (unused columns are later pruned)
    design_matrix_aot = np.zeros(
        (len(tr_sequence), n_features), dtype=int)
    design_matrix_pres = np.zeros(
        (len(tr_sequence), n_features), dtype=int)
    design_matrix_scram = np.zeros(
        (len(tr_sequence), n_features), dtype=int)

    for tr_index, tr_label in enumerate(tr_sequence):
        if tr_label != BLANK:
            feature_idx_aot, feature_idx_pres, feature_idx_scram, prior_videos = label2idx(
                tr_label, prior_videos)
            design_matrix_aot[tr_index][feature_idx_aot] = 1
            design_matrix_pres[tr_index][feature_idx_pres] = 1
            design_matrix_scram[tr_index][feature_idx_scram] = 1

    return design_matrix_aot, design_matrix_pres, design_matrix_scram, prior_videos


def create_session_design_matrices(subject, session, n_features):
    exp_settings = yaml.load(
        open(DIR_EXPERIMENT / 'core_exp_settings.yml'), Loader=yaml.FullLoader)

    design_matrices_aot = []
    design_matrices_pres = []
    design_matrices_scram = []
    prior_videos = []

    # get run design matrices
    n_runs = exp_settings['various']['run_number']
    for run_idx in range(n_runs):
        run = str(run_idx + 1).zfill(2)
        tr_sequence = get_tr_sequence(subject, session, run)
        # prior_videos keeps track of video repeats
        design_matrix_aot, design_matrix_pres, design_matrix_scram, prior_videos = create_run_design_matrices(
            tr_sequence, n_features, prior_videos)
        design_matrices_aot.append(design_matrix_aot)
        design_matrices_pres.append(design_matrix_pres)
        design_matrices_scram.append(design_matrix_scram)

   # get indices of features used in session
    used_indices_aot = used_indices_pres = used_indices_scram = set()
    for run_idx in range(n_runs):
        used_indices_aot.update(
            set(np.where(np.sum(design_matrices_aot[run_idx], axis=0) != 0)[0]))
        used_indices_pres.update(
            set(np.where(np.sum(design_matrices_pres[run_idx], axis=0) != 0)[0]))
        used_indices_scram.update(
            set(np.where(np.sum(design_matrices_scram[run_idx], axis=0) != 0)[0]))

    # remap indices to labels - assumes alphabetical order (i.e., '0001_fw' < '0001_rv' and '0001_1st' < '0001_2nd')
    mapping_aot = {idx: idx2label(feat_idx)[0]
                   for idx, feat_idx in enumerate(sorted(used_indices_aot))}
    mapping_pres = {idx: idx2label(feat_idx)[1]
                    for idx, feat_idx in enumerate(sorted(used_indices_pres))}
    mapping_scram = {idx: idx2label(feat_idx)[1]
                     for idx, feat_idx in enumerate(sorted(used_indices_scram))}

    # save as yaml
    subject = subject.zfill(3)
    output_path = DIR_OUTPUT / \
        f'sub-{subject}' / f'ses-{session}' / 'design_matrices'
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / 'feature_mapping_aot.yml', 'w') as f:
        yaml.dump(mapping_aot, f)
    with open(output_path / 'feature_mapping_pres.yml', 'w') as f:
        yaml.dump(mapping_pres, f)
    with open(output_path / 'feature_mapping_scram.yml', 'w') as f:
        yaml.dump(mapping_scram, f)

    # remove unused features from design matrices
    for run_idx in range(n_runs):
        design_matrices_aot[run_idx] = design_matrices_aot[run_idx][:, list(
            used_indices_aot)]
        design_matrices_pres[run_idx] = design_matrices_pres[run_idx][:, list(
            used_indices_pres)]
        design_matrices_scram[run_idx] = design_matrices_scram[run_idx][:, list(
            used_indices_scram)]

        # save design matrices
        np.save(output_path /
                f'sub-{subject}_ses-{session}_run-{str(run_idx+1).zfill(2)}_design_matrix_aot.npy', design_matrices_aot[run_idx])
        np.save(output_path /
                f'sub-{subject}_ses-{session}_run-{str(run_idx+1).zfill(2)}_design_matrix_pres.npy', design_matrices_pres[run_idx])
        np.save(output_path /
                f'sub-{subject}_ses-{session}_run-{str(run_idx+1).zfill(2)}_design_matrix_scram.npy', design_matrices_scram[run_idx])

    return design_matrices_aot, design_matrices_pres, design_matrices_scram


def create_subject_design_matrices(subject, n_features):
    for session_idx in range(10):
        session = str(session_idx + 1).zfill(2)
        create_session_design_matrices(subject, session, n_features)


BLANK = -1  # encoding of blank trials

config = io_utils.load_config()
DIR_BASE = Path(config['paths']['aot_experiment']['base'])
DIR_EXPERIMENT = DIR_BASE / 'aot/experiment'
DIR_VIDEOS = DIR_BASE / 'aot/data/videos'
DIR_SETTINGS = DIR_BASE / 'aot/data/experiment/settings/main'
DIR_OUTPUT = Path(config['paths']['derivatives'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()
    subject = str(args.subject).zfill(2)
    n_features = config['glmsingle']['design_matrix']['n_features']
    create_subject_design_matrices(subject, n_features)
