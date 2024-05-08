import os
import numpy as np
from pathlib import Path
from pprint import pprint
import yaml
import pandas as pd
import aot


def get_tr_sequence(run, shift=0):
    tr_sequence = yaml.load(open(
        DIR_SETTINGS / f'experiment_settings_sub_{SUBJECT}_ses_{SESSION}_run_{run}.yml'), Loader=yaml.FullLoader)['stimuli']['movie_files']

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


def get_video_conditions():
    video_conditions = pd.read_csv(DIR_VIDEOS / 'video_conditions.tsv',
                                   sep='\t', header=None, names=['video_condition', 'feature_index'])
    video_conditions['video_condition_stripped'] = video_conditions['video_condition'].str.slice(
        0, 4)
    # fix label for blank trials
    video_conditions.loc[0, 'video_condition_stripped'] = 'blank'
    # recode to account for features
    video_conditions['feature_index'] = [-1] + \
        (video_conditions.loc[1:, 'video_condition_stripped'].astype(
            int) - 1).to_list()
    video_conditions['is_reverse'] = video_conditions['video_condition'].str.endswith(
        '_rv.mp4').astype(int)
    # dynamically updated; keeps track of first vs second presentation of each video
    video_conditions['presentation_index'] = 0

    return video_conditions


def create_run_design_matrices(run, tr_sequence, video_conditions):
    # initialize design matrix with shape: n_TRs, n_features (n_video_ids + 1 placeholder -- later filled with is_reverse or is_repeat)
    design_matrix = np.zeros(
        (len(tr_sequence), len(video_conditions['feature_index']) + 1), dtype=int)
    is_reverse = np.zeros(len(tr_sequence), dtype=int)
    is_repeat = np.zeros(len(tr_sequence), dtype=int)

    for tr_index, tr_label in enumerate(tr_sequence):
        if tr_label != BLANK:
            # look up entry in video_conditions df
            video_condition = video_conditions[video_conditions.video_condition == tr_label].squeeze(
            )
            # one-hot encode video condition
            design_matrix[tr_index][video_condition['feature_index']] = 1
            is_reverse[tr_index] = video_condition['is_reverse']
            is_repeat[tr_index] = video_condition['presentation_index']
            video_conditions.at[video_condition.name,
                                'presentation_index'] += 1

            assert video_conditions.at[video_condition.name,
                                       'presentation_index'] <= 2, f'tr_label: {tr_label}, series: {video_condition}, actual: {video_conditions.at[video_condition.name, "presentation_index"]}'

    # create aot and control design matrix from base design matrix -- difference in final feature
    n_erroneous_vals = (design_matrix[:, -1] != 0).sum()
    # ensure placeholder feature remained empty
    assert n_erroneous_vals == 0, f'placeholder erroneously populated with {n_erroneous_vals} values'
    design_matrix_aot = design_matrix.copy()
    design_matrix_aot[:, -1] = is_reverse
    design_matrix_control = design_matrix.copy()
    design_matrix_control[:, -1] = is_repeat

    # save design matrices
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    np.save(DIR_OUTPUT /
            f'sub_{SUBJECT}_ses_{SESSION}_run_{run}_design_matrix_aot.npy', design_matrix_aot)
    np.save(DIR_OUTPUT /
            f'sub_{SUBJECT}_ses_{SESSION}_run_{run}_design_matrix_control.npy', design_matrix_control)

    return design_matrix_aot, design_matrix_control, video_conditions


def create_session_design_matrices(video_conditions):
    core_settings = yaml.load(
        open(DIR_EXPERIMENT / 'core_exp_settings.yml'), Loader=yaml.FullLoader)
    design_matrices_aot = []
    design_matrices_control = []
    for run_index in range(core_settings['various']['run_number']):
        run = str(run_index + 1).zfill(2)
        tr_sequence = get_tr_sequence(run)
        design_matrix_aot, design_matrix_control, video_conditions = create_run_design_matrices(run,
                                                                                                tr_sequence, video_conditions)
        design_matrices_aot.append(design_matrix_aot)
        design_matrices_control.append(design_matrix_control)
    return design_matrices_aot, design_matrices_control


SUBJECT = '01'
SESSION = '03'
BLANK = -1  # encoding of blank trials

DIR_BASE = Path(aot.__path__[0])
DIR_EXPERIMENT = DIR_BASE / 'experiment'
DIR_VIDEOS = DIR_BASE / 'data/videos'
DIR_SETTINGS = DIR_BASE / 'data/experiment/settings/main'
DIR_OUTPUT = Path('./data')
DIR_OUTPUT = DIR_OUTPUT / \
    ('sub-' + SUBJECT) / \
    ('ses-' + SESSION) / \
    'design_matrices'

if __name__ == '__main__':
    video_conditions = get_video_conditions()
    create_session_design_matrices(video_conditions)
