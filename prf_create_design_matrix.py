import h5py
import yaml
from pathlib import Path
import os
import pandas as pd
import pandasql as psql
import imageio
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_data(subject, run):
    events = pd.read_csv(
        DIR_INPUT / f'sub-{subject}_run-{str(run).zfill(2)}_task-pRF_events.tsv', sep='\t')
    seq_timing = h5py.File(
        DIR_INPUT / f'sub-{subject}_run-{str(run).zfill(2)}_task-pRF_seq_timing.h5', 'r')
    apertures = h5py.File(
        DIR_INPUT / f'sub-{subject}_run-{str(run).zfill(2)}_task-pRF_apertures.h5', 'r')
    return events, seq_timing, apertures


def get_pulses(events):
    trs = events[events.event_type == 'pulse']
    first_tr = trs[trs['phase'] == 1].iloc[0]['onset']

    trs['onset_shifted'] = trs['onset'] - first_tr
    trs = trs.reset_index(drop=True)
    trs['tr_nr'] = trs.index + 1
    trs['next_onset_shifted'] = trs['onset_shifted'].shift(
        -1, fill_value=float('inf'))
    return trs


def get_aperture_timing(seq_timing):
    headers = ['seq_index', 'expected_time', 'empirical_time']
    aperture_timing = []
    for trial_id in seq_timing.keys():
        trial_seq = pd.DataFrame(
            seq_timing[trial_id]['apertures']['block0_values'], columns=headers)
        trial_seq['trial_nr'] = int(trial_id[-3:])
        aperture_timing.append(trial_seq)
    return pd.concat(aperture_timing)


def resample(data, temp_res, time_col, agg_cols):
    # sample to temporal resolution
    data['tr_index'] = np.floor(
        data[time_col] // temp_res).astype(int)
    return data.groupby(
        'tr_index')[agg_cols].mean().reset_index()


# def join_apertures_to_pulses(trs, aperture_timing):
#     query = """
#     SELECT
#         trs.tr_nr as tr_nr,
#         trs.trial_nr as trs_trial_nr,
#         trs.onset_shifted as trs_onset_shifted,
#         aperture_timing.trial_nr as trial_seq_trial_nr,
#         aperture_timing.seq_index as trial_seq_index,
#         aperture_timing.empirical_time as trial_seq_empirical_time
#     FROM
#         trs
#     LEFT JOIN
#         aperture_timing
#     ON
#         aperture_timing.empirical_time >= trs.onset_shifted
#         AND aperture_timing.empirical_time < trs.next_onset_shifted
#     """
#     tr_aperture_indices = psql.sqldf(query, locals())
#     tr_aperture_indices = tr_aperture_indices.groupby(
#         'tr_nr').mean('trial_seq_index')
#     return tr_aperture_indices


def map_apertures_to_trials(apertures, trial_list, frames):
    condition_apertures = {condition.split(
        '_')[2]: frames for condition, frames in apertures.items()}

    # strip blank trials
    print(trial_list)
    bar_directions = [
        trial for trial in trial_list if trial != -1]
    trial_nrs = frames[~frames.seq_index.isna()
                       ].trial_nr.unique()

    # assign aperture to trial
    map_trial_nr_condition = {}
    for i, trial_nr in enumerate(trial_nrs):
        print(i, trial_nr)
        bar_direction = str(bar_directions[i])
        map_trial_nr_condition[trial_nr] = condition_apertures[bar_direction]

    return map_trial_nr_condition


def create_design_matrix(frames, map_trial_nr_condition, vhsize):

    dm = np.zeros((len(frames), vhsize[0], vhsize[1]))

    for i in range(len(frames)):
        frame = frames.iloc[i]
        trial_nr = int(frame['trial_nr'])
        seq_index = frame['seq_index']
        if np.isnan(seq_index):
            dm[i, :, :] = np.zeros((512, 512))  # blank frame
        else:
            # average proportionally over nearest frames
            prop_next_frame = seq_index % 1
            seq_index = int(seq_index)  # rounds down

            current_aperture = map_trial_nr_condition[trial_nr][seq_index] * (
                1-prop_next_frame)
            if seq_index + 1 >= len(map_trial_nr_condition[trial_nr]):
                next_aperture = np.zeros((512, 512))
            else:
                next_aperture = map_trial_nr_condition[trial_nr][seq_index +
                                                                 1] * prop_next_frame
            dm[i, :, :] = current_aperture + next_aperture
    return dm


def save_design_matrix(dm, subject, run):
    out_path = DIR_OUTPUT /\
        f'sub-{subject}_run-{run}_design_matrix.npy'
    np.save(out_path, dm)
    print('Design matrix saved at:', out_path)


def create_gif(design_matrix, tr, subject, run):
    num_frames = design_matrix.shape[0]
    print(design_matrix.shape)

    images = []
    for i in range(num_frames):
        fig, ax = plt.subplots()
        ax.imshow(design_matrix[i, :, :], origin='lower')
        ax.set_title('Full run design matrix')
        ax.axis('off')

        # Save the frame as an image in memory
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        plt.close(fig)  # Close the figure to avoid memory issues

    # Save the sequence as a GIF
    gif_filename = DIR_OUTPUT / \
        f"sub-{str(subject).zfill(2)}_run-{
            str(run).zfill(2)}_design_matrix.gif"
    imageio.mimsave(gif_filename, images, fps=1/tr)  # Adjust fps as needed
    print(f"Saved GIF: {gif_filename}")


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
params = core_settings['prf_experiment']
paths = core_settings['paths']['prf_experiment']
DIR_BASE = Path(paths['base'])
DIR_INPUT = Path(DIR_BASE) / paths['input']
DIR_OUTPUT = Path(DIR_BASE) / paths['output']

exp_settings = yaml.load(
    open(DIR_BASE / 'settings.yml'), Loader=yaml.FullLoader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()

    vhsize = params['vhsize']
    fps = params['fps']
    tr = params['tr']
    bar_directions = exp_settings['stimuli']['bar_directions']

    for run in range(3, 11):
        events, seq_timing, apertures = load_data(args.subject, run)
        # trs = get_pulses(events)
        aperture_timing = get_aperture_timing(seq_timing)
        aperture_timing = resample(aperture_timing, tr)
        # tr_aperture_indices = get_tr_aperture_indices(trs, aperture_timing)
        map_trial_nr_condition = map_apertures_to_trials(
            apertures, bar_directions, tr_aperture_indices)
        dm = create_design_matrix(
            tr_aperture_indices, map_trial_nr_condition, vhsize)
        create_gif(dm, args.subject)
