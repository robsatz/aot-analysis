import h5py
import yaml
from pathlib import Path
import os
import pandas as pd
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


def resample(data, tr, time_col, aggs, label):
    # sample to temporal resolution
    data.loc[:, 'tr_index'] = np.floor(
        (data[time_col] + .00001  # prevents rounding errors
         ) / tr).astype(int)

    data = data.groupby(
        'tr_index').agg(aggs).reset_index()

    # prevent multilevelcolumn names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            '_'.join(filter(None, col)) for col in data.columns]
    return data


def create_intervals(events, tr):
    pulses = events[events.event_type == 'pulse'].copy()

    # recode onset times to align with aperture onset times
    start_time = pulses[pulses.phase == 1].iloc[0].onset
    end_time = pulses['onset'].iloc[-1] - start_time
    pulses.loc[:, 'onset'] = pulses['onset'] - start_time

    # resample to intervals of tr duration
    pulses = resample(pulses, tr, 'onset', {
                      'trial_nr': 'min', 'onset': 'min'}, 'pulses')
    pulses = pulses.rename(
        columns={'trial_nr_min': 'trial_nr', 'onset': 'pulse_onset'})

    # create full sequence of intervals of tr duration
    start_tr_index = pulses['tr_index'].iloc[0]  # can be negative
    n_intervals = end_time // tr + 1
    print(start_tr_index, n_intervals)
    all_intervals = pd.Series(
        np.arange(start_tr_index, n_intervals).astype(int), name='tr_index')

    # merge with pulses dataframe to get trial number and onset times
    all_intervals = pd.merge(all_intervals, pulses, how='left', on='tr_index')
    all_intervals['tr_onset'] = all_intervals['tr_index'] * tr
    # trial_n is nan if no pulse occured in interval - fill with last populated trial_nr
    all_intervals['trial_nr'] = all_intervals['trial_nr'].ffill().astype(int)

    return all_intervals


def get_aperture_timing(seq_timing, tr):
    headers = ['seq_index', 'expected_time', 'empirical_time']
    aperture_timing = []
    for trial_id in seq_timing.keys():
        trial_seq = pd.DataFrame(
            seq_timing[trial_id]['apertures']['block0_values'], columns=headers)
        trial_seq['trial_nr'] = int(trial_id[-3:])
        aperture_timing.append(trial_seq)
    aperture_timing = pd.concat(aperture_timing)
    aggregations = {'seq_index': ['min', 'max'], 'empirical_time': [
        'min', 'max'], 'trial_nr': ['min', 'max']}
    return resample(aperture_timing, tr, 'expected_time', aggregations, 'aperture_timing')


def test_intervals(intervals):
    # check that all intervals are associated with exactly one aperture
    assert intervals.tr_index.duplicated().sum(
    ) == 0, "Some interval indices are duplicated"
    assert intervals.trial_nr.isna().sum(
    ) == 0, "Some intervals are not associated with any trial"
    assert intervals.trial_nr_min.equals(
        intervals.trial_nr_max), "Some intervals are associated with more than one trial"
    aperture_trs = ~intervals.trial_nr_min.isna()
    assert intervals.loc[aperture_trs, 'trial_nr'].equals(
        intervals.loc[aperture_trs, 'trial_nr_min'].astype(int)), "Trial onset times do not align between events and apertures"

    assert intervals.seq_index_min.equals(
        intervals.seq_index_max), "Some intervals are associated with more than one aperture"


def map_apertures_to_trials(apertures, trial_list, frames):
    condition_apertures = {condition.split(
        '_')[2]: frames for condition, frames in apertures.items()}

    # remove blank trials
    bar_directions = [
        trial for trial in trial_list if trial != -1]
    trial_nrs = frames[~frames.seq_index_min.isna()
                       ].trial_nr.unique()

    # assign aperture to trial
    map_trial_nr_condition = {}
    for i, trial_nr in enumerate(trial_nrs):
        bar_direction = str(bar_directions[i])
        map_trial_nr_condition[trial_nr] = condition_apertures[bar_direction]

    return map_trial_nr_condition


def create_design_matrix(intervals, map_trial_nr_condition, tr_output, vhsize):

    # resample index to output tr
    intervals['tr_index'] = np.floor(
        (intervals['tr_onset'] + .00001  # prevents rounding errors
         ) / tr_output).astype(int)
    n_intervals = intervals['tr_index'].max() + 1
    dm = np.zeros((n_intervals, vhsize[0], vhsize[1]))

    for i in range(n_intervals):
        included_intervals = intervals[intervals['tr_index'] == i]

        apertures = []
        for _, frame in included_intervals.iterrows():
            start_seq_index = frame['seq_index_min']
            trial_nr = int(frame['trial_nr'])
            if np.isnan(start_seq_index):
                aperture = np.zeros(vhsize)
            else:
                aperture = map_trial_nr_condition[trial_nr][int(
                    start_seq_index)]
            apertures.append(aperture)

        dm[i, :, :] = np.mean(np.stack(apertures), axis=0)
    return dm


def save_design_matrix(dm, label, subject, run):
    out_path = DIR_OUTPUT /\
        f'sub-{subject}_run-{run}_design_matrix_{label}.npy'
    np.save(out_path, dm)
    print('Design matrix saved at:', out_path)


def create_gif(design_matrix, tr, subject, run):
    num_frames = design_matrix.shape[0]
    print(design_matrix.shape)

    images = []
    for i in range(num_frames):
        fig, ax = plt.subplots()
        ax.imshow(design_matrix[i, :, :], origin='lower')
        ax.set_title(f'Design Matrix for Run {run}')
        ax.axis('off')

        # Save the frame as an image in memory
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        plt.close(fig)  # Close the figure to avoid memory issues

    # Save the sequence as a GIF
    gif_filename = DIR_OUTPUT / \
        (f"sub-{str(subject).zfill(2)}"
         + f"_run-{str(run).zfill(2)}_design_matrix.gif")
    imageio.mimsave(gif_filename, images, fps=1/tr)  # Adjust fps as needed
    print(f"Saved GIF: {gif_filename}")


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
params = core_settings['prf']['design_matrix']
DIR_BASE = Path(core_settings['paths']['prf_experiment']['base'])
DIR_INPUT = DIR_BASE / core_settings['paths']['prf_experiment']['input']
DIR_OUTPUT = DIR_BASE / core_settings['paths']['prf_experiment']['design']

exp_settings = yaml.load(
    open(DIR_BASE / 'settings.yml'), Loader=yaml.FullLoader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()

    vhsize = params['vhsize']
    tr_canonical = params['tr_canonical']
    tr_output = params['tr_output']
    gif_speed = params['gif_speed']
    bar_directions = exp_settings['stimuli']['bar_directions']

    for run in range(1, 11):
        events, seq_timing, apertures = load_data(args.subject, run)

        all_intervals = create_intervals(events, tr_canonical)
        aperture_timing = get_aperture_timing(seq_timing, tr_canonical)
        all_intervals = all_intervals.merge(
            aperture_timing, how='left', on='tr_index')
        test_intervals(all_intervals)
        map_trial_nr_condition = map_apertures_to_trials(
            apertures, bar_directions, all_intervals)

        dm_canonical = create_design_matrix(
            all_intervals, map_trial_nr_condition, tr_canonical, vhsize)
        save_design_matrix(dm_canonical, 'canonical', args.subject, run)

        dm_downsampled = create_design_matrix(
            all_intervals, map_trial_nr_condition, tr_output, vhsize)
        save_design_matrix(dm_downsampled, 'downsampled', args.subject, run)
        create_gif(dm_downsampled, tr_output / gif_speed, args.subject,
                   run)
