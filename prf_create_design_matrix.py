import h5py
import yaml
from pathlib import Path
import os
import pandas as pd
import imageio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import zoom
import prf_utils


def load_data(subject, run):
    events = pd.read_csv(
        DIR_INPUT / f'sub-{subject}_run-{str(run).zfill(2)}_task-pRF_events.tsv', sep='\t')
    seq_timing = h5py.File(
        DIR_INPUT / f'sub-{subject}_run-{str(run).zfill(2)}_task-pRF_seq_timing.h5', 'r')
    apertures = h5py.File(
        DIR_INPUT / f'sub-{subject}_run-{str(run).zfill(2)}_task-pRF_apertures.h5', 'r')
    return events, seq_timing, apertures


def resample(timepoints, tr, time_col, aggs):
    timepoints = timepoints.sort_values('onset')

    # sample to temporal resolution
    timepoints.loc[:, 'tr_index'] = np.floor(
        (timepoints[time_col]) / tr).astype(int)

    timepoints = timepoints.groupby(
        'tr_index').agg(aggs).reset_index()

    # prevent multilevelcolumn names
    if isinstance(timepoints.columns, pd.MultiIndex):
        timepoints.columns = [
            '_'.join(filter(None, col)) for col in timepoints.columns]
    return timepoints


def prevent_cutoff(experiment_events, timepoints, tr):
    # prevent final tr being cut off by upsampling
    # assumes last tr is blank
    tr_acquisition = np.median(np.diff(experiment_events['onset']))
    duration = len(experiment_events) * tr_acquisition
    final_tr_index = duration // tr
    # get last timepoint as dataframe to facilitate appending to timepoints
    final_timepoint = timepoints.iloc[[-1]].copy()
    final_timepoint.loc[:, 'tr_index'] = final_tr_index
    timepoints = pd.concat([timepoints, final_timepoint]
                           )
    return timepoints


def resize(dm, vhsize):
    print(f'Resizing frames from {dm.shape[1:]} to {vhsize}')
    step_vertical = int(dm.shape[1] / vhsize[0])
    step_horizontal = int(dm.shape[2] / vhsize[1])
    shift_vertical = int(step_vertical / 2)
    shift_horizontal = int(step_horizontal / 2)

    return dm[:, shift_vertical::step_vertical, shift_horizontal::step_horizontal]


def get_experiment_events(events):

    pulses = events[events.event_type == 'pulse'].copy()
    pulses.loc[:,
               'bar_direction'] = pulses['bar_direction'].fillna(BLANK)

    # recode onsets to start at 0
    start = pulses['onset'].min()
    pulses.loc[:, 'onset'] -= start
    end = pulses['onset'].max()

    # detect delay to first trial: later used as reference point for aperture events
    aperture_start = pulses.loc[pulses.phase == 1, 'onset'].min()

    # # resample pulses to tr
    # aggregations = {'trial_nr': 'min', 'onset': 'min', 'bar_direction': 'min'}
    # experiment_events = resample(pulses, tr, 'onset', aggregations)

    return pulses[['trial_nr', 'onset', 'bar_direction']], aperture_start, end


def get_aperture_events(seq_timing, delay):
    # concatenate aperture events for all bar directions
    aperture_events = []
    for trial_id in seq_timing.keys():
        trial_seq = pd.DataFrame(
            seq_timing[trial_id]['apertures']['block0_values'], columns=['seq_index', 'expected_time', 'empirical_time'])
        trial_seq['trial_nr'] = int(trial_id[-3:])
        aperture_events.append(trial_seq)
    aperture_events = pd.concat(aperture_events)

    # shift onset times to align with events dataset
    aperture_events['onset'] = aperture_events['empirical_time'] + delay

    return aperture_events[['trial_nr', 'seq_index', 'onset']]


def create_timepoints(experiment_events, aperture_events, tr):
    events = pd.concat([experiment_events, aperture_events], axis=0)
    timepoints = resample(events, tr, 'onset', ['min', 'max', 'count'])
    timepoints = prevent_cutoff(experiment_events, timepoints, tr)
    timepoints['tr_onset'] = timepoints['tr_index'] * tr
    timepoints = timepoints.sort_values('tr_onset').reset_index(drop=True)

    # forward fill bar_direction
    timepoints.loc[:,
                   'bar_direction_min'] = timepoints['bar_direction_min'].ffill()
    prf_utils.test_alignment(timepoints)
    return timepoints


def create_design_matrix(timepoints, apertures, vhsize):
    condition_apertures = {int(condition.split(
        '_')[2]): frames for condition, frames in apertures.items()}

    # missing timepoints represented as blanks
    n_timepoints = timepoints['tr_index'].max() + 1
    dm = np.zeros((n_timepoints, vhsize[0], vhsize[1]))

    for i in range(n_timepoints):
        included_timepoints = timepoints[timepoints['tr_index'] == i]

        apertures = []
        for _, frame in included_timepoints.iterrows():
            start_seq_index = frame['seq_index_min']
            if not (np.isnan(start_seq_index)):
                aperture = condition_apertures[int(frame['bar_direction_min'])][int(
                    start_seq_index)]
                apertures.append(aperture)
        if len(apertures) != 0:
            dm[i, :, :] = np.mean(np.stack(apertures), axis=0)
    return dm


def save_design_matrix(dm, label, subject, run):
    out_path = DIR_OUTPUT /\
        f'sub-{str(subject).zfill(2)}_run-{str(run).zfill(2)
                                           }_design_matrix_{label}.npy'
    np.save(out_path, dm)
    print('Design matrix saved at:', out_path)


def create_subject_design_matrices(subject, tr_canonical, tr_output, vhsize_canonical, vhsize_output, gif_speed):
    dms = []
    for run in range(1, 11):
        print(f'Creating design matrix for subject {subject}, run {run}.')
        events, seq_timing, apertures = load_data(subject, run)
        experiment_events, aperture_start, end = get_experiment_events(
            events)
        print(f'Number of pulses: {len(experiment_events)}')
        aperture_events = get_aperture_events(seq_timing, aperture_start)

        # create canonical design matrix sampled at stimulus FPS
        timepoints = create_timepoints(
            experiment_events, aperture_events, tr_canonical)
        design_matrix_canonical = create_design_matrix(
            timepoints, apertures, vhsize_canonical)
        save_design_matrix(design_matrix_canonical, 'canonical', subject, run)

        # downsample temporally (to scanner/acquisition TR)
        timepoints.loc[:, 'tr_index'] = timepoints['tr_onset'] // tr_output
        design_matrix_output = create_design_matrix(
            timepoints, apertures, vhsize_canonical)
        # downsample spatially
        design_matrix_output = resize(design_matrix_output, vhsize_output)
        save_design_matrix(design_matrix_output, 'output', subject, run)

        prf_utils.create_gif(design_matrix_output,
                             tr_output / gif_speed, 'output', subject, run)

        dms.append(design_matrix_output)
    return dms


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
params = core_settings['prf']['design_matrix']
DIR_BASE = Path(core_settings['paths']['prf_experiment']['base'])
DIR_INPUT = DIR_BASE / core_settings['paths']['prf_experiment']['input']
DIR_OUTPUT = DIR_BASE / core_settings['paths']['prf_experiment']['design']
BLANK = params['blank']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()

    tr_canonical = params['tr_canonical']
    tr_output = params['tr_output']
    vhsize_canonical = params['vhsize_canonical']
    vhsize_output = params['vhsize_output']
    gif_speed = params['gif_speed']

    create_subject_design_matrices(
        args.subject, tr_canonical, tr_output, vhsize_canonical, vhsize_output, gif_speed)
