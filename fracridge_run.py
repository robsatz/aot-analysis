import numpy as np
import yaml
from pathlib import Path
from fracridge import FracRidgeRegressorCV
import argparse
import pickle
import os


def load_data(subject, session, condition):

    glmsingle = np.load(DIR_DERIVATIVES /
                        f'sub-{subject}/ses-{session}/GLMsingle_{condition}/TYPED_FITHRF_GLMDENOISE_RR.npy',  allow_pickle=True).item()
    # beta weights from GLMsingle treated as neural response amplitudes
    return glmsingle['betasmd'], glmsingle['R2']


def select_voxels(amplitudes, r2, threshold=None):
    if threshold is not None:
        r2 = r2.ravel()
        if type(threshold) == float:
            # select voxels with r2 > threshold
            selected_ids = np.where(r2 > threshold * 100)[0]
        elif type(threshold) == int:
            # select top n
            valid_ids = np.where(~np.isnan(r2))[0]
            selected_ids = valid_ids[np.argsort(r2[valid_ids])[-threshold:]]
        else:
            raise ValueError('Invalid threshold type. Expected float or int.')

        original_shape = amplitudes.shape
        flattened_shape = (-1, original_shape[-1])  # n_voxels x n_trials
        amplitudes = np.reshape(amplitudes, flattened_shape)
        amplitudes = amplitudes[selected_ids, :]
    return amplitudes


def create_design_matrix(motion_energy, n_trials, subject, session):
    # n_trials x n_channels
    design_matrix = np.zeros((n_trials, motion_energy.shape[1]))
    session_trial_sequence = []
    for run_idx in range(10):
        with open(DIR_SETTINGS / f'experiment_settings_sub_{subject[1:]}_ses_{session}_run_{str(run_idx + 1).zfill(2)}.yml') as f:
            run_trial_sequence = yaml.load(f, Loader=yaml.FullLoader)[
                'stimuli']['movie_files']
        run_trial_sequence = [
            label for label in run_trial_sequence if label != 'blank']
        session_trial_sequence.extend(run_trial_sequence)

    for trial_idx, trial_label in enumerate(session_trial_sequence):
        design_matrix[trial_idx, :] = motion_energy[int(trial_label[:4])-1, :]
    return design_matrix


def z_score(array, axis=0):
    mean = np.nanmean(array, axis=axis, keepdims=True)
    std = np.nanstd(array, axis=axis, keepdims=True)
    return (array - mean) / std


def model(X, y, cv):
    # specifying default parameters for clarity
    model = FracRidgeRegressorCV(normalize=False, cv=cv)
    model.fit(X, y)
    return model


def save_model(model, subject, session, condition):
    path = DIR_DERIVATIVES / \
        f'sub-{subject}/ses-{session}/fracridge_{condition}'
    os.makedirs(path, exist_ok=True)
    with open(path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    np.save(path / 'betas.npy', model.coef_)
    np.save(path / 'alphas.npy', model.alpha_)


def main(subject, session, threshold, cv):
    motion_energy = np.load(DIR_MOTION_ENERGY / 'motion_energy_all.npy')

    for condition in ('aot', 'control'):
        amplitudes, r2 = load_data(subject, session, condition)
        print(f'Shapes before transformations - motion energy: {motion_energy.shape} and amplitudes: {amplitudes.shape}'
              )

        amplitudes = select_voxels(amplitudes, r2, threshold=threshold)

        design_matrix = create_design_matrix(
            motion_energy, amplitudes.shape[1], subject, session)
        design_matrix, amplitudes = z_score(
            design_matrix), z_score(amplitudes.T)

        print(f"Shapes after transformations - design: {design_matrix.shape}, amplitudes: {amplitudes.shape}"
              )

        fracridge_obj = model(design_matrix, amplitudes, cv)
        save_model(fracridge_obj, subject, session, condition)


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_BASE = Path(core_settings['paths']['aot_experiment']['base'])
DIR_SETTINGS = DIR_BASE / core_settings['paths']['aot_experiment']['settings']
DIR_MOTION_ENERGY = Path(core_settings['paths']['motion_energy'])
DIR_DERIVATIVES = Path(core_settings['paths']['derivatives'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    parser.add_argument("-ses", "--session", type=int, default=1,
                        help="Session number.")
    args = parser.parse_args()
    subject = str(args.subject).zfill(3)
    session = str(args.session).zfill(2)
    threshold = core_settings['fracridge']['threshold']
    cv = core_settings['fracridge']['cv']

    main(subject, session, threshold, cv)
