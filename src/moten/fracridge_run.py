import numpy as np
import yaml
from pathlib import Path
from fracridge import FracRidgeRegressorCV
import argparse
import pickle
import os
import nibabel as nib


def load_motion_energy():
    return np.load(DIR_MOTION_ENERGY / 'motion_energy_all.npy')


def load_r2(subject):
    # using subject's overall mean R2 to select voxels
    return nib.load(DIR_DERIVATIVES /
                    f'sub-{subject}/GLMsingle_analysis/sub-{subject}_R2_aot_mean.nii.gz').get_fdata()


def load_amplitudes(subject, session, condition):
    # beta weights from GLMsingle treated as neural response amplitudes
    glmsingle = np.load(DIR_DERIVATIVES /
                        f'sub-{subject}/ses-{session}/GLMsingle_{condition}/TYPED_FITHRF_GLMDENOISE_RR.npy',  allow_pickle=True).item()
    return glmsingle['betasmd']


def select_voxels(amplitudes, r2, rsq_threshold=None):
    r2 = r2.ravel()
    original_shape = amplitudes.shape
    # flatten to n_voxels x n_trials
    flattened_shape = (-1, original_shape[-1])
    amplitudes = np.reshape(amplitudes, flattened_shape)

    if rsq_threshold is not None:
        selected_ids = r2 > rsq_threshold * 100
    else:
        selected_ids = ~np.isnan(amplitudes)

    return amplitudes[selected_ids, :]


def create_design_matrix(motion_energy, n_trials, subject, session):
    n_channels = motion_energy.shape[1]
    design_matrix = np.zeros((n_trials, n_channels))
    session_trial_sequence = []
    for run_idx in range(10):
        with open(DIR_SETTINGS / f'experiment_settings_sub_{subject[1:]}_ses_{session}_run_{str(run_idx + 1).zfill(2)}.yml') as f:
            run_trial_sequence = yaml.load(f, Loader=yaml.FullLoader)[
                'stimuli']['movie_files']
        run_trial_sequence = [
            label for label in run_trial_sequence if label != 'blank']
        session_trial_sequence.extend(run_trial_sequence)

    # chronologically inserting motion energy features into dm
    for trial_idx, trial_label in enumerate(session_trial_sequence):
        source_video_idx = int(trial_label[:4])-1
        design_matrix[trial_idx, :] = motion_energy[source_video_idx, :]

    np.save(DIR_DERIVATIVES / 'prf_fits' /
            f'sub-{subject}_ses-{session}_design_matrix_fracridge.npy', design_matrix)
    return design_matrix


def z_score(array, axis):
    mean = np.nanmean(array, axis=axis, keepdims=True)
    std = np.nanstd(array, axis=axis, keepdims=True)
    return (array - mean) / std


def model(X, y, cv):
    model = FracRidgeRegressorCV(normalize=False, cv=cv)
    fracgrid = np.linspace(0.1, 1, 10)
    model.fit(X, y, frac_grid=fracgrid)
    return model


def save_model(model, subject, session, condition):
    path = DIR_DERIVATIVES /\
        f'sub-{subject}/ses-{session}/fracridge_{condition}'
    os.makedirs(path, exist_ok=True)
    with open(path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    np.save(path / 'betas.npy', model.coef_)
    np.save(path / 'alphas.npy', model.alpha_)


def run_fracridge(subject, session, rsq_threshold, cv):
    motion_energy = load_motion_energy()
    r2 = load_r2(subject)

    for condition in ('aot', 'control'):
        amplitudes = load_amplitudes(subject, session, condition)
        print(
            f'Running FracRidge for subject {int(subject)}, session {session}, condition {condition}')
        print(f'Shapes before transformations - motion energy: {motion_energy.shape} and amplitudes: {amplitudes.shape}'
              )

        amplitudes = select_voxels(amplitudes, r2, rsq_threshold=rsq_threshold)
        amplitudes = amplitudes.T  # n_trials x n_voxels

        design_matrix = create_design_matrix(
            motion_energy, amplitudes.shape[0], subject, session)
        # design_matrix[design_matrix < 0] = 0

        # standardize features
        design_matrix = z_score(design_matrix, axis=0)
        amplitudes = z_score(amplitudes, axis=0)

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
    args = parser.parse_args()
    subject = str(args.subject).zfill(3)

    rsq_threshold = core_settings['fracridge']['rsq_threshold']
    cv = core_settings['fracridge']['cv']

    for session in range(1, 6):
        session = str(session).zfill(2)
        run_fracridge(subject, session, rsq_threshold, cv)
