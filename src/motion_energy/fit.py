import numpy as np
import yaml
from pathlib import Path
from fracridge import FracRidgeRegressor
import argparse
import pickle
import os
import nibabel as nib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

from src import io_utils


def load_motion_energy():
    return np.load(DIR_MOTION_ENERGY / 'motion_energy_all.npy')


def load_r2(subject):
    subject = str(subject).zfill(3)
    # using subject's overall mean R2 to select voxels
    return nib.load(DIR_DERIVATIVES /
                    f'sub-{subject}/GLMsingle_analysis/sub-{subject}_R2_aot_mean.nii.gz').get_fdata()


def load_amplitudes(subject, session, segmentation):
    subject = str(subject).zfill(3)
    session = str(session).zfill(2)
    # beta weights from GLMsingle treated as neural response amplitudes
    glmsingle = np.load(DIR_DERIVATIVES /
                        f'sub-{subject}/ses-{session}/GLMsingle_{segmentation}/TYPED_FITHRF_GLMDENOISE_RR.npy',  allow_pickle=True).item()
    return glmsingle['betasmd']


def create_trial_sequence(subject, session, aot_condition=None):
    # load trial sequence
    session_trial_sequence = []
    for run_idx in range(10):
        with open(DIR_SETTINGS / f'experiment_settings_sub_{str(subject).zfill(2)}_ses_{str(session).zfill(2)}_run_{str(run_idx + 1).zfill(2)}.yml') as f:
            run_trial_sequence = yaml.load(f, Loader=yaml.FullLoader)[
                'stimuli']['movie_files']
        run_trial_sequence = [
            label for label in run_trial_sequence if label != 'blank']
        session_trial_sequence.extend(run_trial_sequence)

    # optionally filter for AOT condition ('fw' or 'rv')
    if aot_condition is not None:
        trial_indices = []
        trial_labels = []
        for trial_idx, trial_label in enumerate(session_trial_sequence):
            if trial_label[5:7] == aot_condition:
                trial_indices.append(trial_idx)
                trial_labels.append(trial_label)
    else:
        trial_indices = list(range(len(session_trial_sequence)))
        trial_labels = session_trial_sequence
    return trial_indices, trial_labels


def select_trials(amplitudes, trial_indices):
    return amplitudes[:, trial_indices]


def select_voxels(subject, amplitudes, n_slices, slice_nr, r2, rsq_threshold=None):
    r2 = r2.ravel()
    original_shape = amplitudes.shape

    # flatten to n_voxels x n_trials
    flattened_shape = (-1, original_shape[-1])
    amplitudes = np.reshape(amplitudes, flattened_shape)

    # filter by GLMsingle results
    if rsq_threshold is None:
        selected_ids = np.where(~np.isnan(amplitudes).all(axis=1))[0]
    else:
        selected_ids = np.where(r2 > rsq_threshold * 100)[0]

    # get slice
    slice_vertices = np.array_split(selected_ids, n_slices, axis=0)[slice_nr]
    amplitudes = amplitudes[slice_vertices]

    # save vertices
    subject = str(subject).zfill(3)
    out_path = DIR_DERIVATIVES / f'sub-{subject}' / 'moten_slices'
    os.makedirs(out_path, exist_ok=True)
    np.save(
        out_path / f'vertices_slice_{str(slice_nr).zfill(4)}.npy', slice_vertices)

    return amplitudes


def create_design_matrix(motion_energy, trial_labels, subject, session):

    # initialize design matrix
    n_channels = motion_energy.shape[1]
    design_matrix = np.zeros((len(trial_labels), n_channels))

    # chronologically insert motion energy features into design matrix
    for trial_idx, trial_label in enumerate(trial_labels):
        source_video_idx = int(trial_label[:4])-1
        design_matrix[trial_idx, :] = motion_energy[source_video_idx, :]

    subject = str(subject).zfill(3)
    session = str(session).zfill(2)
    np.save(DIR_DERIVATIVES / f'sub-{subject}' / f'ses-{session}' /
            f'sub-{subject}_ses-{session}_motion_energy_design_matrix.npy', design_matrix)
    return design_matrix


def fit(X, y, cv):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # z-scores motion energy features
        ('fracridge', TransformedTargetRegressor(
            transformer=StandardScaler(),  # z-scores neural response amplitudes
            regressor=FracRidgeRegressor(),
            check_inverse=False))])
    fracgrid = np.linspace(0.1, 1, 10)
    gridsearch = GridSearchCV(
        pipeline,
        param_grid={'fracridge__regressor__fracs': fracgrid},
        cv=cv)
    gridsearch.fit(X, y)
    return gridsearch


def save_pipeline(grid_search_obj, subject, segmentation):
    subject = str(subject).zfill(3)
    path = DIR_DERIVATIVES /\
        f'sub-{subject}/fracridge_{segmentation}'
    os.makedirs(path, exist_ok=True)
    with open(path / 'model.pkl', 'wb') as f:
        pickle.dump(grid_search_obj, f)
    np.save(path / 'betas.npy',
            grid_search_obj.best_estimator_.named_steps['fracridge'].regressor_.coef_)
    np.save(path / 'alphas.npy',
            grid_search_obj.best_estimator_.named_steps['fracridge'].regressor_.alpha_)


def main(subject, n_slices, slice_nr, segmentation, aot_condition, rsq_threshold, cv):

    motion_energy = load_motion_energy()
    r2 = load_r2(subject)

    subject_amplitudes = []
    subject_design_matrix = []
    for session in range(1, 6):

        trial_indices, trial_labels = create_trial_sequence(
            subject, session, aot_condition)

        session_amplitudes = load_amplitudes(subject, session, segmentation)

        print(f'Shapes before transformations - motion energy: {motion_energy.shape} and amplitudes: {session_amplitudes.shape}'
              )

        session_amplitudes = select_voxels(
            subject, session_amplitudes, n_slices, slice_nr, r2, rsq_threshold)
        session_amplitudes = select_trials(session_amplitudes, trial_indices)
        session_amplitudes = session_amplitudes.T  # n_trials x n_voxels

        session_design_matrix = create_design_matrix(
            motion_energy, trial_labels, subject, session)
        print('Number of negative motion energy values:',
              (session_design_matrix < 0).sum())
        # design_matrix[design_matrix < 0] = 0

        print(f"Shapes after transformations - design: {session_design_matrix.shape}, amplitudes: {session_amplitudes.shape}"
              )
        subject_amplitudes.append(session_amplitudes)
        subject_design_matrix.append(session_design_matrix)

    subject_amplitudes = np.concatenate(subject_amplitudes, axis=0)
    subject_design_matrix = np.concatenate(subject_design_matrix, axis=0)

    print(
        f'Final shapes - design: {subject_design_matrix.shape}, amplitudes: {subject_amplitudes.shape}')

    gridsearch_obj = fit(subject_design_matrix, subject_amplitudes, cv)
    save_pipeline(gridsearch_obj, subject, segmentation)


core_settings = io_utils.load_config()
DIR_BASE = Path(core_settings['paths']['aot_experiment']['base'])
DIR_SETTINGS = DIR_BASE / core_settings['paths']['aot_experiment']['settings']
DIR_MOTION_ENERGY = Path(core_settings['paths']['motion_energy'])
DIR_DERIVATIVES = Path(core_settings['paths']['derivatives'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    parser.add_argument("--slice_nr", type=int, default=1,
                        help="Slice number.")
    args = parser.parse_args()
    subject = args.subject
    slice_nr = args.slice_nr

    rsq_threshold = core_settings['fracridge']['rsq_threshold']
    cv = core_settings['fracridge']['cv']
    segmentation = core_settings['fracridge']['segmentation']
    aot_condition = core_settings['fracridge']['aot_condition']
    n_slices = core_settings['fracridge']['n_slices']

    for session in range(1, 6):
        session = str(session).zfill(2)
        main(subject, n_slices, slice_nr, segmentation,
             aot_condition, rsq_threshold, cv)
