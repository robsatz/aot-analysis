import os
from pathlib import Path
from glmsingle import GLM_single
import argparse
from src.glmsingle.design_matrix import create_session_design_matrices
import copy
import nibabel as nib

from src import io_utils


def run_glmsingle(subject, session, data_params, model_params, fit_params, n_features):

    in_path = Path(core_settings['paths']['data'])
    bold_data = []
    for run in range(1, 11):
        filepath = in_path / \
            (f'sub-{subject.zfill(3)}_'
             + f'ses-{session}_'
             + f'task-AOT_rec-nordicstc_'
             + f'run-{run}_space-T1w_desc-preproc_part-mag_bold.nii.gz')
        # sub-002_ses-01_task-AOT_rec-nordicstc_run-1_space-T1w_desc-preproc_part-mag_bold.nii.gz
        try:
            print('Loading data file:', filepath)
            run_data = nib.load(filepath).get_fdata()
            bold_data.append(run_data)
        except FileNotFoundError:
            print(f'File not found: {filepath}')
            continue

    design_aot, design_pres, design_scram = create_session_design_matrices(
        subject, session, n_features)
    # designs = {'pres': design_pres, 'aot': design_aot}
    designs = {'scram': design_scram}

    for label, design in designs.items():
        print('RUNNING GLMSINGLE FOR:', label)

        output_dir = DIR_OUTPUT / \
            f'sub-{subject.zfill(3)}' / f'ses-{session}' / f'GLMsingle_{label}'
        os.makedirs(output_dir, exist_ok=True)
        print(model_params, fit_params)

        # copy avoids overwriting
        print(design[0].shape)
        glm = GLM_single(copy.deepcopy(model_params))
        glm.fit(design=design, data=bold_data, stimdur=fit_params['stimdur'], tr=fit_params['tr'], outputdir=str(
            output_dir), figuredir=str(output_dir))


core_settings = io_utils.load_config()
DIR_OUTPUT = Path(core_settings['paths']['derivatives'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    parser.add_argument("-ses", "--session", type=int, default=1,
                        help="Session number.")
    args = parser.parse_args()
    subject = str(args.subject).zfill(2)
    session = str(args.session).zfill(2)

    glmsingle_settings = core_settings['glmsingle']
    run_glmsingle(subject, session,
                  data_params=glmsingle_settings['data'],
                  model_params=glmsingle_settings['model'],
                  fit_params=glmsingle_settings['fit'],
                  n_features=glmsingle_settings['design_matrix']['n_features']
                  )
