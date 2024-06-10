import os
import numpy as np
from pathlib import Path
from GLMsingle.glmsingle import GLM_single
import nibabel as nib
import argparse
from glmsingle_create_design_matrix import create_session_design_matrices
from aot.analysis.glmsingle.code_mainexp.design_constrct import construct_bold_for_one_session
import yaml
import copy


def save_niftis(subject, session):
    aot = np.load(DIR_OUTPUT / 'GLMsingle_aot' /
                  'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
    control = np.load(DIR_OUTPUT / 'GLMsingle_control' /
                      'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()

    aot_r2 = aot['R2']
    control_r2 = control['R2']
    r2diff = (aot_r2 - control_r2)

    outputs = {
        'R2aot': aot_r2,
        'R2control': control_r2,
        'R2diff': r2diff
    }

    header = nib.load(DIR_DATA / f'sub-{str(subject).zfill(3)}_ses-{str(session).zfill(
        2)}_task-AOT_rec-nordicstc_run-1_space-T1w_part-mag_boldref.nii.gz').header

    out_path = DIR_OUTPUT / 'GLMsingle_analysis'
    os.makedirs(out_path, exist_ok=True)
    for label, data in outputs.items():
        nib.nifti1.Nifti1Image(data, header=header, affine=None).to_filename(
            out_path / f'{label}.nii.gz')


def run_glmsingle(subject, session, data_params, model_params, fit_params):

    bold_data = construct_bold_for_one_session(
        subject, session, data_params['datatype'], data_params['nordictype'])

    design_aot, design_control = create_session_design_matrices(
        subject, session)
    designs = {'control': design_control, 'aot': design_aot}

    for label, design in designs.items():
        print('RUNNING GLMSINGLE FOR:', label)

        output_dir = DIR_OUTPUT / \
            f'sub-{subject.zfill(3)}' / f'ses-{session}' / f'GLMsingle_{label}'
        os.makedirs(output_dir, exist_ok=True)
        print(model_params, fit_params)

        # copy avoids overwriting
        glm = GLM_single(copy.deepcopy(model_params))
        glm.fit(design=design, data=bold_data, stimdur=fit_params['stimdur'], tr=fit_params['tr'], outputdir=str(
            output_dir), figuredir=str(output_dir))


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_DATA = Path(core_settings['paths']['data'])
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
                  fit_params=glmsingle_settings['fit']
                  )
    save_niftis(subject, session)
