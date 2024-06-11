import os
import numpy as np
from pathlib import Path
import nibabel as nib
import argparse
import yaml
import copy


def load_data(session_path, subject, session):
    results_aot = np.load(session_path / 'GLMsingle_aot' /
                          'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
    results_control = np.load(session_path / 'GLMsingle_control' /
                              'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
    header = nib.load(DIR_DATA / f'sub-{subject}' /
                      f'ses-{session}' / 'func' /
                      f'sub-{subject}_ses-{session}_task-AOT_rec-nordicstc_run-1_space-T1w_part-mag_boldref.nii.gz').header
    return results_aot, results_control, header


def compute_diff(results_aot, results_control):
    aot_r2 = results_aot['R2']
    control_r2 = results_control['R2']
    r2diff = (aot_r2 - control_r2)

    return {
        'R2aot': aot_r2,
        'R2control': control_r2,
        'R2diff': r2diff
    }


def save_niftis(out_dir, outputs, header, subject, session):
    out_path = out_dir / 'GLMsingle_analysis'
    os.makedirs(out_path, exist_ok=True)
    for label, data in outputs.items():
        nib.nifti1.Nifti1Image(data, header=header, affine=None).to_filename(
            out_path / f'sub-{subject}_ses-{session}_{label}.nii.gz')
    print('Niftis saved to:', out_path)


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_DATA = Path(core_settings['paths']['data'])
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
    session_path = DIR_DERIVATIVES / \
        f'sub-{subject}' / f'ses-{session}'

    results_aot, results_control, header = load_data(
        session_path, subject, session)

    outputs = compute_diff(results_aot, results_control)

    save_niftis(session_path, outputs, header, subject, session)
