import os
import numpy as np
from pathlib import Path
import nibabel as nib
import argparse
import yaml
import copy


def load_bold(subject):
    # same header/image used for all outputs
    filename = f'sub-{subject}_ses-01_task-AOT_rec-nordicstc_run-1_space-T1w_part-mag_boldref.nii.gz'
    bold = nib.load(DIR_DATA / f'sub-{subject}' /
                    f'ses-01' / 'func' / filename)
    return filename, bold.header, bold.get_fdata()


def load_evals(session_path):
    results_aot = np.load(session_path / 'GLMsingle_aot' /
                          'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
    results_control = np.load(session_path / 'GLMsingle_control' /
                              'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
    return results_aot, results_control


def compute_diff(results_aot, results_control):
    aot_r2 = results_aot['R2']
    control_r2 = results_control['R2']
    r2diff = (aot_r2 - control_r2)

    return {
        'R2_aot': aot_r2,
        'R2_control': control_r2,
        'R2_diff': r2diff
    }


def aggregate(results):
    aggregates = {}
    for measure in ['R2_aot', 'R2_control', 'R2_diff']:
        aggregates[f'{measure}_mean'] = np.mean(results[measure], axis=0)
        aggregates[f'{measure}_median'] = np.median(results[measure], axis=0)
        aggregates[f'{measure}_std'] = np.std(results[measure], axis=0)
    return aggregates


def save_niftis(out_dir, outputs, bold_header, subject, session=None, bold_img=None, bold_filename=None):
    out_path = out_dir / 'GLMsingle_analysis'
    os.makedirs(out_path, exist_ok=True)
    for label, data in outputs.items():
        session_label = ''
        if session is not None:
            session_label = f'ses-{session}_'
        nib.nifti1.Nifti1Image(data, header=bold_header, affine=None).to_filename(
            out_path / f'sub-{subject}_{session_label}{label}.nii.gz')
    if bold_img is not None:
        nib.nifti1.Nifti1Image(bold_img, header=bold_header, affine=None).to_filename(
            out_path / bold_filename)
    print('Niftis saved to:', out_path)


def compute_session_aggregates(subject, session, header):
    session_path = DIR_DERIVATIVES / \
        f'sub-{subject}' / f'ses-{str(session).zfill(2)}'
    results_aot, results_control = load_evals(
        session_path)

    results = compute_diff(results_aot, results_control)
    save_niftis(session_path, results, header, subject, session=session)

    return results


def compute_subject_aggregates(subject):
    subject = str(subject).zfill(3)
    bold_filename, bold_header, bold_img = load_bold(subject)
    results = {}
    for session in range(1, 6):
        session = str(session).zfill(2)
        session_results = compute_session_aggregates(
            subject, session, bold_header)
        for measure in session_results:
            if measure not in results:
                results[measure] = []
            results[measure].append(session_results[measure])
    results = aggregate(results)

    out_dir = DIR_DERIVATIVES / f'sub-{subject}'
    save_niftis(out_dir, results, bold_header, subject,
                bold_filename=bold_filename, bold_img=bold_img)


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_DATA = Path(core_settings['paths']['data'])
DIR_DERIVATIVES = Path(core_settings['paths']['derivatives'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    args = parser.parse_args()
    subject = str(args.subject).zfill(3)
    compute_subject_aggregates(subject)
