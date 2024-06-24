import yaml
import nibabel as nib
import gzip
import shutil
from pathlib import Path
import os


def load_config():
    with open('./config.yml', 'r') as file:
        return yaml.safe_load(file)


def load_metadata(subject, session=1, task='AOT_rec-nordicstc', run=1):
    subject = str(subject).zfill(3)
    session = str(session).zfill(2)
    task = str(task).zfill(2)
    run = str(run)
    filename = f'sub-{subject}_'\
        + f'ses-{session}_' \
        + f'task-{task}_' \
        + f'run-{run}_' \
        + 'space-T1w_part-mag_boldref.nii.gz'
    filepath = DIR_DATA \
        / f'sub-{subject}' \
        / f'ses-01' \
        / 'func' \
        / filename
    nifti_image = nib.load(filepath)
    return nifti_image.header, nifti_image.affine


def save_nifti(data, filepath, subject, **kwargs):
    header, affine = load_metadata(subject, **kwargs)
    nifti_image = nib.Nifti1Image(data, header=header, affine=affine)
    nifti_image.to_filename(filepath)

    print(f'Nifti saved to: {filepath}')


config = load_config()
DIR_DATA = Path(config['paths']['data'])
