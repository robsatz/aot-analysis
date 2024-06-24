import yaml
import nibabel as nib
import gzip
import shutil
from pathlib import Path
import os


def load_config():
    with open(DIR_CONFIG / 'config.yml', 'r') as file:
        return yaml.safe_load(file)


def load_metadata(subject, metadata_path=None, session=1, task='AOT_rec-nordicstc', run=1):
    subject = str(subject).zfill(3)
    if type(session) == int:
        session = str(session).zfill(2)
    task = str(task).zfill(2)
    run = str(run)
    if not metadata_path:
        filename = f'sub-{subject}_'\
            + f'ses-{session}_' \
            + f'task-{task}_' \
            + f'run-{run}_' \
            + 'space-T1w_part-mag_boldref.nii.gz'
        metadata_path = DIR_DATA \
            / f'sub-{subject}' \
            / f'ses-01' \
            / 'func' \
            / filename
    nifti_image = nib.load(metadata_path)
    return nifti_image.header, nifti_image.affine


def save_nifti(data, out_path, subject, **kwargs):
    header, affine = load_metadata(subject, **kwargs)
    nifti_image = nib.Nifti1Image(data, header=header, affine=affine)
    nifti_image.to_filename(out_path)

    print(f'Nifti saved to: {out_path}')


DIR_CONFIG = Path(__file__).parent.parent
config = load_config()
DIR_DATA = Path(config['paths']['data'])
