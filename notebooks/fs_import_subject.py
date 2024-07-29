import cortex
import argparse
from aot_analysis import io_utils

parser = argparse.ArgumentParser()
parser.add_argument("-fsub", "--freesurfer_subject", type=int, default=1,
                    help="Subject number.")
parser.add_argument("-csub", "--pycortex_subject", type=int, default=1,)

args = parser.parse_args()
freesurfer_subject = f'sub-{str(args.freesurfer_subject).zfill(3)}'
pycortex_subject = f'sub-{str(args.pycortex_subject).zfill(3)}'

config = io_utils.load_config()

print(
    f'Importing freesurfer subject {freesurfer_subject} as pycortex subject {pycortex_subject}...')
cortex.freesurfer.import_subj(
    freesurfer_subject,
    pycortex_subject=pycortex_subject,  # aot subject id
    freesurfer_subject_dir=config['paths']['freesurfer'])
