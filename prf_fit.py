
import numpy as np
import matplotlib.pyplot as pl
from prfpy import stimulus, model, fit
from prf_logger import FitLogger
import yaml
from pathlib import Path
import numpy as np
import nibabel as nib
import argparse


def load_data():
    pass


def select_voxels():
    pass


def create_grid():
    pass


def batch():
    pass


def fit_session_prfs(subject, session, slice_nr, n_jobs):
    FitLogger()
    pass


config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
params = config['prf']

paths = config['paths']['prf_experiment']
DIR_BASE = Path(paths['base'])
DIR_DESIGN = DIR_BASE / paths['design']
DIR_DATA = Path(paths['bold'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    parser.add_argument("-ses", "--session", type=int, default=1)
    parser.add_argument("-slice", "--slice_nr", type=int, default=1)
    parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1)
    args = parser.parse_args()

    subject = args.subject
    session = args.session
    slice_nr = args.slice_nr
    n_jobs = args.n_jobs

    fit_session_prfs(subject, session, slice_nr, n_jobs)
