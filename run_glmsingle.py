import os
import numpy as np
from pathlib import Path
from GLMsingle.glmsingle import GLM_single
import nibabel as nib
import argparse
from create_design_matrices import create_session_design_matrices
from aot.analysis.glmsingle.code_mainexp.design_constrct import construct_bold_for_one_session


def run_glmsingle(subject, session):

    design_matrices_aot, design_matrices_control = create_session_design_matrices(
        subject, session)
    # bold_data_session = []
    # for run_idx in range(len(design_matrices_aot)):
    bold_data = construct_bold_for_one_session(
        subject, session, 'T1W', 'nordicstc')

    output_dir = DIR_OUTPUT / \
        f'sub-{subject.zfill(3)}' / f'ses-{session}' / 'GLMsingle'
    os.makedirs(output_dir, exist_ok=True)
    outputtype = [1, 1, 1, 1]
    opt = dict()
    # set important fields for completeness (but these would be enabled by default)
    opt["wantlibrary"] = 1
    opt["wantglmdenoise"] = 1
    if outputtype[-1] == 1:
        opt["wantfracridge"] = 1
    else:
        opt["wantfracridge"] = 0
    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt["wantfileoutputs"] = outputtype
    opt["wantmemoryoutputs"] = outputtype
    glm = GLM_single(opt)
    glm.fit(design=design_matrices_aot, data=bold_data, stimdur=2.5,
            tr=0.9, outputdir=str(output_dir), figuredir=str(output_dir))
    # TODO: repeat for control design matrix


DIR_DATA = Path(
    '/tank/shared/2024/visual/AOT/derivatives/fmripreps/aotfull_preprocs/fullpreproc1')
DIR_OUTPUT = Path('./derivatives')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject", type=int, default=1,
                        help="Subject number.")
    parser.add_argument("-ses", "--session", type=int, default=1,
                        help="Session number.")
    args = parser.parse_args()
    subject = str(args.subject).zfill(2)
    session = str(args.session).zfill(2)
    run_glmsingle(subject, session)
