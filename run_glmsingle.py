import os
import numpy as np
from pathlib import Path
from GLMsingle.glmsingle import GLM_single
import nibabel as nib
import argparse
from create_design_matrices import create_session_design_matrices
from aot.analysis.glmsingle.code_mainexp.design_constrct import construct_bold_for_one_session
import yaml


def run_glmsingle(subject, session):

    design_aot, design_control = create_session_design_matrices(
        subject, session)
    designs = {'aot': design_aot, 'control': design_control}
    bold_data = construct_bold_for_one_session(
        subject, session, DATATYPE, NORDICTYPE)

    for label, design in designs.items():
        print('RUNNING GLMSINGLE FOR:', label)
        output_dir = DIR_OUTPUT / \
            f'sub-{subject.zfill(3)}' / f'ses-{session}' / f'GLMsingle_{label}'
        os.makedirs(output_dir, exist_ok=True)

        glm = GLM_single(PARAMS)
        glm.fit(design=design, data=bold_data, stimdur=2.5,
                tr=0.9, outputdir=str(output_dir), figuredir=str(output_dir))


core_settings = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_DATA = Path(core_settings['paths']['data'])
DIR_OUTPUT = Path(core_settings['paths']['derivatives'])
PARAMS = core_settings['glmsingle']['params']
DATATYPE = core_settings['glmsingle']['datatype']
NORDICTYPE = core_settings['glmsingle']['nordictype']

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
