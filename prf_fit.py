import numpy as np
import matplotlib.pyplot as pl
import yaml
import os
from prfpy import stimulus, model, fit
import numpy as np
from scipy import io
import nibabel as nib
from datetime import datetime, timedelta
import time
import sys
import prf_fit
import logging


class FitLogger():
    def __init__(self, fitter, rsq_threshold):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        self.fitter = fitter
        self.rsq_threshold = rsq_threshold

    def _reset_timer(self):
        self.start = time.time()

    def _print_time(self):
        elapsed = (time.time() - self.start)
        self.logger.log(
            f"Elapsed time: {timedelta(seconds=elapsed)}", flush=True)

    def grid_fit(self, params):
        self._reset_timer()
        self.fitter.grid_fit(params)
        self._print_time()
        self.save_results(self.fitter.gridsearch_params)

    def iterative_fit(self, params):
        self._reset_timer()
        self.fitter.iterative_fit(params)
        self._print_time()
        self.save_results(self.fitter.iterative_search_params)

    def save_results(self, search_params):
        gauss_grid = np.nan_to_num(search_params)
        mean_rsq = np.mean(
            gauss_grid[gauss_grid[:, -1] > self.rsq_threshold, -1])
        self.log(mean_rsq)
        pass


config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
params = config['prf']['design_matrix']
