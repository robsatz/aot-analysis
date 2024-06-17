import numpy as np
import numpy as np
from datetime import timedelta
import time
import logging


class FitLogger():
    def __init__(self, fitter, rsq_threshold, loglevel=logging.DEBUG):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=loglevel)
        self.fitter = fitter
        self.rsq_threshold = rsq_threshold

    def _reset_timer(self):
        self.start = time.time()

    def _print_time(self):
        elapsed = (time.time() - self.start)
        self.logger.info(
            f"Elapsed time: {timedelta(seconds=elapsed)}", flush=True)

    def grid_fit(self, param_dict):
        self._reset_timer()
        self.fitter.grid_fit(**param_dict)
        self._print_time()
        self.save_results(self.fitter.gridsearch_params)

    def iterative_fit(self, param_dict):
        self._reset_timer()
        self.fitter.iterative_fit(**param_dict)
        self._print_time()
        self.save_results(self.fitter.iterative_search_params)

    def save_results(self, search_params):
        gauss_grid = np.nan_to_num(search_params)
        mean_rsq = np.mean(
            gauss_grid[gauss_grid[:, -1] > self.rsq_threshold, -1])
        self.info(f'Mean rsq: {mean_rsq}')
