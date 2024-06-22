import numpy as np
from datetime import datetime, timedelta
import time
import logging
import copy
import yaml
import os
from pathlib import Path


class FitLogger():
    def __init__(self, subject, slice_nr, loglevel='INFO'):

        self.subject = str(subject).zfill(3)
        self.slice_nr = str(slice_nr).zfill(5)
        self.fitter = None

        # prepare output dir
        self.results_path = DIR_DERIVATIVES / \
            f'sub-{self.subject}' / 'prf_fits'
        os.makedirs(self.results_path, exist_ok=True)
        self.log_path = DIR_DERIVATIVES / f'sub-{self.subject}' / 'logs'
        os.makedirs(self.log_path, exist_ok=True)

        # setup logger
        self.logger = self._create_logger(loglevel)

    def _create_logger(self, loglevel):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(loglevel)
        # setup to write logs to file
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_handler = logging.FileHandler(
            self.log_path / f'PRF_{current_time}.log')
        file_handler.setLevel(loglevel)
        # setup specific logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # add formatter to file_handler, file_handler to logger
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _reset_timer(self):
        self.start = time.time()

    def _print_time(self):
        elapsed = (time.time() - self.start)
        self.logger.info(
            f"Elapsed time: {timedelta(seconds=elapsed)}")

    def _create_mask(self, params, rsq_threshold, filter_positive):
        rsq_mask = self.fitter.gridsearch_params[:, -1] > rsq_threshold
        if filter_positive:
            ampl_mask = self.fitter.gridsearch_params[:, 3] > 0.0
            return (rsq_mask & ampl_mask)
        return rsq_mask

    def attach_fitter(self, name, fitter):
        # maintain previous fitter
        if self.fitter is not None:
            self.previous_fitter = copy.deepcopy(self.fitter)
        self.name = name
        self.fitter = fitter
        self.logger.info(f'Fitter attached: {name}')

    def grid_fit(self, fit_params, rsq_threshold, filter_positive):
        self.logger.info(
            f'Starting grid search with {self.fitter.data.shape[0]} datapoints')
        self.logger.debug(fit_params)
        self._reset_timer()
        self.fitter.grid_fit(**fit_params)
        self._print_time()
        grid_mask = self._create_mask(
            fit_params, rsq_threshold, filter_positive)
        self.save_results('grid', self.fitter.gridsearch_params, grid_mask)
        return grid_mask.sum()

    def iterative_fit(self, fit_params):
        self.logger.debug(fit_params)
        self._reset_timer()
        self.fitter.iterative_fit(**fit_params)
        self._print_time()
        self.save_results(
            'iter', self.fitter.iterative_search_params, self.fitter.rsq_mask)

    def filter_positive_prfs(self):
        self.fitter.iterative_search_params[self.fitter.iterative_search_params[:, 3] < 0] = 0.0

    def save_results(self, stage, search_params, mask):
        search_results = np.nan_to_num(search_params)
        mean_rsq = np.mean(search_results[mask, -1])
        self.logger.info(f'STAGE COMPLETED: {stage}')
        self.logger.info(f'---- mean rsq: {mean_rsq}')
        self.logger.info(
            f'---- n_voxels > rsq_threshold: {mask.sum()} / {len(mask)} = {mask.sum()/len(mask)}')
        np.save(
            self.results_path / f'sub-{self.subject}_{self.slice_nr}_{self.name}_{stage}_fit.npy', search_results)


config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
DIR_DERIVATIVES = Path(config['paths']['derivatives'])
