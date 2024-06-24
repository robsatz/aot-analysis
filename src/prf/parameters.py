# Courtesy of Jurjen Heij (https://github.com/gjheij/linescanning/blob/main/linescanning/prf.py)

import numpy as np
import pandas as pd


class Parameters():

    def __init__(
            self,
            params,
            model="gauss"):

        self.params = params
        self.model = model
        self.allow_models = ["gauss", "dog", "css", "norm", 'abc', 'abd']

    def to_df(self):

        if isinstance(self.params, pd.DataFrame):
            return self.params

        if not isinstance(self.params, np.ndarray):
            raise ValueError(
                f"Input must be np.ndarray, not '{type(self.params)}'")

        if self.params.ndim == 1:
            self.params = self.params[np.newaxis, :]

        # see: https://github.com/VU-Cog-Sci/prfpytools/blob/0649d7cae6d93536b51cc39bef36a92e7ae89747/prfpytools/postproc_utils.py#L348-L403
        if self.model in self.allow_models:
            if self.model == "gauss":
                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "r2": self.params[:, -1],
                    "ecc": np.sqrt(self.params[:, 0]**2+self.params[:, 1]**2),
                    "polar": np.angle(self.params[:, 0]+self.params[:, 1]*1j)
                }

                if self.params.shape[-1] > 6:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_disp"] = self.params[:, -2]

            elif self.model in ["norm", "abc", "abd"]:

                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "surr_ampl": self.params[:, 5],
                    "surr_size": self.params[:, 6],
                    "neur_bsl": self.params[:, 7],
                    "surr_bsl": self.params[:, 8],
                    "A": self.params[:, 3],
                    "B": self.params[:, 7],  # /params[:,3],
                    "C": self.params[:, 5],
                    "D": self.params[:, 8],
                    "ratio_B-to-D": self.params[:, 7]/self.params[:, 8],
                    "r2": self.params[:, -1],
                    "size ratio": self.params[:, 6]/self.params[:, 2],
                    "suppression index": (self.params[:, 5]*self.params[:, 6]**2)/(self.params[:, 3]*self.params[:, 2]**2),
                    "ecc": np.sqrt(self.params[:, 0]**2+self.params[:, 1]**2),
                    "polar": np.angle(self.params[:, 0]+self.params[:, 1]*1j)}

                if self.params.shape[-1] > 10:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_dsip"] = self.params[:, -2]

            elif self.model == "dog":
                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "surr_ampl": self.params[:, 5],
                    "surr_size": self.params[:, 6],
                    "r2": self.params[:, -1],
                    "size ratio": self.params[:, 6]/self.params[:, 2],
                    "suppression index": (self.params[:, 5]*self.params[:, 6]**2)/(self.params[:, 3]*self.params[:, 2]**2),
                    "ecc": np.sqrt(self.params[:, 0]**2+self.params[:, 1]**2),
                    "polar": np.angle(self.params[:, 0]+self.params[:, 1]*1j)}

                if self.params.shape[-1] > 8:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_dsip"] = self.params[:, -2]

            elif self.model == "css":
                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "css_exp": self.params[:, 5],
                    "r2": self.params[:, -1],
                    "ecc": np.sqrt(self.params[:, 0]**2+self.params[:, 1]**2),
                    "polar": np.angle(self.params[:, 0]+self.params[:, 1]*1j)}

                if self.params.shape[-1] > 7:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_dsip"] = self.params[:, -2]

        else:
            raise ValueError(f"Model must be one of {
                             self.allow_models}. Not '{self.model}'")

        return pd.DataFrame(params_dict)
