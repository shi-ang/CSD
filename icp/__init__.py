from __future__ import division

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from utils.util_survival import compute_decensor_times
from icp.scorer import SurvivalNC
from icp.error_functions import OnsSideQuantileRegErrFunc


class IcpSurvival(BaseEstimator):
    """Inductive conformal survival predictor."""
    def __init__(
            self,
            nc_function: SurvivalNC,
            condition=None,
            decensor_method: str = 'margin',
            n_quantiles: int = None
    ):
        self.train_data = None
        self.cal_data = None
        self.feature_names = None
        self.nc_function = nc_function
        self.decensor_method = decensor_method
        # Quantile level (1 - significance) of predictions.
        # Should be a float between 0 and 1. If ``None``, then we use the
        # default quantile levels (0.1, 0.2, ..., 0.9).
        if n_quantiles is None:
            self.n_quantiles = 9
        else:
            assert isinstance(n_quantiles, int) and n_quantiles > 0, "n_quantiles must be a positive integer"
            self.n_quantiles = n_quantiles
        self.quantile_levels = np.linspace(1 / (self.n_quantiles + 1), self.n_quantiles / (self.n_quantiles + 1),
                                           self.n_quantiles)

        # Check if condition-parameter is the default function (i.e.,
        # lambda x: 0). This is so we can safely clone the object without
        # the clone accidentally having self.conditional = True.
        default_condition = lambda x: 0
        is_default = callable(condition) and (condition.__code__.co_code == default_condition.__code__.co_code)

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

        self.categories = None
        self.cal_scores = None

    def fit(self, data_train, data_val):
        self.train_data = data_train
        self.feature_names = data_train.drop(['time', 'event'], axis=1).columns.tolist()
        self.nc_function.fit(self.train_data, data_val)

    def calibrate(self, data_val, increment=False):
        self._update_calibration_set(data_val, increment)

        features, t, e = compute_decensor_times(self.cal_data, self.train_data, method=self.decensor_method)

        if self.conditional:
            # Not tested yet for this current project
            # category_map = np.array([self.condition((x[i, :], y[i]))
            #                          for i in range(y.size)])
            # self.categories = np.unique(category_map)
            # self.cal_scores = defaultdict(partial(np.ndarray, 0))
            #
            # for cond in self.categories:
            #     idx = category_map == cond
            #     cal_scores = self.nc_function.score(self.cal_x[idx, :],
            #                                         self.cal_y[idx])
            #     self.cal_scores[cond] = np.sort(cal_scores, 0)[::-1]
            raise NotImplementedError
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(feature_df=features, t=t, e=e,
                                                quantile_levels=self.quantile_levels,
                                                method=self.decensor_method)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

    def predict(self, x):
        """Predict the output values for a set of input patterns.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        Returns
        -------
        p : numpy array of shape [n_samples, n_quantiles]
            If `quantile_levels` is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each quantile
            level (0.1, 0.2, ..., 0.9).
        """
        quan_pred = np.zeros((x.shape[0], self.n_quantiles + 1))

        condition_map = np.array([self.condition((x[i, :], None)) for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :], self.cal_scores[condition],
                                             self.feature_names, self.quantile_levels)
                quan_pred[idx, :] = p

        if 0 not in self.quantile_levels:
            quan_levels = np.insert(self.quantile_levels, 0, 0)
        else:
            quan_levels = self.quantile_levels

        return quan_levels, quan_pred

    def _update_calibration_set(self, data: pd.DataFrame, increment):
        if increment and self.cal_data is not None:
            self.cal_data = pd.concat([self.cal_data, data])
        else:
            self.cal_data = data
