from __future__ import division

import abc
import numpy as np


class RegressionErrFunc(object):
    """Base class for regression model error functions.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):  # , norm=None, beta=0):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, n_classes]
            Class probability estimates for each sample.

        y : numpy array of shape [n_samples]
            True output labels of each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of the samples.
        """
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):  # , norm=None, beta=0):
        """Apply the inverse of the nonconformity function (i.e.,
        calculate prediction interval).

        Parameters
        ----------
        nc : numpy array of shape [n_calibration_samples]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (0, 1).

        Returns
        -------
        interval : numpy array of shape [n_samples, 2]
            Minimum and maximum interval boundaries for each prediction.
        """
        pass


class AbsErrorErrFunc(RegressionErrFunc):
    """Calculates absolute error nonconformity for regression problems.

        For each correct output in ``y``, nonconformity is defined as

        .. math::
            | y_i - \hat{y}_i |
    """

    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class QuantileRegErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression error.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        max{\hat{q}_low - y, y - \hat{q}_high}

    """

    def __init__(self):
        super(QuantileRegErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])


# CQR asymmetric error function
class QuantileRegAsymmetricErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression asymmetric error function.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        E_low = \hat{q}_low - y
        E_high = y - \hat{q}_high

    """

    def __init__(self):
        super(QuantileRegAsymmetricErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]

        error_high = y - y_upper
        error_low = y_lower - y

        err_high = np.reshape(error_high, (y_upper.shape[0], 1))
        err_low = np.reshape(error_low, (y_lower.shape[0], 1))

        return np.concatenate((err_low, err_high), 1)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance / 2) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index, 0], nc[index, 1]])


class OnsSideQuantileRegErrFunc(RegressionErrFunc):
    def __init__(self):
        super(OnsSideQuantileRegErrFunc, self).__init__()

    def apply(self, predictions, y):
        """

        :param predictions:
        :param y:
        :return: (n_samples, n_significance)
        """
        t_ = np.expand_dims(y[:, 0], axis=-1)
        errors = predictions - t_
        return errors

    def apply_inverse(self, nc: np.ndarray, quantile_levels: np.ndarray):
        """

        :param nc: numpy array of shape [n_calibration_samples, n_significance]
        :param quantile_levels:
        :return:
        """
        nc = np.sort(nc, axis=0)
        # -1 because python is 0-indexed
        index1 = np.ceil((1 - quantile_levels) * (nc.shape[0] + 1)) - 1
        index1 = index1.astype(int)
        index1 = index1.clip(0, nc.shape[0] - 1)
        # because i-th number in index1 is the index for i-th significance
        index2 = np.arange(quantile_levels.shape[0])
        errors = nc[index1, index2]
        # using quantile directly
        # quantiles = np.ceil((1 - quantile_levels) * (nc.shape[0] + 1)) / nc.shape[0]
        # errors = np.quantile(nc, quantiles, axis=0).diagonal()
        return errors
