# -*- coding: utf-8 -*-
"""Canonical Correlation Analysis (CCA). http://en.wikipedia.org/wiki/Canonical_correlation
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import functools
# from ..classes.classifier import Classifier
from .reference_frequencies import generate_reference_signals


# class CCA(Classifier):
class CCA():
    r"""
    Calculates the canonical correlation coefficient and
    corresponding weights which maximize a correlation coefficient
    between linear combinations of the two specified multivariable
    signals.

    Args:
        sampling_rate: int
            Sampling frequency
        target_frequencies : array
            Frequencies for SSVEP classification
        signal_size : int
            Window/segment length in time samples
        sampling_rate : int
            Sampling frequency
        num_harmonics : int, default: 2
            Generate till n-th harmonics
    """
    def __init__(self, sampling_rate, target_frequencies, signal_size, num_harmonics=2):
        self.sampling_rate = sampling_rate
                
        self.reference_frequencies = generate_reference_signals(
            target_frequencies, 
            size=signal_size, 
            sampling_rate=sampling_rate, 
            num_harmonics=num_harmonics
        )

    def predict(self, X):
        r"""
        Predict the label for each trial.

        Args:
            X : ndarray, shape (trial, channels, samples)
                3-dim signal data by trial
        Returns:
            results : array
                Predicted targets
        """
        predicted_class, _, _, _, _ = perform_cca(X, self.reference_frequencies, labels=None)
        return predicted_class
    

def calculate_cca(dat_x, dat_y, time_axis=-2):
    r"""
    Calculate the Canonical Correlation Analysis (CCA).
    This method calculates the canonical correlation coefficient and
    corresponding weights which maximize a correlation coefficient
    between linear combinations of the two specified multivariable
    signals.
    Args:
        dat_x : continuous Data object
            these data should have the same length on the time axis.
        dat_y : continuous Data object
            these data should have the same length on the time axis.
        time_axis : int, optional
            the index of the time axis in ``dat_x`` and ``dat_y``.
    Returns:
        rho : float
            the canonical correlation coefficient.
        w_x, w_y : 1d array
            the weights for mapping from the specified multivariable signals
            to canonical variables.
    Raises:
        AssertionError :
            If:
                * ``dat_x`` and ``dat_y`` is not continuous Data object
                * the length of ``dat_x`` and ``dat_y`` is different on the
                  ``time_axis``
    Dependencies:
        functools : functools package
        np : numpy package
    Reference:
        https://github.com/venthur/wyrm/blob/master/wyrm/processing.py
        http://en.wikipedia.org/wiki/Canonical_correlation
    """

    assert (len(dat_x.data.shape) == len(dat_y.data.shape) == 2 and
            dat_x.data.shape[time_axis] == dat_y.data.shape[time_axis])

    if time_axis == 0 or time_axis == -2:
        x = dat_x.copy()
        y = dat_y.copy()
    else:
        x = dat_x.T.copy()
        y = dat_y.T.copy()

    # calculate covariances and it's inverses
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    n = x.shape[0]
    c_xx = np.dot(x.T, x) / n
    c_yy = np.dot(y.T, y) / n
    c_xy = np.dot(x.T, y) / n
    c_yx = np.dot(y.T, x) / n
    ic_xx = np.linalg.pinv(c_xx)
    ic_yy = np.linalg.pinv(c_yy)
    # calculate w_x
    w, v = np.linalg.eig(functools.reduce(np.dot, [ic_xx, c_xy, ic_yy, c_yx]))
    w_x = v[:, np.argmax(w)].real
    w_x = w_x / np.sqrt(functools.reduce(np.dot, [w_x.T, c_xx, w_x]))
    # calculate w_y
    w, v = np.linalg.eig(functools.reduce(np.dot, [ic_yy, c_yx, ic_xx, c_xy]))
    w_y = v[:, np.argmax(w)].real
    w_y = w_y / np.sqrt(functools.reduce(np.dot, [w_y.T, c_yy, w_y]))
    # calculate rho
    rho = abs(functools.reduce(np.dot, [w_x.T, c_xy, w_y]))
    return rho, w_x, w_y


def find_correlation_cca(signal, reference_signals):
    r"""
    Perform canonical correlation analysis (CCA)
    Args:
        signal : ndarray, shape (channel,time)
            Input signal in time domain
        reference_signals : ndarray, shape (len(flick_freq),2*num_harmonics,time)
            Required sinusoidal reference templates corresponding to the flicker frequency for SSVEP classification
    Returns:
        result : array, size: (reference_signals.shape[0])
            Probability for each reference signals
        wx : array, size: (reference_signals.shape[0],signal.shape[0])
            Wx obtain from CCA
        wy : array, size: (reference_signals.shape[0],signal.shape[0])
            Wy obtain from CCA
    Dependencies:
        np : numpy package
        calculate_cca : function
    """

    result = np.zeros(reference_signals.shape[0])
    wx = np.zeros((reference_signals.shape[0],signal.shape[0]))
    wy = np.zeros((reference_signals.shape[0],reference_signals.shape[1]))
    
    for freq_idx in range(0, reference_signals.shape[0]):
        dat_y = np.squeeze(reference_signals[freq_idx, :, :]).T
        rho, w_x, w_y = calculate_cca(signal.T, dat_y)
        result[freq_idx] = rho
        wx[freq_idx,:] = w_x
        wy[freq_idx,:] = w_y
    return result, wx, wy


def perform_cca(signal, reference_frequencies, labels=None):
    r"""
    Perform canonical correlation analysis (CCA)
    Args:
        signal : ndarray, shape (trial,channel,time) or (trial,channel,segment,time)
            Input signal in time domain
        reference_frequencies : ndarray, shape (len(flick_freq),2*num_harmonics,time)
            Required sinusoidal reference templates corresponding to the flicker frequency for SSVEP classification
        labels : ndarray shape (classes,)
            True labels of `signal`. Index of the classes must be match the sequence of `reference_frequencies`
    Returns:
        predicted_class : ndarray, size: (classes,)
            Predicted classes according to reference_frequencies
        accuracy : double
            If `labels` are given, `accuracy` denote classification accuracy
        predicted_probabilities : ndarray, size: (classes,)
            Predicted probabilities for each target
        wx : array, size: (reference_signals.shape[0],signal.shape[0])
            Wx obtain from CCA
        wy : array, size: (reference_signals.shape[0],signal.shape[0])
            Wy obtain from CCA
    Dependencies:
        confusion_matrix : sklearn.metrics.confusion_matrix
        find_correlation_cca : function
    """

    assert (len(signal.shape) == 3 or len(signal.shape) == 4), "signal shape must be 3 or 4 dimension"

    actual_class = []
    predicted_class = []
    predicted_probabilities = []
    accuracy = None
    Wx = []
    Wy = []

    for trial in range(0, signal.shape[0]):

        if len(signal.shape) == 3:
            if labels is not None:
                actual_class.append(labels[trial])
            tmp_signal = signal[trial, :, :]

            result, wx, wy = find_correlation_cca(tmp_signal, reference_frequencies)
            predicted_class.append(np.argmax(result))
            result = np.around(result, decimals=3, out=None)
            predicted_probabilities.append(result)
            Wx.append(wx)
            Wy.append(wy)

        if len(signal.shape) == 4:
            for segment in range(0, signal.shape[2]):

                if labels is not None:
                    actual_class.append(labels[trial])
                tmp_signal = signal[trial, :, segment, :]

                result, wx, wy = find_correlation_cca(tmp_signal, reference_frequencies)
                predicted_class.append(np.argmax(result))
                result = np.around(result, decimals=3, out=None)
                predicted_probabilities.append(result)
                Wx.append(wx)
                Wy.append(wy)

    actual_class = np.array(actual_class)
    predicted_class = np.array(predicted_class)

    if labels is not None:
        # creating a confusion matrix of true versus predicted classification labels
        c_mat = confusion_matrix(actual_class, predicted_class)
        # computing the accuracy from the confusion matrix
        accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
    
    Wx = np.array(Wx)
    Wy = np.array(Wy)
    
    return predicted_class, accuracy, predicted_probabilities, Wx, Wy
