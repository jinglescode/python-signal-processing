# -*- coding: utf-8 -*-
"""Use CCA for spatial filtering to improve the signal.
"""
import numpy as np
from splearn.cross_decomposition import perform_cca


def cca_spatial_filtering(signal, reference_frequencies):
    r"""
    Use CCA for spatial filtering is to find a spatial filter that maximizes the correlation between the spatially filtered signal and the average evoked response, thereby improving the signal-to-noise ratio of the filtered signal on a single-trial basis.
    Read more: https://github.com/jinglescode/papers/issues/90, https://github.com/jinglescode/papers/issues/89
    Args:
        signal : ndarray, shape (trial,channel,time)
            Input signal in time domain
        reference_frequencies : ndarray, shape (len(flick_freq),2*num_harmonics,time)
            Required sinusoidal reference templates corresponding to the flicker frequency for SSVEP classification
    Returns:
        filtered_signal : ndarray, shape (reference_frequencies.shape[0],signal.shape[0],signal.shape[1],signal.shape[2])
            Signal after spatial filter
    Dependencies:
        np : numpy package
        perform_cca : function
    """
    
    _, _, _, wx, _ = perform_cca(signal, reference_frequencies)
    
    filtered_signal = np.zeros((reference_frequencies.shape[0], signal.shape[0], signal.shape[1], signal.shape[2]))
    
    wx = np.swapaxes(wx, 0, 1)
    swapped_s = np.swapaxes(signal, 1, 2)
    
    for target in range(reference_frequencies.shape[0]):

        this_wx = wx[target]
        this_wx = np.expand_dims(this_wx,1)

        filtered_s = swapped_s * this_wx
        filtered_s = np.swapaxes(filtered_s, 1, 2)

        filtered_signal[target] = filtered_s

    return filtered_signal
