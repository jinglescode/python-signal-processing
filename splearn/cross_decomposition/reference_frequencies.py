# -*- coding: utf-8 -*-
"""Generate sinusoidal signals for CCA.
"""
import numpy as np


def get_reference_signals(target_frequencies, size, sampling_rate, num_harmonics=2):
    r"""
    Generating a single sinusoidal template for SSVEP classification
    Args:
        target_frequencies : array
            Frequencies for SSVEP classification
        size : int
            Window/segment length in time samples
        sampling_rate : int
            Sampling frequency
        num_harmonics : int, default: 2
            Generate till n-th harmonics
    Returns:
        reference_signals : ndarray, shape (len(flick_freq),4,time)
            Reference frequency signals
    Example:
        Refer to `generate_reference_signals()`
    Dependencies:
        np : numpy package
    """

    reference_signals = []
    t = np.arange(0, (size/sampling_rate), step=1.0/sampling_rate)

    for i in range(1, num_harmonics+1):
        j = i*2
        reference_signals.append(np.sin(np.pi*j*target_frequencies*t))
        reference_signals.append(np.cos(np.pi*j*target_frequencies*t))

    reference_signals = np.array(reference_signals)
    return reference_signals


def generate_reference_signals(flick_freq, size, sampling_rate, num_harmonics=2):
    r"""
    Generating the required sinusoidal templates for SSVEP classification
    Args:
        flick_freq : array
            Frequencies for SSVEP classification
        size : int
            Window/segment length in time samples
        sampling_rate : int
            Sampling frequency
        num_harmonics : int
            Generate till n-th harmonics
    Returns:
        reference_signals : ndarray, shape (len(flick_freq),2*num_harmonics,time)
            Reference frequency signals
    Example:
        reference_frequencies = generate_reference_signals(
            [5,7.5,10,12], size=4000, sampling_rate=1000, num_harmonics=3)
    Dependencies:
        np : numpy package
        get_reference_frequencies : function
    """

    reference_frequencies = []
    for fr in range(0, len(flick_freq)):
        ref = get_reference_signals(flick_freq[fr], size, sampling_rate, num_harmonics)
        reference_frequencies.append(ref)
    reference_frequencies = np.array(reference_frequencies, dtype='float32')

    return reference_frequencies
