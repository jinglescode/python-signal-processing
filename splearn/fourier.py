# -*- coding: utf-8 -*-
"""Fourier analysis is a method for expressing a function as a sum of periodic components, and for recovering the signal from those components.
"""
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt


def fast_fourier_transform(signal, sample_rate, plot=False, **kwargs):
    r"""
    Use Fourier transforms to find the frequency components of a signal buried in noise.
    Reference:  https://www.mathworks.com/help/matlab/ref/fft.html
                https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
    
    Args:
        signal : ndarray, shape (time,)
            Single input signal in time domain
        sample_rate: int
            Sampling frequency
        plot : boolean, optional, default: False
            To plot the single-sided amplitude spectrum
        plot_xlim : array of shape [lower, upper], optional, default: [0, int(`sample_rate`/2)]
            If `plot=True`, set a limit on the X-axis between lower and upper bound
        plot_ylim : array of shape [lower, upper], optional, default: None
            If `plot=True`, set a limit on the Y-axis between lower and upper bound
        plot_label : string, optional, default: ''
            If `plot=True`, text label for this signal in plot, shown in legend
            
    Returns:
        P1 : ndarray
            Frequency domain. Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
            See https://www.mathworks.com/help/matlab/ref/fft.html.
    
    Usage:
        >>> from splearn.data.generate import signal
        >>> 
        >>> s1 = signal(
        >>>     length_seconds=3.5, 
        >>>     sampling_rate=100, 
        >>>     frequencies=[4,7],  
        >>>     plot=True
        >>> )
        >>> 
        >>> p1 = fast_fourier_transform(
        >>>     signal=s1, 
        >>>     sample_rate=100, 
        >>>     plot=True, 
        >>>     plot_xlim=[0, 10]
        >>> )

    Future plans:
        - Expand to n-D array
    """

    plot_xlim = kwargs['plot_xlim'] if 'plot_xlim' in kwargs else [0, int(sample_rate/2)]
    plot_ylim = kwargs['plot_ylim'] if 'plot_ylim' in kwargs else None
    plot_label = kwargs['plot_label'] if 'plot_label' in kwargs else ''

    signal_length = signal.shape[0]

    if signal_length % 2 != 0:
        signal_length = signal_length+1

    y = fft(signal)
    p2 = np.abs(y/signal_length)
    p1 = p2[0:round(signal_length/2+1)]
    p1[1:-1] = 2*p1[1:-1]

    if plot:
        f = sample_rate*np.arange(0, (signal_length/2)+1)/signal_length
        plt.plot(f, p1, label=plot_label)
        plt.title('Signal in frequency domain after performing FFT')
        plt.xlabel('Frequencies ('+ " to ".join(map(str, plot_xlim)) +' Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(plot_xlim)
        if plot_ylim is not None:
            plt.ylim(plot_ylim)

        if plot_label != '':
            plt.legend()

    return p1


if __name__ == "__main__":
    
    from splearn.data.generate import signal

    s1 = signal(
        length_seconds=3.5, 
        sampling_rate=100, 
        frequencies=[4,7],  
        plot=True
    )

    p1 = fast_fourier_transform(
        signal=s1, 
        sample_rate=100, 
        plot=True, 
        plot_xlim=[0, 15],
    )
