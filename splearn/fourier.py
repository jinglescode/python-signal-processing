# -*- coding: utf-8 -*-
"""Fourier analysis is a method for expressing a function as a sum of periodic components, and for recovering the signal from those components.
"""
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt


def fast_fourier_transform(signal, sampling_rate, plot=False, **kwargs):
    r"""
    Use Fourier transforms to find the frequency components of a signal buried in noise.
    
    Args:
        signal : ndarray, shape (time,) or (channel,time) or (trial,channel,time)
            Single input signal in time domain
        sampling_rate: int
            Sampling frequency
        plot : boolean, optional, default: False
            To plot the single-sided amplitude spectrum
        plot_xlim : array of shape [lower, upper], optional, default: [0, int(`sampling_rate`/2)]
            If `plot=True`, set a limit on the X-axis between lower and upper bound
        plot_ylim : array of shape [lower, upper], optional, default: None
            If `plot=True`, set a limit on the Y-axis between lower and upper bound
        plot_label : string, optional, default: ''
            If `plot=True`, text label for this signal in plot, shown in legend
        plot_line_freq : int or float or list, option, default: None
            If `plot=True`, plot a vertical line to mark the target frequency. If a list is given, will plot multiple lines.
    Returns:
        P1 : ndarray
            Frequency domain. Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L. 
            See https://www.mathworks.com/help/matlab/ref/fft.html
    Usage:
        >>> from splearn.data.generate import generate_signal
        >>> from splearn.fourier import fast_fourier_transform
        >>> 
        >>> s1 = generate_signal(
        >>>     length_seconds=3.5, 
        >>>     sampling_rate=100, 
        >>>     frequencies=[4,7],  
        >>>     plot=True
        >>> )
        >>> 
        >>> p1 = fast_fourier_transform(
        >>>     signal=s1, 
        >>>     sampling_rate=100, 
        >>>     plot=True, 
        >>>     plot_xlim=[0, 10],
        >>>     plot_line_freq=7
        >>> )
    Reference:
        - https://www.mathworks.com/help/matlab/ref/fft.html
        - https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
    """
    
    plot_xlim = kwargs['plot_xlim'] if 'plot_xlim' in kwargs else [0, int(sampling_rate/2)]
    plot_ylim = kwargs['plot_ylim'] if 'plot_ylim' in kwargs else None
    plot_label = kwargs['plot_label'] if 'plot_label' in kwargs else ''
    plot_line_freq = kwargs['plot_line_freq'] if 'plot_line_freq' in kwargs else None
    plot_label = kwargs['plot_label'] if 'plot_label' in kwargs else ''
    
    fft_p1 = None
    
    if len(signal.shape) == 1:
        fft_p1 = _fast_fourier_transform(signal, sampling_rate)
        fft_p1 = np.expand_dims(fft_p1,0)
        
    if len(signal.shape) == 2:
        for ch in range(signal.shape[0]):
            fft_c = _fast_fourier_transform(signal[ch, :], sampling_rate=sampling_rate)

            if fft_p1 is None:
                fft_p1 = np.zeros((signal.shape[0], fft_c.shape[0]))

            fft_p1[ch] = fft_c
    
    if len(signal.shape) == 3:
        for trial in range(signal.shape[0]):
            for ch in range(signal.shape[1]):
                fft_c = _fast_fourier_transform(signal[trial, ch, :], sampling_rate=sampling_rate)

                if fft_p1 is None:
                    fft_p1 = np.zeros((signal.shape[0], signal.shape[1], fft_c.shape[0]))
                
                fft_p1[trial,ch,:] = fft_c
    
    if plot:
        signal_length = signal.shape[ len(signal.shape)-1 ]
        f = sampling_rate*np.arange(0, (signal_length/2)+1)/signal_length
        
        if len(fft_p1.shape) == 3:
            means = np.mean(fft_p1, 0)
            stds = np.std(fft_p1, 0)
            for c in range(fft_p1.shape[1]):
                plt.plot(f, means[c], label=plot_label)
                plt.xlim(plot_xlim)
                plt.fill_between(f, means[c]-stds[c],means[c]+stds[c],alpha=.1)
        else:
            for c in range(fft_p1.shape[0]):
                plt.plot(f, fft_p1[c], label=plot_label)
                plt.xlim(plot_xlim)
            
        if plot_ylim is not None:
            plt.ylim(plot_ylim)

        if plot_label != '':
            plt.legend()

        if plot_line_freq is not None:
            if isinstance(plot_line_freq, list):
                for i in plot_line_freq:
                    plt.axvline(x=i, color='r', linewidth=1.5)
            else:
                plt.axvline(x=plot_line_freq, color='r', linewidth=1.5)
    
    if len(signal.shape) == 1:
        fft_p1 = fft_p1[0]
        
    return fft_p1

def _fast_fourier_transform(signal, sampling_rate):
    r"""
    Use Fourier transforms to find the frequency components of a signal buried in noise.
    
    Args:
        signal : ndarray, shape (time,)
            Single input signal in time domain
        sampling_rate: int
            Sampling frequency
    Returns:
        P1 : ndarray
            Frequency domain. Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
            See https://www.mathworks.com/help/matlab/ref/fft.html.
    Usage:
        See `fast_fourier_transform`
    Reference:  
        - https://www.mathworks.com/help/matlab/ref/fft.html
        - https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
    """
    
    signal_length = signal.shape[0]

    if signal_length % 2 != 0:
        signal_length = signal_length+1

    y = fft(signal)
    p2 = np.abs(y/signal_length)
    p1 = p2[0:round(signal_length/2+1)]
    p1[1:-1] = 2*p1[1:-1]

    return p1


if __name__ == "__main__":
    
    from splearn.data.generate import generate_signal
    from splearn.fourier import fast_fourier_transform

    s1 = generate_signal(
        length_seconds=3.5, 
        sampling_rate=100, 
        frequencies=[4,7],  
        plot=True
    )

    p1 = fast_fourier_transform(
        signal=s1, 
        sampling_rate=100, 
        plot=True, 
        plot_xlim=[0, 10],
        plot_line_freq=7
    )
