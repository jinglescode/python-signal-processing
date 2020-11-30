# -*- coding: utf-8 -*-
"""Digital filter bandpass zero-phase implementation (filtfilt). Apply a digital filter forward and backward to a signal.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt, freqz
from splearn.fourier import fast_fourier_transform


def butter_bandpass(signal, lowcut, highcut, sample_rate, type="sos", order=4, plot=False, **kwargs):
    r"""
    Design a `order`th-order bandpass Butterworth filter with a cutoff frequency between `lowcut`-Hz and `highcut`-Hz, which, for data sampled at `sample_rate`-Hz.

    Reference:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html

    Args:
        signal : ndarray, shape (time,) or (channel,time) or (trial,channel,time)
            Input signal (1D/2D/3D), where last axis is time samples.
        lowcut : int
            Lower bound filter
        highcut : int
            Upper bound filter
        sample_rate : int
            Sampling frequency
        type: string, optional, default: sos
            Type of output: numerator/denominator (‘ba’), or second-order sections (‘sos’). 
            Default is ‘ba’ for backwards compatibility, but ‘sos’ should be used for general-purpose filtering.
        order : int, optional, default: 4
            Order of the filter
        plot : boolean, optional, default: False
            Plot signal and filtered signal in frequency domain
        plot_xlim : array of shape [lower, upper], optional, default: [lowcut-10 if (lowcut-10) > 0 else 0, highcut+20]
            If `plot=True`, set a limit on the X-axis between lower and upper bound
        plot_ylim : array of shape [lower, upper], optional, default: None
            If `plot=True`, set a limit on the Y-axis between lower and upper bound
    
    Returns:
        y : ndarray
            Filtered signal that has same shape in input `signal`

    Usage:
        >>> from splearn.data.generate import signal
        >>>
        >>> signal_1d = signal(
        >>>     length_seconds=4, 
        >>>     sampling_rate=100, 
        >>>     frequencies=[4,7,11,17,40, 50],
        >>>     plot=True
        >>> )
        >>> print('signal_1d.shape', signal_1d.shape)
        >>> 
        >>> signal_2d = signal(
        >>>     length_seconds=4, 
        >>>     sampling_rate=100, 
        >>>     frequencies=[[4,7,11,17,40, 50],[1, 3]],
        >>>     plot=True
        >>> )
        >>> print('signal_2d.shape', signal_2d.shape)
        >>> 
        >>> signal_3d = np.expand_dims(s1, 0)
        >>> print('signal_3d.shape', signal_3d.shape)
        >>> 
        >>> signal_1d_filtered = butter_bandpass(
        >>>     signal=signal_1d, 
        >>>     lowcut=5, 
        >>>     highcut=20, 
        >>>     sample_rate=100,
        >>>     plot=True,
        >>> )
        >>> print('signal_1d_filtered.shape', signal_1d_filtered.shape)
        >>> 
        >>> signal_2d_filtered = butter_bandpass(
        >>>     signal=signal_2d, 
        >>>     lowcut=5, 
        >>>     highcut=20, 
        >>>     sample_rate=100,
        >>>     type='sos',
        >>>     order=4, 
        >>>     plot=True,
        >>>     plot_xlim=[3,20]
        >>> )
        >>> print('signal_2d_filtered.shape', signal_2d_filtered.shape)
        >>> 
        >>> signal_3d_filtered = butter_bandpass(
        >>>     signal=signal_3d, 
        >>>     lowcut=5, 
        >>>     highcut=20, 
        >>>     sample_rate=100,
        >>>     type='ba',
        >>>     order=4, 
        >>>     plot=True,
        >>>     plot_xlim=[0,40]
        >>> )
        >>> print('signal_3d_filtered.shape', signal_3d_filtered.shape)
    """

    dim = len(signal.shape)-1

    if type == 'ba':
        b, a = _butter_bandpass(lowcut, highcut, sample_rate, order)
        y = filtfilt(b, a, signal)
    else:
        sos = _butter_bandpass(lowcut, highcut, sample_rate,
                            order=order, output='sos')
        y = sosfiltfilt(sos, signal, axis=dim)
    
    if plot:
        tmp_x = signal
        tmp_y = y
        if dim == 1:
            tmp_x = signal[0]
            tmp_y = y[0]
        elif dim == 2:
            tmp_x = signal[0, 0]
            tmp_y = y[0, 0]

        if type == 'ba':
            # plot frequency response of filter
            w, h = freqz(b, a)
            plt.plot((sample_rate * 0.5 / np.pi) * w,
                    abs(h), label="order = %d" % order)
            plt.plot([0, 0.5 * sample_rate], [np.sqrt(0.5), np.sqrt(0.5)],
                    '--', label='sqrt(0.5)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')
            low = max(0, lowcut-(sample_rate/100))
            high = highcut+(sample_rate/100)
            plt.xlim([low, high])
            plt.ylim([0, 1.2])
            plt.title('Frequency response of filter - lowcut:' +
                    str(lowcut)+', highcut:'+str(highcut))
            plt.show()

        plot_xlim = kwargs['plot_xlim'] if 'plot_xlim' in kwargs else [lowcut-10 if (lowcut-10) > 0 else 0, highcut+20]
        plot_ylim = kwargs['plot_ylim'] if 'plot_ylim' in kwargs else None

        # frequency domain
        fast_fourier_transform(
            tmp_x, 
            sample_rate, 
            plot=True, 
            plot_xlim=plot_xlim, 
            plot_ylim=plot_ylim,
            plot_label='Signal'
        )
        fast_fourier_transform(
            tmp_y, 
            sample_rate, 
            plot=True, 
            plot_xlim=plot_xlim, 
            plot_ylim=plot_ylim,
            plot_label='Filtered'
        )

        plt.title('Signal and filtered signal in frequency domain, type:' + type + ',lowcut:' + str(lowcut) + ',highcut:' + str(highcut) + ',order:' + str(order))
        plt.legend()
        plt.show()

    return y


def _butter_bandpass(lowcut, highcut, sample_rate, order=4, output='ba'):
    r"""
    Create a Butterworth bandpass filter. Design an Nth-order digital or analog Butterworth filter and return the filter coefficients.
    Reference:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    Args:
        lowcut : int
            Lower bound filter
        highcut : int
            Upper bound filter
        sample_rate : int
            Sampling frequency
        order : int, default: 4
            Order of the filter
        output : string, default: ba
            Type of output {‘ba’, ‘zpk’, ‘sos’}. Type of output: numerator/denominator (‘ba’), pole-zero (‘zpk’), or second-order sections (‘sos’). 
            Default is ‘ba’ for backwards compatibility, but ‘sos’ should be used for general-purpose filtering.
    Returns:
        butter : ndarray
            Scipy butterworth filter
    Dependencies:
        butter : scipy.signal.butter
    """
    nyq = sample_rate * 0.5
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', output=output)


if __name__ == "__main__":

    from splearn.data.generate import signal

    signal_1d = signal(
        length_seconds=4, 
        sampling_rate=100, 
        frequencies=[4,7,11,17,40, 50],
        plot=True
    )
    print('signal_1d.shape', signal_1d.shape)

    signal_2d = signal(
        length_seconds=4, 
        sampling_rate=100, 
        frequencies=[[4,7,11,17,40, 50],[1, 3]],
        plot=True
    )
    print('signal_2d.shape', signal_2d.shape)

    signal_3d = np.expand_dims(s1, 0)
    print('signal_3d.shape', signal_3d.shape)

    signal_1d_filtered = butter_bandpass(
        signal=signal_1d, 
        lowcut=5, 
        highcut=20, 
        sample_rate=100,
        plot=True,
    )
    print('signal_1d_filtered.shape', signal_1d_filtered.shape)

    signal_2d_filtered = butter_bandpass(
        signal=signal_2d, 
        lowcut=5, 
        highcut=20, 
        sample_rate=100,
        type='sos',
        order=4, 
        plot=True,
        plot_xlim=[3,20]
    )
    print('signal_2d_filtered.shape', signal_2d_filtered.shape)

    signal_3d_filtered = butter_bandpass(
        signal=signal_3d, 
        lowcut=5, 
        highcut=20, 
        sample_rate=100,
        type='ba',
        order=4, 
        plot=True,
        plot_xlim=[0,40]
    )
    print('signal_3d_filtered.shape', signal_3d_filtered.shape)
