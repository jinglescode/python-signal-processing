from scipy.signal import filtfilt, iirnotch

def notch_filter(data, sampling_rate=1000, notch_freq=50.0, quality_factor=30.0):
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, sampling_rate)
    data_notched = filtfilt(b_notch, a_notch, data)
    return data_notched