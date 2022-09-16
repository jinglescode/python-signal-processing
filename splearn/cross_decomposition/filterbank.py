import warnings
import scipy.signal
import numpy as np

def filterbank(eeg, fs, idx_fb):    
    if idx_fb == None:
        warnings.warn('stats:filterbank:MissingInput '\
                      +'Missing filter index. Default value (idx_fb = 0) will be used.')
        idx_fb = 0
    elif (idx_fb < 0 or 9 < idx_fb):
        raise ValueError('stats:filterbank:InvalidInput '\
                          +'The number of sub-bands must be 0 <= idx_fb <= 9.')
            
    if (len(eeg.shape)==2):
        num_chans = eeg.shape[0]
        num_trials = 1
    else:
        num_chans, _, num_trials = eeg.shape
    
    # Nyquist Frequency = Fs/2N
    Nq = fs/2
    
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Wp = [passband[idx_fb]/Nq, 90/Nq]
    Ws = [stopband[idx_fb]/Nq, 100/Nq]
    [N, Wn] = scipy.signal.cheb1ord(Wp, Ws, 3, 40) # band pass filter StopBand=[Ws(1)~Ws(2)] PassBand=[Wp(1)~Wp(2)]
    [B, A] = scipy.signal.cheby1(N, 0.5, Wn, 'bandpass') # Wn passband edge frequency
    
    y = np.zeros(eeg.shape)
    if (num_trials == 1):
        for ch_i in range(num_chans):
            #apply filter, zero phass filtering by applying a linear filter twice, once forward and once backwards.
            # to match matlab result we need to change padding length
            y[ch_i, :] = scipy.signal.filtfilt(B, A, eeg[ch_i, :], padtype = 'odd', padlen=3*(max(len(B),len(A))-1))
        
    else:
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[ch_i, :, trial_i] = scipy.signal.filtfilt(B, A, eeg[ch_i, :, trial_i], padtype = 'odd', padlen=3*(max(len(B),len(A))-1))
           
    return y
        
        
if __name__ == '__main__':
    from scipy.io import loadmat
    
    D = loadmat("../data/sample/ssvep.mat")
    eeg = D['eeg']
    eeg = eeg[:, :, (33):(33+125), :]
    eeg = eeg[:,:,:,0] #first bank
    eeg = eeg[0, :, :] #first target
    
    y1 = filterbank(eeg, 250, 0)
    y2 = filterbank(eeg, 250, 9)
    
    y1_from_matlab = loadmat("y1_from_matlab.mat")['y1']
    y2_from_matlab = loadmat("y2_from_matlab.mat")['y2']

    dif1 = y1 - y1_from_matlab
    dif2 = y2 - y2_from_matlab    
    
    print("Difference between matlab and python = ", np.sum(dif1))
    print("Difference between matlab and python = ", np.sum(dif2))
    