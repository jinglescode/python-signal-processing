"""
Created on Wed Oct 30 10:17:50 2019
@author: ALU
Steady-state visual evoked potentials (SSVEPs) detection using the filter
bank canonical correlation analysis (FBCCA)-based method [1].
function results = test_fbcca(eeg, list_freqs, fs, num_harms, num_fbs)
Input:
  eeg             : Input eeg data 
                    (# of targets, # of channels, Data length [sample])
  list_freqs      : List for stimulus frequencies
  fs              : Sampling frequency
  num_harms       : # of harmonics
  num_fbs         : # of filters in filterbank analysis
Output:
  results         : The target estimated by this method
Reference:
  [1] X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao,
      "Filter bank canonical correlation analysis for implementing a 
       high-speed SSVEP-based brain-computer interface",
      J. Neural Eng., vol.12, 046008, 2015.
"""
from sklearn.cross_decomposition import CCA
from .filterbank import filterbank
from scipy.stats import pearsonr
import numpy as np

def fbcca(eeg, list_freqs, fs, num_harms=3, num_fbs=5):
    
    fb_coefs = np.power(np.arange(1,num_fbs+1),(-1.25)) + 0.25
    
    num_targs, _, num_smpls = eeg.shape  #40 taget (means 40 fre-phase combination that we want to predict)
    y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms)
    cca = CCA(n_components=1) #initilize CCA
    
    # result matrix
    r = np.zeros((num_fbs,num_targs))
    print("r", r.shape)
    results = np.zeros(num_targs)
    
    for targ_i in range(num_targs):
        test_tmp = np.squeeze(eeg[targ_i, :, :])  #deal with one target a time
        for fb_i in range(num_fbs):  #filter bank number, deal with different filter bank
             testdata = filterbank(test_tmp, fs, fb_i)  #data after filtering
             for class_i in range(num_targs):
                 refdata = np.squeeze(y_ref[class_i, :, :])   #pick corresponding freq target reference signal
                 print(111, testdata.T.shape, refdata.T.shape)
                 test_C, ref_C = cca.fit_transform(testdata.T, refdata.T)
                 # len(row) = len(observation), len(column) = variables of each observation
                 # number of rows should be the same, so need transpose here
                 # output is the highest correlation linear combination of two sets
                 print(222, test_C.shape, ref_C.shape)
                 r_tmp, _ = pearsonr(np.squeeze(test_C), np.squeeze(ref_C)) #return r and p_value, use np.squeeze to adapt the API 
                 print(333, fb_i, class_i, r_tmp)
                 r[fb_i, class_i] = r_tmp
                 print(444)
        
        print(333, fb_coefs.shape, r.shape)
        rho = np.dot(fb_coefs, r)  #weighted sum of r from all different filter banks' result
        tau = np.argmax(rho)  #get maximum from the target as the final predict (get the index)
        results[targ_i] = tau #index indicate the maximum(most possible) target
    return results

'''
Generate reference signals for the canonical correlation analysis (CCA)
-based steady-state visual evoked potentials (SSVEPs) detection [1, 2].
function [ y_ref ] = cca_reference(listFreq, fs,  nSmpls, nHarms)
Input:
  listFreq        : List for stimulus frequencies
  fs              : Sampling frequency
  nSmpls          : # of samples in an epoch
  nHarms          : # of harmonics
Output:
  y_ref           : Generated reference signals
                   (# of targets, 2*# of channels, Data length [sample])
Reference:
  [1] Z. Lin, C. Zhang, W. Wu, and X. Gao,
      "Frequency Recognition Based on Canonical Correlation Analysis for 
       SSVEP-Based BCI",
      IEEE Trans. Biomed. Eng., 54(6), 1172-1176, 2007.
  [2] G. Bin, X. Gao, Z. Yan, B. Hong, and S. Gao,
      "An online multi-channel SSVEP-based brain-computer interface using
       a canonical correlation analysis method",
      J. Neural Eng., 6 (2009) 046002 (6pp).
'''      
def cca_reference(list_freqs, fs, num_smpls, num_harms=3):
    
    num_freqs = len(list_freqs)
    tidx = np.arange(1,num_smpls+1)/fs #time index
    
    y_ref = np.zeros((num_freqs, 2*num_harms, num_smpls))
    for freq_i in range(num_freqs):
        tmp = []
        for harm_i in range(1,num_harms+1):
            stim_freq = list_freqs[freq_i]  #in HZ
            # Sin and Cos
            tmp.extend([np.sin(2*np.pi*tidx*harm_i*stim_freq),
                       np.cos(2*np.pi*tidx*harm_i*stim_freq)])
        y_ref[freq_i] = tmp # 2*num_harms because include both sin and cos
    
    return y_ref


'''
Base on fbcca, but adapt to our input format
'''   
def fbcca_realtime(data, list_freqs, fs, num_harms=3, num_fbs=5):
    
    fb_coefs = np.power(np.arange(1,num_fbs+1),(-1.25)) + 0.25
    
    num_targs = len(list_freqs)
    _, num_smpls = data.shape
    
    y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms)
    cca = CCA(n_components=1) #initialize CCA
    
    # result matrix
    r = np.zeros((num_fbs,num_targs))
    
    for fb_i in range(num_fbs):  #filter bank number, deal with different filter bank
        testdata = filterbank(data, fs, fb_i)  #data after filtering
        for class_i in range(num_targs):
            refdata = np.squeeze(y_ref[class_i, :, :])   #pick corresponding freq target reference signal
            test_C, ref_C = cca.fit_transform(testdata.T, refdata.T)
            r_tmp, _ = pearsonr(np.squeeze(test_C), np.squeeze(ref_C)) #return r and p_value
            if r_tmp == np.nan:
                r_tmp=0
            r[fb_i, class_i] = r_tmp
    
    rho = np.dot(fb_coefs, r)  #weighted sum of r from all different filter banks' result
    print(rho) #print out the correlation
    result = np.argmax(rho)  #get maximum from the target as the final predict (get the index), and index indicates the maximum entry(most possible target)
    ''' Threshold '''
    THRESHOLD = 2.1
    if abs(rho[result])<THRESHOLD:  #2.587=np.sum(fb_coefs*0.8) #2.91=np.sum(fb_coefs*0.9) #1.941=np.sum(fb_coefs*0.6)
        return 999 #if the correlation isn't big enough, do not return any command
    else:
        return result