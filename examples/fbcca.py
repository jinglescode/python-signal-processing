# for running locally (may remove this if your path is right)
import os
cwd = os.getcwd()
import sys
path = cwd
sys.path.append(path)

# # imports
from splearn.data import Benchmark
from splearn.cross_decomposition.fbcca import fbcca, fbcca_realtime



# # config
# load_subject_id = 1
# path_to_dataset = "../data/hsssvep"

# # load data
# subject_dataset = Benchmark(root=path_to_dataset, subject_id=load_subject_id)
# print(subject_dataset.data.shape)


# eeg = subject_dataset.data[:, :, 250:500]
# fs = 250
# list_freqs = subject_dataset.stimulus_frequencies
# print("list_freqs", list_freqs)

# # results = fbcca(eeg, list_freqs, fs, num_harms=3, num_fbs=5)
# results = fbcca_realtime(eeg, list_freqs, fs, num_harms=3, num_fbs=5)
# print(results)


import numpy as np

SAMPLE_RATE = 500
t = np.linspace(0,1, num=SAMPLE_RATE)
s = np.sin(2*np.pi*11*t)
s = s[np.newaxis,:]
ss = np.repeat(s, 32, axis=0)
sss = ss[np.newaxis,:]
sss = np.repeat(sss, 8, axis=0)
list_freqs = np.arange(8.0,13.0+1,1)
print("sss", sss.shape)
print("list_freqs", list_freqs.shape)
results = fbcca(sss, list_freqs, SAMPLE_RATE, num_harms=3, num_fbs=5)
print(results)
