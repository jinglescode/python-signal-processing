# for running locally (may remove this if your path is right)
import os
from re import sub
cwd = os.getcwd()
import sys
path = cwd
sys.path.append(path)

import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt

from splearn.data import JFPM

# config
load_subject_id = 1
path_to_dataset = "../data/jfpm"

# load data
subject_dataset = JFPM(root=path_to_dataset, subject_id=load_subject_id)
print(subject_dataset.data.shape)
print(subject_dataset.targets.shape)
print(subject_dataset.sampling_rate)
