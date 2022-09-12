import os
import numpy as np
import scipy.io as sio
from typing import Tuple

from splearn.data.pytorch_dataset import PyTorchDataset


class Beta(PyTorchDataset):
    """
    BETA: A Large Benchmark Database Toward SSVEP-BCI Application
    Bingchuan Liu, Xiaoshan Huang, Yijun Wang, Xiaogang Chen and Xiaorong Gao
    https://www.frontiersin.org/articles/10.3389/fnins.2020.00627/full
    Sampling rate: 250 Hz
    stimulus frequencies: [8.6,8.8,9.,9.2,9.4,9.6,9.8,10.,10.2,10.4,10.6,10.8,11.,11.2,11.4,11.6,11.8,12.,12.2,12.4,12.6,12.8,13.,13.2,13.4,13.6,13.8,14.,14.2,14.4,14.6,14.8,15.,15.2,15.4,15.6,15.8,8.,8.2,8.4]
    channel_names ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    """
    def __init__(self, root: str, subject_id: int, verbose: bool = False, file_prefix='neuroscan_S') -> None:
        self.root = root
        self.sample_rate = 1000
        self.data, self.targets, self.channel_names, self.stimulus_frequencies = _load_data(self.root, subject_id, verbose, file_prefix)        
        self.sampling_rate = 250
        self.targets_frequencies = self.stimulus_frequencies[self.targets]

    def __getitem__(self, n: int) -> Tuple[np.ndarray, int]:
        return (self.data[n], self.targets[n])

    def __len__(self) -> int:
        return len(self.data)

def _load_data(root, subject_id, verbose, file_prefix='neuroscan_S'):
    path = os.path.join(root, file_prefix+str(subject_id)+'.mat')
    data_mat = sio.loadmat(path)
    
    mat_data = data_mat['data'].copy()
    raw_data = mat_data[0][0][0] # this is raw data, shape: (64, 750, 4, 40)
    raw_data = np.transpose(raw_data, (3,2,0,1))

    channel_names = []
    raw_channels = mat_data[0][0][1][0][0][3]
    for i in raw_channels:
        channel_names.append(i[3][0])

    stimulus_frequencies = mat_data[0][0][1][0][0][4][0]
    
    data = []
    targets = []
    for target_id in np.arange(raw_data.shape[0]):
        data.extend(raw_data[target_id])
        
        this_target = np.array([target_id]*raw_data.shape[1])
        targets.extend(this_target)
    
    data = np.array(data) # (160, 64, 750)
    targets = np.array(targets)

    # Each trial comprises 0.5-s data before the event onset and 0.5-s data after the time window of 2 s or 3 s. For S1-S15, the time window is 2 s and the trial length is 3 s, whereas for S16-S70 the time window is 3 s and the trial length is 4 s.
    # Trials began with a 0.5s cue (a red square covering the target) for gaze shift, which was followed by flickering on all the targets, and ended with a rest time of 0.5 s.
    # We remove the 0.5s from start and end
    # We limit all trials from all subjects to 2 seconds
    data = np.array(data)[:,:,125:625]

    if verbose:
        print('Load path:', path)
        print('Data shape', data.shape)
        print('Targets shape', targets.shape)   

    return data, targets, channel_names, stimulus_frequencies
