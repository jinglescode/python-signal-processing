# -*- coding: utf-8 -*-
"""A 40-target SSVEP dataset recorded from a single subject.
"""
import numpy as np
from scipy.io import loadmat


class SampleSSVEPData():
    r"""
    A 40-target SSVEP dataset recorded from a single subject.
    
    Data description:
        Original Data shape : (40, 9, 1250, 6) [# of targets, # of channels, # of sampling points, # of blocks]
        Stimulus frequencies : 8.0 - 15.8 Hz with an interval of 0.2 Hz
        Stimulus phases : 0pi, 0.5pi, 1.0pi, and 1.5pi
        Number of channels : 9 (1: Pz, 2: PO5,3: PO3, 4: POz, 5: PO4, 6: PO6, 7: O1, 8: Oz, and 9: O2)
        Number of recording blocks : 6
        Length of an epoch : 5 seconds
        Sampling rate : 250 Hz
    Args:
        path: str
            Path to ssvepdata.mat file
    Reference:
        https://www.pnas.org/content/early/2015/10/14/1508080112.abstract
    """
    def __init__(self, path="./"):
        # Get EEG data
        data = loadmat(path+"data_ssvep.mat")
        data = data["eeg"]
        data = data.transpose([3,0,1,2])
        self.data = data
        
        # Prepare labels
        n_blocks, n_targets, n_channels, n_samples = self.data.shape
        targets = np.tile(np.arange(0, n_targets+0), (1, n_blocks)).squeeze()
        targets = targets.reshape((n_blocks, n_targets))
        self.targets = targets

        self.sampling_rate = 250
        self.channels = ["Pz", "PO5","PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]
    
    def get_data(self):
        r"""
        Data shape: (6, 40, 9, 1250) [# of blocks, # of targets, # of channels, # of sampling points]
        """
        return self.data
    
    def get_targets(self):
        r"""
        Targets shape: (6, 40) [# of blocks, # of targets]
        """
        return self.targets
