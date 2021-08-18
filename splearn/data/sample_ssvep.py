# -*- coding: utf-8 -*-
"""A 40-target SSVEP dataset recorded from a single subject.
"""
import numpy as np
from scipy.io import loadmat
import os


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
        path: str, default: None
            Path to ssvepdata.mat file
    Usage:
            >>> from splearn.cross_decomposition.trca import TRCA
            >>> from splearn.data.sample_ssvep import SampleSSVEPData
            >>> 
            >>> data = SampleSSVEPData()
            >>> eeg = data.get_data()
            >>> labels = data.get_targets()
            >>> print("eeg.shape:", eeg.shape)
            >>> print("labels.shape:", labels.shape)
    Reference:
        https://www.pnas.org/content/early/2015/10/14/1508080112.abstract
    """
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")
        
        # Get EEG data
        data = loadmat(os.path.join(path,"ssvep.mat"))
        data = data["eeg"]
        data = data.transpose([3,0,1,2])
        self.data = data
        
        # Prepare targets
        n_blocks, n_targets, n_channels, n_samples = self.data.shape
        targets = np.tile(np.arange(0, n_targets+0), (1, n_blocks))
        targets = targets.reshape((n_blocks, n_targets))
        self.targets = targets
        
        # Prepare targets frequencies
        self.stimulus_frequencies = np.array([8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,8.6,9.6,10.6,11.6,12.6,13.6,14.6,15.6,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8])
        
        targets_frequencies = np.tile(self.stimulus_frequencies, (1, n_blocks))
        targets_frequencies = targets_frequencies.reshape((n_blocks, n_targets))
        self.targets_frequencies = targets_frequencies

        self.sampling_rate = 250
        self.channels = ["Pz", "PO5","PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]
    
    def get_data(self):
        r"""
        Data shape: (6, 40, 9, 1250) [# of blocks, # of targets, # of channels, # of sampling points]
        """
        return self.data
    
    def get_targets(self):
        r"""
        Targets index from 0 to 39. Shape: (6, 40) [# of blocks, # of targets]
        """
        return self.targets
    
    def get_stimulus_frequencies(self):
        r"""
        A list of frequencies of each stimulus:
        [8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,8.6,9.6,10.6,11.6,12.6,13.6,14.6,15.6,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8]
        """
        return self.stimulus_frequencies
    
    def get_targets_frequencies(self):
        r"""
        Targets by frequencies, range between 8.0 Hz to 15.8 Hz.
        Shape: (6, 40) [# of blocks, # of targets]
        """
        return self.targets_frequencies


if __name__ == "__main__":
    from splearn.data.sample_ssvep import SampleSSVEPData
    
    data = SampleSSVEPData()
    eeg = data.get_data()
    labels = data.get_targets()
    print("eeg.shape:", eeg.shape)
    print("labels.shape:", labels.shape)
