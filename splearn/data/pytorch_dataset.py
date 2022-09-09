from torch.utils.data import Dataset
import numpy as np


class PyTorchDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.data = self.data.astype(np.float32)
        self.targets = targets
        self.channel_names = None

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def set_data_targets(self, data: [] = None, targets: [] = None) -> None:
        if data is not None:
            self.data = data.copy()
        if targets is not None:
            self.targets = targets.copy()
            self.targets = self.targets.astype(int)

    def set_channel_names(self,channel_names):
        self.channel_names = channel_names
    
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

class PyTorchDataset2Views(Dataset):
    def __init__(self, data_view1, data_view2, targets):
        self.data_view1 = data_view1.astype(np.float32)
        self.data_view2 = data_view2.astype(np.float32)
        self.targets = targets

    def __getitem__(self, index):
        return self.data_view1[index], self.data_view2[index], self.targets[index]

    def __len__(self):
        return len(self.data_view1)