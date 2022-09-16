import os
import numpy as np
import scipy.io as sio
from typing import Tuple

from splearn.data.pytorch_dataset import PyTorchDataset


class JFPM(PyTorchDataset):
    """
    A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials
    Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung
    PLoS One, vol.10, no.10, e140703, 2015. http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703

    This dataset contains 12-class joint frequency-phase modulated steady-state visual evoked potentials (SSVEPs) acquired from 10 subjects used to estimate an online performance of brain-computer interface (BCI) in the reference study (Nakanishi et al., 2015).

    * Number of targets 	    : 12
    * Number of channels 	    : 8
    * Number of sampling points : 1114
    * Number of trials 		    : 15
    * Sampling rate [Hz] 		: 256

    The order of the stimulus frequencies in the EEG data:  
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz

    Download the data: https://github.com/mnakanishi/12JFPM_SSVEP
    """

    def __init__(self, root: str, subject_id: int, verbose: bool = False, file_prefix='S') -> None:

        self.root = root
        self.sampling_rate = 256
        self.data, self.targets = _load_data(self.root, subject_id, verbose, file_prefix)
        self.stimulus_frequencies = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
        self.targets_frequencies = self.stimulus_frequencies[self.targets]

    def __getitem__(self, n: int) -> Tuple[np.ndarray, int]:
        return (self.data[n], self.targets[n])

    def __len__(self) -> int:
        return len(self.data)


def _load_data(root, subject_id, verbose, file_prefix='s'):

    path = os.path.join(root, file_prefix+str(subject_id)+'.mat')
    data_mat = sio.loadmat(path)

    raw_data = data_mat['eeg'].copy()

    num_classes = raw_data.shape[0]
    num_chan = raw_data.shape[1]
    num_trials = raw_data.shape[3]
    sample_rate = 256

    trial_len = int(38+0.135*sample_rate+4*sample_rate) - int(38+0.135*sample_rate)

    filtered_data = np.zeros((num_classes, num_chan, trial_len, num_trials))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                signal_to_filter = np.squeeze(raw_data[target, channel, int(38+0.135*sample_rate):
                                               int(38+0.135*sample_rate+4*sample_rate), 
                                               trial])
                filtered_data[target, channel, :, trial] = signal_to_filter
                
    filtered_data = np.transpose(filtered_data, (0,3,1,2))

    data = []
    targets = []
    for target_id in np.arange(num_classes):
        data.extend(filtered_data[target_id])
        this_target = np.array([target_id]*num_trials)
        targets.extend(this_target)
    
    data = np.array(data)
    targets = np.array(targets)

    if verbose:
        print('Load path:', path)
        print('Data shape', data.shape)
        print('Targets shape', targets.shape)

    return data, targets
