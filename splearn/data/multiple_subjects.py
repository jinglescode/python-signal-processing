import numpy as np
from splearn.data.pytorch_dataset import PyTorchDataset


class MultipleSubjects(PyTorchDataset):
    def __init__(self, 
        dataset: PyTorchDataset, 
        root: str, 
        subject_ids: [], 
        verbose: bool = False, 
    ) -> None:
        
        self.root = root
        self.subject_ids = subject_ids
        
        self._load_multiple(root, subject_ids, verbose)
        self.targets_frequencies = self.stimulus_frequencies[self.targets]
    
    def _load_multiple(self, root, subject_ids: [], verbose: bool = False) -> None:
        is_first = True
        
        for subject_i in range(len(subject_ids)):
            
            subject_id = subject_ids[subject_i]
            print('Load subject:', subject_id)
            
            subject_dataset = HSSSVEP(root="../data/hsssvep", subject_id=subject_id)
            
            sub_data = subject_dataset.data
            sub_targets = subject_dataset.targets
            
            if is_first:
                self.data = np.zeros((len(subject_ids), sub_data.shape[0], sub_data.shape[1], sub_data.shape[2]))
                self.targets = np.zeros((len(subject_ids), sub_targets.shape[0]))
                self.sampling_rate = subject_dataset.sampling_rate
                self.stimulus_frequencies = subject_dataset.stimulus_frequencies
                self.channel_names = subject_dataset.channel_names
                is_first = False

            self.data[subject_i, :, :, :] = sub_data
            self.targets[subject_i] = sub_targets
            
        self.targets = self.targets.astype(np.int32)
