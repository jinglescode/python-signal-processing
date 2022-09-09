import numpy as np
from sklearn.model_selection import StratifiedKFold
from splearn.data.pytorch_dataset import PyTorchDataset


class MultipleSubjects(PyTorchDataset):
    def __init__(
        self, 
        dataset: PyTorchDataset, 
        root: str, 
        subject_ids: [], 
        func_preprocessing=None,
        func_get_train_val_test_dataset=None,
        verbose: bool = False, 
    ) -> None:
        
        self.root = root
        self.subject_ids = subject_ids
        
        self._load_multiple(root, dataset, subject_ids, func_preprocessing, verbose)
        self.targets_frequencies = self.stimulus_frequencies[self.targets]
        
        self.func_get_train_val_test_dataset = func_get_train_val_test_dataset
    
    def _load_multiple(self, root, dataset: PyTorchDataset, subject_ids: [], func_preprocessing, verbose: bool = False) -> None:
        is_first = True
        
        for subject_i in range(len(subject_ids)):
            
            subject_id = subject_ids[subject_i]
            print('Load subject:', subject_id)
            
            subject_dataset = dataset(root=root, subject_id=subject_id)
            
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
        
        if func_preprocessing is not None:
            func_preprocessing(self)
        
    def set_data(self, x):
        self.data = x
        
    def set_targets(self, targets):
        self.targets = targets
    
    def get_subject(self, subject_id):
        index = list(self.subject_ids).index(subject_id)
        return self.data[index], self.targets[index]
        
    def dataset_split_stratified(self, X, y, k=0, n_splits=3, seed=71, shuffle=True):
        skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=shuffle)
        split_data = skf.split(X, y)

        for idx, value in enumerate(split_data):

            if k != idx:
                continue
            else:
                train_index, test_index = value

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                return (X_train, y_train), (X_test, y_test)
    
    def get_train_val_test_dataset(self, **kwargs):
        if self.func_get_train_val_test_dataset is None:
            return self._leave_one_subject_out(**kwargs)
        else:
            return self.func_get_train_val_test_dataset(self, **kwargs)
            
    def _leave_one_subject_out(self, **kwargs):
        
        test_subject_id = kwargs["test_subject_id"] if "test_subject_id" in kwargs else 1
        kfold_k = kwargs["kfold_k"] if "kfold_k" in kwargs else 0
        kfold_split = kwargs["kfold_split"] if "kfold_split" in kwargs else 3            

        # get test data
        # test_sub_idx = self.subject_ids.index(test_subject_id)
        test_sub_idx = np.where(self.subject_ids == test_subject_id)[0][0]
        selected_subject_data = self.data[test_sub_idx]
        selected_subject_targets = self.targets[test_sub_idx]
        test_dataset = PyTorchDataset(selected_subject_data, selected_subject_targets)

        # get train val data
        indices = np.arange(self.data.shape[0])
        train_val_data = self.data[indices!=test_sub_idx, :, :, :]
        train_val_data = train_val_data.reshape((train_val_data.shape[0]*train_val_data.shape[1], train_val_data.shape[2], train_val_data.shape[3]))
        train_val_targets = self.targets[indices!=test_sub_idx, :]
        train_val_targets = train_val_targets.reshape((train_val_targets.shape[0]*train_val_targets.shape[1]))

        # train test split
        (X_train, y_train), (X_val, y_val) = self.dataset_split_stratified(train_val_data, train_val_targets, k=kfold_k, n_splits=kfold_split)
        train_dataset = PyTorchDataset(X_train, y_train)
        val_dataset = PyTorchDataset(X_val, y_val)

        return train_dataset, val_dataset, 