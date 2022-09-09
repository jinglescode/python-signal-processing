# for running locally (may remove this if your path is right)
import os
cwd = os.getcwd()
import sys
path = cwd
sys.path.append(path)

# imports
import numpy as np
from torch.utils.data import DataLoader

from splearn.data import MultipleSubjects, Benchmark
from splearn.utils import Config
from splearn.filter.butterworth import butter_bandpass_filter
from splearn.filter.notch import notch_filter
from splearn.filter.channels import pick_channels

# config

config = {
    "data": {
        "load_subject_ids": np.arange(1,4), # get subject #1, #2 and #3
        "root": "../data/hsssvep",
        "selected_channels": ["PZ", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"],
    },
    "training": {
        "batchsize": 256,
    },
}
config = Config(config)

# define custom preprocessing steps
def func_preprocessing(data):
    data_x = data.data
    data_x = pick_channels(data_x, channel_names=data.channel_names, selected_channels=config.data.selected_channels)
    data_x = notch_filter(data_x, sampling_rate=data.sampling_rate, notch_freq=50.0)
    data_x = butter_bandpass_filter(data_x, lowcut=7, highcut=90, sampling_rate=data.sampling_rate, order=6)
    start_t = 160
    end_t = start_t + 250
    data_x = data_x[:,:,:,start_t:end_t]
    data.set_data(data_x)

# load data
data = MultipleSubjects(
    dataset=Benchmark, 
    root=os.path.join(path,config.data.root), 
    subject_ids=config.data.load_subject_ids, 
    func_preprocessing=func_preprocessing,
    verbose=True, 
)

# display data info
num_channel = data.data.shape[2]
signal_length = data.data.shape[3]
print("Final data shape:", data.data.shape)
print("num of subjects", data.data.shape[0])
print("num channels: ", num_channel)
print("signal length: ", signal_length)

def prepare_dataloaders(test_subject_id, kfold_split=3, kfold_k=0):

    train_dataset, val_dataset, test_dataset = data.get_train_val_test_dataset(test_subject_id=test_subject_id, kfold_split=kfold_split, kfold_k=kfold_k)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batchsize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batchsize, shuffle=False)

    print("train_loader shape", train_loader.dataset.data.shape)
    print("val_loader shape", val_loader.dataset.data.shape)
    print("test_loader shape", test_loader.dataset.data.shape)

test_subject_id = 1
prepare_dataloaders(test_subject_id)
