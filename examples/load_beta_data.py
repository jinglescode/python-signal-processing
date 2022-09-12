# for running locally (may remove this if your path is right)
import os
cwd = os.getcwd()
import sys
path = cwd
sys.path.append(path)

# imports
from splearn.data import Beta

# config
load_subject_id = 1
path_to_dataset = "../data/beta"

# load data
subject_dataset = Beta(root=path_to_dataset, subject_id=load_subject_id, verbose=True)

# # display
print("About the data:")
print("sample rate:", subject_dataset.sample_rate)
print("data shape:", subject_dataset.data.shape)
print("targets shape:", subject_dataset.targets.shape)
print("stimulus frequencies:", subject_dataset.stimulus_frequencies)
print("targets frequencies:", subject_dataset.targets_frequencies)
print("targets:", subject_dataset.targets)
print("channel_names", subject_dataset.channel_names)
