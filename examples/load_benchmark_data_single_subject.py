# for running locally (may remove this if your path is right)
import os
cwd = os.getcwd()
import sys
path = cwd
sys.path.append(path)

# imports
from splearn.data import Benchmark

# config
load_subject_id = 1
path_to_dataset = "../data/hsssvep"

# load data
subject_dataset = Benchmark(root=path_to_dataset, subject_id=load_subject_id)

# display
print("About the data:")
print("sample rate:", subject_dataset.sample_rate)
print("data shape:", subject_dataset.data.shape)
print("targets shape:", subject_dataset.targets.shape)
print("stimulus frequencies:", subject_dataset.stimulus_frequencies)
print("targets frequencies:", subject_dataset.targets_frequencies)

# expected output:
# ```
# About the data:
# sample rate: 1000
# data shape: (240, 64, 1500)
# targets shape: (240,)
# stimulus frequencies: [ 8.   9.  10.  11.  12.  13.  14.  15.   8.2  9.2 10.2 11.2 12.2 13.2
#  14.2 15.2  8.4  9.4 10.4 11.4 12.4 13.4 14.4 15.4  8.6  9.6 10.6 11.6
#  12.6 13.6 14.6 15.6  8.8  9.8 10.8 11.8 12.8 13.8 14.8 15.8]
# targets frequencies: [ 8.   8.   8.   8.   8.   8.   9.   9.   9.   9.   9.   9.  10.  10.
#  10.  10.  10.  10.  11.  11.  11.  11.  11.  11.  12.  12.  12.  12.
#  12.  12.  13.  13.  13.  13.  13.  13.  14.  14.  14.  14.  14.  14.
#  15.  15.  15.  15.  15.  15.   8.2  8.2  8.2  8.2  8.2  8.2  9.2  9.2
#   9.2  9.2  9.2  9.2 10.2 10.2 10.2 10.2 10.2 10.2 11.2 11.2 11.2 11.2
#  11.2 11.2 12.2 12.2 12.2 12.2 12.2 12.2 13.2 13.2 13.2 13.2 13.2 13.2
#  14.2 14.2 14.2 14.2 14.2 14.2 15.2 15.2 15.2 15.2 15.2 15.2  8.4  8.4
#   8.4  8.4  8.4  8.4  9.4  9.4  9.4  9.4  9.4  9.4 10.4 10.4 10.4 10.4
#  10.4 10.4 11.4 11.4 11.4 11.4 11.4 11.4 12.4 12.4 12.4 12.4 12.4 12.4
#  13.4 13.4 13.4 13.4 13.4 13.4 14.4 14.4 14.4 14.4 14.4 14.4 15.4 15.4
#  15.4 15.4 15.4 15.4  8.6  8.6  8.6  8.6  8.6  8.6  9.6  9.6  9.6  9.6
#   9.6  9.6 10.6 10.6 10.6 10.6 10.6 10.6 11.6 11.6 11.6 11.6 11.6 11.6
#  12.6 12.6 12.6 12.6 12.6 12.6 13.6 13.6 13.6 13.6 13.6 13.6 14.6 14.6
#  14.6 14.6 14.6 14.6 15.6 15.6 15.6 15.6 15.6 15.6  8.8  8.8  8.8  8.8
#   8.8  8.8  9.8  9.8  9.8  9.8  9.8  9.8 10.8 10.8 10.8 10.8 10.8 10.8
#  11.8 11.8 11.8 11.8 11.8 11.8 12.8 12.8 12.8 12.8 12.8 12.8 13.8 13.8
#  13.8 13.8 13.8 13.8 14.8 14.8 14.8 14.8 14.8 14.8 15.8 15.8 15.8 15.8
#  15.8 15.8]
# ```