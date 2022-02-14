import os
cwd = os.getcwd()
import sys
path = os.path.join(cwd, "..\\..\\")
sys.path.append(path)

import numpy as np

from splearn.data import MultipleSubjects, HSSSVEP
from splearn.filter.butterworth import butter_bandpass_filter
from splearn.filter.notch import notch_filter
from splearn.filter.channels import pick_channels
from splearn.utils import Logger, Config
from splearn.cross_validate.leave_one_out import block_evaluation
from splearn.cross_decomposition.cca import * # https://github.com/jinglescode/python-signal-processing/blob/main/splearn/cross_decomposition/
from splearn.cross_decomposition.reference_frequencies import * # https://github.com/jinglescode/python-signal-processing/blob/main/splearn/cross_decomposition/

####

config = {
    "run_name": "cca_hsssvep_run2",
    "data": {
        "load_subject_ids": np.arange(1,36),
        # "selected_channels": ["PO8", "PZ", "PO7", "PO4", "POz", "PO3", "O2", "Oz", "O1"], # AA paper
        "selected_channels": ["PZ", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"], # hsssvep paper        
    },
    "seed": 1234
}

main_logger = Logger(filename_postfix=config["run_name"])
main_logger.write_to_log("Config")
main_logger.write_to_log(config)

config = Config(config)

####

"""
def func_preprocessing(data):
    data_x = data.data
    # selected_channels = ['P7','P3','PZ','P4','P8','O1','Oz','O2','P1','P2','POz','PO3','PO4']
    selected_channels = config.data.selected_channels
    data_x = pick_channels(data_x, channel_names=data.channel_names, selected_channels=selected_channels)
    # data_x = notch_filter(data_x, sampling_rate=data.sampling_rate, notch_freq=50.0)
    data_x = butter_bandpass_filter(data_x, lowcut=4, highcut=75, sampling_rate=data.sampling_rate, order=6)
    start_t = 125
    end_t = 125 + 250
    data_x = data_x[:,:,:,start_t:end_t]
    data.set_data(data_x)
"""

def func_preprocessing(data):
    data_x = data.data
    data_x = pick_channels(data_x, channel_names=data.channel_names, selected_channels=config.data.selected_channels)
    # data_x = notch_filter(data_x, sampling_rate=data.sampling_rate, notch_freq=50.0)
    data_x = butter_bandpass_filter(data_x, lowcut=7, highcut=90, sampling_rate=data.sampling_rate, order=6)
    start_t = 160
    end_t = start_t + 250
    data_x = data_x[:,:,:,start_t:end_t]
    data.set_data(data_x)

data = MultipleSubjects(
    dataset=HSSSVEP, 
    root=os.path.join(path, "../data/hsssvep"), 
    subject_ids=config.data.load_subject_ids, 
    func_preprocessing=func_preprocessing,
    verbose=True, 
)

print("Final data shape:", data.data.shape)

num_channel = data.data.shape[2]
num_classes = 40
signal_length = data.data.shape[3]

sampling_rate = data.sampling_rate
signal_duration_seconds = 1
target_frequencies = data.stimulus_frequencies
reference_frequencies = generate_reference_signals(target_frequencies, size=signal_duration_seconds*sampling_rate, sampling_rate=sampling_rate, num_harmonics=5)
print("reference_frequencies.shape", reference_frequencies.shape)

####


def test_cca_subject(test_subject_id):
    data_subject, labels = data.get_subject(test_subject_id)
    predicted_class, accuracy, predicted_probabilities, _, _ = perform_cca(data_subject, reference_frequencies, labels=labels)
    return accuracy

test_results_acc = []

for test_subject_id in config.data.load_subject_ids:
    test_acc = test_cca_subject(test_subject_id)    
    test_results_acc.append(test_acc)
    
    this_result = {
        "test_subject_id": test_subject_id,
        "acc": test_acc,
    }

    main_logger.write_to_log(this_result)

mean_acc = np.array(test_results_acc).mean().round(3)*100

print(f'Mean test accuracy: {mean_acc}%')

main_logger.write_to_log("Mean acc: "+str(mean_acc), break_line=True)
