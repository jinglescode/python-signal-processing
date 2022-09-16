# for running locally
import os
cwd = os.getcwd()
import sys
# path = os.path.join(cwd, "..\\..\\")
path = cwd
sys.path.append(path)

# imports
import numpy as np
import logging
logging.getLogger('lightning').setLevel(0)
import warnings
warnings.filterwarnings('ignore')


from splearn.data import MultipleSubjects, JFPM
from splearn.utils import Logger, Config
from splearn.filter.butterworth import butter_bandpass_filter
from splearn.filter.notch import notch_filter
from splearn.cross_decomposition.trca import TRCA
from splearn.cross_validate.leave_one_out import block_evaluation

config = {
    "experiment_name": "trcaEnsemble_jfpm",
    "data": {
        "load_subject_ids": np.arange(1,11),
        "root": "../data/jfpm",
        "duration": 1,
    },
    "trca": {
        "ensemble": True
    },
    "seed": 1234
}
main_logger = Logger(filename_postfix=config["experiment_name"])
main_logger.write_to_log("Config")
main_logger.write_to_log(config)
config = Config(config)

# define custom preprocessing steps
def func_preprocessing(data):
    data_x = data.data
    data_x = notch_filter(data_x, sampling_rate=data.sampling_rate, notch_freq=50.0)
    data_x = butter_bandpass_filter(data_x, lowcut=7, highcut=90, sampling_rate=data.sampling_rate, order=6)
    start_t = 35
    end_t = start_t + (config.data.duration * data.sampling_rate)
    data_x = data_x[:,:,:,start_t:end_t]
    data.set_data(data_x)

# load data
data = MultipleSubjects(
    dataset=JFPM, 
    root=os.path.join(path,config.data.root), 
    subject_ids=config.data.load_subject_ids, 
    func_preprocessing=func_preprocessing,
    verbose=True, 
)

num_channel = data.data.shape[2]
num_classes = data.stimulus_frequencies.shape[0]
signal_length = data.data.shape[3]


def leave_one_block_evaluation(classifier, X, Y, block_seq_labels=None):
    test_results_acc = []
    blocks, targets, channels, samples = X.shape
    
    main_logger.write_to_log("Begin", break_line=True)
    
    for block_i in range(blocks):
        test_acc = block_evaluation(classifier, X, Y, block_i)
        test_results_acc.append(test_acc)
        
        this_result = {
            "test_subject_id": block_i+1,
            "acc": test_acc,
        }
        
        main_logger.write_to_log(this_result)
        
    mean_acc = np.array(test_results_acc).mean().round(3)*100

    print(f'Mean test accuracy: {mean_acc}%')
    
    main_logger.write_to_log("Mean acc: "+str(mean_acc), break_line=True)


trca_classifier = TRCA(sampling_rate=data.sampling_rate, ensemble=config.trca.ensemble)
print("data:", data.data.shape)
print("targets:", data.targets.shape)
leave_one_block_evaluation(classifier=trca_classifier, X=data.data, Y=data.targets)
