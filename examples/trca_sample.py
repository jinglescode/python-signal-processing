# this code is for reproducing the results in: https://github.com/mnakanishi/TRCA-SSVEP

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


from splearn.data import SampleSSVEPData
from splearn.utils import Logger, Config
from splearn.cross_decomposition.trca import TRCA
from splearn.cross_validate.leave_one_out import block_evaluation


main_logger = Logger(filename_postfix="trca sample")
main_logger.write_to_log("Config")

# load data

data = SampleSSVEPData()
print(data.data.shape)

# method 1
eeg = data.get_data()
labels = data.get_targets()
trca_classifier = TRCA(sampling_rate=data.sampling_rate)
test_accuracies = trca_classifier.leave_one_block_evaluation(eeg, labels)

# method 2

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


trca_classifier = TRCA(sampling_rate=data.sampling_rate)
leave_one_block_evaluation(classifier=trca_classifier, X=data.data, Y=data.targets)


# expected output:
# Block: 1 | Train acc: 100.00% | Test acc: 97.50%
# Block: 2 | Train acc: 100.00% | Test acc: 100.00%
# Block: 3 | Train acc: 100.00% | Test acc: 100.00%
# Block: 4 | Train acc: 100.00% | Test acc: 100.00%
# Block: 5 | Train acc: 100.00% | Test acc: 97.50%
# Block: 6 | Train acc: 100.00% | Test acc: 100.00%
