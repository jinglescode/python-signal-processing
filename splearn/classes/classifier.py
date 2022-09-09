import numpy as np
from splearn.cross_validate.leave_one_out import block_evaluation

class Classifier():

    def __init__(self):
        pass

    def predict(self, X):
        return None
    
    def fit(self, X, Y):
        pass

    def leave_one_block_evaluation(classifier, X, Y, block_seq_labels=None):
        test_results_acc = []
        blocks, targets, channels, samples = X.shape
                
        for block_i in range(blocks):
            test_acc = block_evaluation(classifier, X, Y, block_i)
            test_results_acc.append(test_acc)
                        
        mean_acc = np.array(test_results_acc).mean().round(3)*100

        print(f'Mean test accuracy: {mean_acc}%')
        return (mean_acc, test_results_acc)
