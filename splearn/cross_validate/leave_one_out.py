import numpy as np
from sklearn.metrics import accuracy_score


def leave_one_block_evaluation(classifier, X, Y, block_seq_labels=None):
    r"""
    Estimate classification performance with a Leave-One-Block-Out cross-validation approach.
    Iteratively select a block for testing and use all other blocks for training.

    Args:
        X : ndarray, shape (blocks, targets, channels, samples)
            4-dim signal data
        Y : ndarray, shape (blocks, targets)
            Targets are int, starts from 0
        block_seq_labels : list
            A list of labels for each block
    Returns:
        test_accuracies : list
            Test accuracies by block
    Usage:
        >>> from splearn.cross_decomposition.trca import TRCA
        >>> from splearn.data.sample_ssvep import SampleSSVEPData
        >>> from splearn.cross_validate.leave_one_out import leave_one_block_evaluation
        >>> 
        >>> data = SampleSSVEPData()
        >>> eeg = data.get_data()
        >>> labels = data.get_targets()
        >>> print("eeg.shape:", eeg.shape)
        >>> print("labels.shape:", labels.shape)
        >>> 
        >>> trca_classifier = TRCA(sampling_rate=data.sampling_rate)
        >>> test_accuracies = leave_one_block_evaluation(trca_classifier, eeg, labels)
    """

    test_accuracies = []
    blocks, targets, channels, samples = X.shape

    for block_i in range(blocks):
        test_acc = block_evaluation(classifier, X, Y, block_i, block_seq_labels[block_i] if block_seq_labels is not None else None)
        test_accuracies.append(test_acc)

    print(f'Mean test accuracy: {np.array(test_accuracies).mean().round(3)*100}%')
    return test_accuracies

def block_evaluation(classifier, X, Y, block_i, block_label=None):
    r"""
    Select a block for testing, use all other blocks for training.

    Args:
        X : ndarray, shape (blocks, targets, channels, samples)
            4-dim signal data
        Y : ndarray, shape (blocks, targets)
            Targets are int, starts from 0
        block_i: int
            Index of the selected block for testing
        block_label : str or int
            Labels for this block, for printing
    Returns:
        train_acc : float
            Train accuracy
        test_acc : float
            Test accuracy of the selected block
    """

    blocks, targets, channels, samples = X.shape
    
    train_acc = 0
    if classifier.can_train:
        x_train = np.delete(X, block_i, axis=0)
        x_train = x_train.reshape((blocks-1*targets, channels, samples))
        y_train = np.delete(Y, block_i, axis=0)
        y_train = y_train.reshape((blocks-1*targets))
        classifier.fit(x_train, y_train)
#         p1 = classifier.predict(x_train)
#         train_acc = accuracy_score(y_train, p1)

    x_test = X[block_i,:,:,:]
    y_test = Y[block_i]
    p2 = classifier.predict(x_test)
    test_acc = accuracy_score(y_test, p2)
    
    if block_label is None:
        block_label = 'Block:' + str(block_i+1)
        
#     if classifier.can_train:
#         print(f'{block_label} | Train acc: {train_acc*100:.2f}% | Test acc: {test_acc*100:.2f}%')
#     else:
#         print(f'{block_label} | Test acc: {test_acc*100:.2f}%')
        
    print(f'{block_label} | Test acc: {test_acc*100:.2f}%')

    return test_acc
