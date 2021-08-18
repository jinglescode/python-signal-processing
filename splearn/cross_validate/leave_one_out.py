import numpy as np
from sklearn.metrics import accuracy_score


def leave_one_block_evaluation(classifier, X, Y):
    r"""
    Estimate classification performance with a Leave-One-Block-Out cross-validation approach.
    Iteratively select a block for testing and use all other blocks for training.

    Args:
        X : ndarray, shape (blocks, targets, channels, samples)
            4-dim signal data
        Y : ndarray, shape (blocks, targets)
            Targets are int, starts from 0
    Returns:
        test_accuracies : list
            Test accuracies by block
    Usage:
        >>> from splearn.cross_decomposition.trca import TRCA
        >>> from splearn.data.sample_ssvep import SampleSSVEPData
        >>> 
        >>> data = SampleSSVEPData()
        >>> eeg = data.get_data()
        >>> labels = data.get_targets()
        >>> print("eeg.shape:", eeg.shape)
        >>> print("labels.shape:", labels.shape)
        >>> 
        >>> trca_classifier = TRCA(sampling_rate=data.sampling_rate)
        >>> test_accuracies = trca_classifier.leave_one_block_evaluation(eeg, labels)
    """

    test_accuracies = []
    blocks, targets, channels, samples = X.shape

    for block_i in range(blocks):
        train_acc, test_acc = block_evaluation(classifier, X, Y, block_i)
        test_accuracies.append(test_acc)

    print(f'Mean test accuracy: {np.array(test_accuracies).mean().round(3)*100}%')
    return test_accuracies

def block_evaluation(classifier, X, Y, block_i):
    r"""
    Select a block for testing, use all other blocks for training.

    Args:
        X : ndarray, shape (blocks, targets, channels, samples)
            4-dim signal data
        Y : ndarray, shape (blocks, targets)
            Targets are int, starts from 0
        block_i: int
            Index of the selected block for testing
    Returns:
        train_acc : float
            Train accuracy
        test_acc : float
            Test accuracy of the selected block
    """

    blocks, targets, channels, samples = X.shape

    x_train = np.delete(X, block_i, axis=0)
    x_train = x_train.reshape((blocks-1*targets, channels, samples))
    y_train = np.delete(Y, block_i, axis=0)
    y_train = y_train.reshape((blocks-1*targets))
    classifier.fit(x_train, y_train)
    p1 = classifier.predict(x_train)
    train_acc = accuracy_score(y_train, p1)

    x_test = X[block_i,:,:,:]
    y_test = Y[block_i]
    p2 = classifier.predict(x_test)
    test_acc = accuracy_score(y_test, p2)

    print(f'Block: {block_i+1} | Train acc: {train_acc*100:.2f}% | Test acc: {test_acc*100:.2f}%')

    return train_acc, test_acc
