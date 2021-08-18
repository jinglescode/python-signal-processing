# -*- coding: utf-8 -*-
"""Task-related component analysis for frequency classification.
"""
import numpy as np
from scipy.signal import cheb1ord, cheby1, filtfilt
from scipy.sparse.linalg import eigs
from sklearn.metrics import accuracy_score
from ..classes.classifier import Classifier


class TRCA(Classifier):
    r"""
    Task-related component analysis for frequency classification.

    Args:
        sampling_rate: int
            Sampling frequency
        num_filterbanks : int, default: 5
            Number of filterbanks
        ensemble : boolean, default: True
            Use ensemble
    Reference:
        - M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, “Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis”, IEEE Trans. Biomed. Eng, 65(1): 104-112, 2018.
        - X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao, “Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface”, J. Neural Eng., 12: 046008, 2015.
        - https://github.com/okbalefthanded/BCI-Baseline
        - https://github.com/nbara/python-meegkit
        - https://github.com/mnakanishi/TRCA-SSVEP
    """
    def __init__(self, sampling_rate, num_filterbanks=5, ensemble=True):
        self.sampling_rate = sampling_rate
        self.num_filterbanks = num_filterbanks
        self.ensemble = ensemble
        self.trains = []
        self.w = []
        self.num_targs = 0

    def fit(self, X, Y):
        r"""
        Construction of the spatial filter.

        Args:
            X : ndarray, shape (trial, channels, samples)
                3-dim signal data by trial
            Y : ndarray, shape (trial)
                Target labels, targets are int, starts from 0
        """
        epochs, channels, samples = X.shape
        targets_count = len(np.unique(Y))
        idx = np.argsort(Y)
        X = X[idx, :, :]

        if isinstance(epochs/targets_count, int):
            X = X.reshape((epochs/targets_count, targets_count,
                           channels, samples), order='F')
        else:
            tr = np.floor(epochs/targets_count).astype(int)
            X = X[0:tr*targets_count, :,
                  :].reshape((tr, targets_count, channels, samples), order='F')

        X = X.transpose((1, 2, 3, 0))
        
        num_targs, num_chans, num_smpls, _ = X.shape
        self.num_targs = num_targs
        self.trains = np.zeros((num_targs, self.num_filterbanks, num_chans, num_smpls))
        self.w = np.zeros((self.num_filterbanks, num_targs, num_chans))

        for targ_i in range(num_targs):
            eeg_tmp = X[targ_i, :, :, :]
            for fb_i in range(self.num_filterbanks):
                eeg_tmp = self._filterbank(eeg_tmp, fb_i)
                self.trains[targ_i, fb_i, :, :] = np.mean(eeg_tmp, axis=-1)
                w_tmp = self._trca(eeg_tmp)
                self.w[fb_i, targ_i, :] = w_tmp[:, 0]

    def predict(self, X):
        r"""
        Predict the label for each trial.

        Args:
            X : ndarray, shape (trial, channels, samples)
                3-dim signal data by trial
        Returns:
            results : array
                Predicted targets
        """
        
        epochs = X.shape[0]
        
        fb_coefs = np.arange(1, 6)**(-1.25) + 0.25
        # r = np.zeros((self.num_filterbanks, epochs))
        r = np.zeros((self.num_filterbanks, self.num_targs))
        results = []
        for targ_i in range(epochs):
            test_tmp = X[targ_i, :, :]
            for fb_i in range(self.num_filterbanks):
                testdata = self._filterbank(test_tmp, fb_i)
                for class_i in range(self.num_targs):
                    traindata = self.trains[class_i, fb_i, :, :]
                    if self.ensemble:
                        w = self.w[fb_i, :, :].T
                        # Follows corrcoef MATLAB function implementation
                        r_tmp = np.corrcoef(np.matmul(testdata.T, w).flatten(), np.matmul(traindata.T, w).flatten())
                    else:
                        w = self.w[fb_i, class_i, :]
                        r_tmp = np.corrcoef(np.matmul(testdata.T, w), np.matmul(traindata.T, w))
                    r[fb_i, class_i] = r_tmp[0, 1]

            rho = np.matmul(fb_coefs, r)
            tau = np.argmax(rho)
            results.append(tau)

        results = np.array(results)
        return results

    def _trca(self, eeg):
        num_chans, num_smpls, num_trials = eeg.shape

        S = np.zeros((num_chans, num_chans))

        for trial_i in range(num_trials-1):
            x1 = eeg[:, :, trial_i]
            x1 = x1 - np.mean(x1, axis=1)[:, None]
            for trial_j in range(trial_i+1, num_trials):
                x2 = eeg[:, :, trial_j]
                x2 = x2 - np.mean(x2, axis=1)[:, None]
                S = S + np.matmul(x1, x2.T) + np.matmul(x2, x1.T)

        UX = eeg.reshape((num_chans, num_smpls*num_trials))
        UX = UX - np.mean(UX, axis=1)[:, None]
        Q = np.matmul(UX, UX.T)
        _, W = eigs(S, M=Q)
        #_, W = np.linalg.eig(np.dot(np.linalg.inv(Q), S))
        return np.real(W)

    def _filterbank(self, eeg, idx_fbi):
        r"""
        We use the filterbank specification described in X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao, “Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface”, J. Neural Eng., 12: 046008, 2015.
        """        
            
        if eeg.ndim == 2:
            num_chans = eeg.shape[0]
            num_trials = 1
        else:
            num_chans, _, num_trials = eeg.shape
        fs = self.sampling_rate / 2
        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
        Wp = [passband[idx_fbi]/fs, 90/fs]
        Ws = [stopband[idx_fbi]/fs, 100/fs]
        [N, Wn] = cheb1ord(Wp, Ws, 3, 40)
        [B, A] = cheby1(N, 0.5, Wn, 'bp')
        yy = np.zeros_like(eeg)
        if num_trials == 1:
            yy = filtfilt(B, A, eeg, axis=1)
        else:
            for trial_i in range(num_trials):
                for ch_i in range(num_chans):
                    yy[ch_i, :, trial_i] = filtfilt(B, A, eeg[ch_i, :, trial_i], padtype='odd', padlen=3*(max(len(B),len(A))-1))
        return yy

    def leave_one_block_evaluation(self, X, Y):
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
            train_acc, test_acc = self.block_evaluation(X, Y, block_i)
            test_accuracies.append(test_acc)
            
        print(f'Mean test accuracy: {np.array(test_accuracies).mean().round(3)*100}%')
        return test_accuracies
    
    def block_evaluation(self, X, Y, block_i):
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
        self.fit(x_train, y_train)
        p1 = self.predict(x_train)
        train_acc = accuracy_score(y_train, p1)

        x_test = X[block_i,:,:,:]
        y_test = Y[block_i]
        p2 = self.predict(x_test)
        test_acc = accuracy_score(y_test, p2)
        
        print(f'Block: {block_i+1} | Train acc: {train_acc*100:.2f}% | Test acc: {test_acc*100:.2f}%')
        
        return train_acc, test_acc


if __name__ == "__main__":
    from splearn.cross_decomposition.trca import TRCA
    from splearn.data.sample_ssvep import SampleSSVEPData
    
    data = SampleSSVEPData()
    eeg = data.get_data()
    labels = data.get_targets()
    print("eeg.shape:", eeg.shape)
    print("labels.shape:", labels.shape)

    trca_classifier = TRCA(sampling_rate=data.sampling_rate)
    test_accuracies = trca_classifier.leave_one_block_evaluation(eeg, labels)
