# -*- coding: utf-8 -*-
"""Task-related component analysis for frequency classification.
Code adapt from https://github.com/nbara/python-meegkit
Authors: Giuseppe Ferraro <giuseppe.ferraro@isae-supaero.fr> Ludovic Darmet <ludovic.darmet@isae-supaero.fr>
"""
import numpy as np
import scipy.linalg as linalg
from scipy.signal import filtfilt, cheb1ord, cheby1
from ..classes.classifier import Classifier


class TRCA(Classifier):
    """Task-Related Component Analysis (TRCA).
    
    Args:
        sampling_rate : float
            Sampling rate.
        filterbank : list[[2-tuple, 2-tuple]]
            Filterbank frequencies. Each list element is itself a list of passband
            `Wp` and stopband `Ws` edges frequencies `[Wp, Ws]`. For example, this
            creates 3 bands, starting at 6, 14, and 22 hz respectively::
                [[(6, 90), (4, 100)],
                 [(14, 90), (10, 100)],
                 [(22, 90), (16, 100)]]
            See :func:`scipy.signal.cheb1ord()` for more information on how to
            specify the `Wp` and `Ws`.
        ensemble : bool
            If True, perform the ensemble TRCA analysis (default=False).
    References:
        [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
        "Enhancing detection of SSVEPs for a high-speed brain speller using
        task-related component analysis", IEEE Trans. Biomed. Eng,
        65(1):104-112, 2018.
        [2] Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2010,
        October). Common spatial pattern revisited by Riemannian geometry. In
        2010 IEEE International Workshop on Multimedia Signal Processing (pp.
        472-476). IEEE.
    """

    def __init__(self, sampling_rate, filterbank=None, ensemble=False):
        self.sampling_rate = sampling_rate
        self.ensemble = ensemble
        self.filterbank = filterbank
        
        if self.filterbank is None:
            self.filterbank = [[(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
              [(14, 90), (10, 100)],
              [(22, 90), (16, 100)],
              [(30, 90), (24, 100)],
              [(38, 90), (32, 100)],
              [(46, 90), (40, 100)],
              [(54, 90), (48, 100)]]
        
        self.n_bands = len(self.filterbank)
        self.coef_ = None
        self.can_train = True

    def fit(self, X, y):
        """Training stage of the TRCA-based SSVEP detection.
        Args:
            X : ndarray, shape (trial, channels, samples)
                Training EEG data.
            Y : ndarray, shape (trial)
                True label corresponding to each trial of the data array.
        """
        X = np.transpose(X, (2,1,0))
        
        n_samples, n_chans, _ = X.shape

        classes = np.unique(y)

        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))

        W = np.zeros((self.n_bands, len(classes), n_chans))

        for class_i in classes:
            # Select data with a specific label
            eeg_tmp = X[..., y == class_i]
            for fb_i in range(self.n_bands):
                # Filter the signal with fb_i
                eeg_tmp = self._bandpass(eeg_tmp, self.sampling_rate,
                                   Wp=self.filterbank[fb_i][0],
                                   Ws=self.filterbank[fb_i][1])
                if (eeg_tmp.ndim == 3):
                    # Compute mean of the signal across trials
                    trains[class_i, fb_i] = np.mean(eeg_tmp, -1)
                else:
                    trains[class_i, fb_i] = eeg_tmp
                # Find the spatial filter for the corresponding filtered signal and label
                w_best = self._trca(eeg_tmp)

                W[fb_i, class_i, :] = w_best  # Store the spatial filter

        self.trains = trains
        self.coef_ = W
        self.classes = classes

    def predict(self, X):
        """Test phase of the TRCA-based SSVEP detection.
        Args:
            X : ndarray, shape (trial, channels, samples)
                Test data.
        Returns:
            pred: np.array, shape (trials)
                The target estimated by the method.
        """
        X = np.transpose(X, (2,1,0))
        
        if self.coef_ is None:
            raise RuntimeError('TRCA is not fitted')

        # Alpha coefficients for the fusion of filterbank analysis
        # fb_coefs = [(x + 1)**(-1.25) + 0.25 for x in range(self.n_bands)]
        fb_coefs = np.arange(1, self.n_bands+1)**(-1.25) + 0.25

        _, _, n_trials = X.shape

        r = np.zeros((self.n_bands, len(self.classes)))
        pred = np.zeros((n_trials), 'int')  # To store predictions

        for trial in range(n_trials):
            test_tmp = X[..., trial]  # pick a trial to be analysed
            for fb_i in range(self.n_bands):

                # filterbank on testdata
                testdata = self._bandpass(test_tmp, self.sampling_rate,
                                    Wp=self.filterbank[fb_i][0],
                                    Ws=self.filterbank[fb_i][1])

                for class_i in self.classes:
                    # Retrieve reference signal for class i
                    # (shape: n_chans, n_samples)
                    traindata = np.squeeze(self.trains[class_i, fb_i])
                    if self.ensemble:
                        # shape = (n_chans, n_classes)
                        w = np.squeeze(self.coef_[fb_i]).T
                    else:
                        # shape = (n_chans)
                        w = np.squeeze(self.coef_[fb_i, class_i])

                    # Compute 2D correlation of spatially filtered test data with ref
                    r_tmp = np.corrcoef((testdata @ w).flatten(),
                                        (traindata @ w).flatten())
                    r[fb_i, class_i] = r_tmp[0, 1]

            rho = np.dot(fb_coefs, r)  # fusion for the filterbank analysis

            tau = np.argmax(rho)  # retrieving index of the max
            pred[trial] = int(tau)

        return pred

    def _trca(self, X):
        """Task-related component analysis.
        This function implements the method described in [1]_.
        Parameters
        ----------
        X : array, shape=(n_samples, n_chans[, n_trials])
            Training data.
        Returns
        -------
        W : array, shape=(n_chans,)
            Weight coefficients for electrodes which can be used as a spatial
            filter.
        References
        ----------
        .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
           "Enhancing detection of SSVEPs for a high-speed brain speller using
           task-related component analysis", IEEE Trans. Biomed. Eng,
           65(1):104-112, 2018.
        """
        n_samples, n_chans, n_trials = X.shape

        # 1. Compute empirical covariance of all data (to be bounded)
        # -------------------------------------------------------------------------
        # Concatenate all the trials to have all the data as a sequence
        UX = np.zeros((n_chans, n_samples * n_trials))
        for trial in range(n_trials):
            UX[:, trial * n_samples:(trial + 1) * n_samples] = X[..., trial].T

        # Mean centering
        UX -= np.mean(UX, 1)[:, None]
        # Covariance
        Q = UX @ UX.T

        # 2. Compute average empirical covariance between all pairs of trials
        # -------------------------------------------------------------------------
        S = np.zeros((n_chans, n_chans))
        for trial_i in range(n_trials - 1):
            x1 = np.squeeze(X[..., trial_i])

            # Mean centering for the selected trial
            x1 -= np.mean(x1, 0)

            # Select a second trial that is different
            for trial_j in range(trial_i + 1, n_trials):
                x2 = np.squeeze(X[..., trial_j])

                # Mean centering for the selected trial
                x2 -= np.mean(x2, 0)

                # Compute empirical covariance between the two selected trials and
                # sum it
                S = S + x1.T @ x2 + x2.T @ x1

        # 3. Compute eigenvalues and vectors
        # -------------------------------------------------------------------------
        lambdas, W = linalg.eig(S, Q, left=True, right=False)

        # Select the eigenvector corresponding to the biggest eigenvalue
        W_best = W[:, np.argmax(lambdas)]

        return W_best

    def _bandpass(self, eeg, sampling_rate, Wp, Ws):
        """Filter bank design for decomposing EEG data into sub-band components.
        Parameters
        ----------
        eeg : np.array, shape=(n_samples, n_chans[, n_trials])
            Training data.
        sampling_rate : int
            Sampling frequency of the data.
        Wp : 2-tuple
            Passband for Chebyshev filter.
        Ws : 2-tuple
            Stopband for Chebyshev filter.
        Returns
        -------
        y: np.array, shape=(n_trials, n_chans, n_samples)
            Sub-band components decomposed by a filter bank.
        See Also
        --------
        scipy.signal.cheb1ord :
            Chebyshev type I filter order selection.
        """
        # Chebyshev type I filter order selection.
        N, Wn = cheb1ord(Wp, Ws, 3, 40, fs=sampling_rate)

        # Chebyshev type I filter design
        B, A = cheby1(N, 0.5, Wn, btype="bandpass", fs=sampling_rate)

        # the arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)'
        # correspond to Matlab filtfilt : https://dsp.stackexchange.com/a/47945
        y = filtfilt(B, A, eeg, axis=0, padtype='odd',
                     padlen=3 * (max(len(B), len(A)) - 1))
        return y


########


# """Task-related component analysis for frequency classification.
# """
# import numpy as np
# from scipy.signal import cheb1ord, cheby1, filtfilt
# from scipy.sparse.linalg import eigs
# from ..classes.classifier import Classifier


# class TRCA(Classifier):
#     r"""
#     Task-related component analysis for frequency classification.

#     Args:
#         sampling_rate: int
#             Sampling frequency
#         num_filterbanks : int, default: 5
#             Number of filterbanks
#         ensemble : boolean, default: True
#             Use ensemble
#     Reference:
#         - M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, “Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis”, IEEE Trans. Biomed. Eng, 65(1): 104-112, 2018.
#         - X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao, “Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface”, J. Neural Eng., 12: 046008, 2015.
#         - https://github.com/okbalefthanded/BCI-Baseline
#         - https://github.com/nbara/python-meegkit
#         - https://github.com/mnakanishi/TRCA-SSVEP
#     """
#     def __init__(self, sampling_rate, num_filterbanks=5, ensemble=True):
#         self.sampling_rate = sampling_rate
#         self.num_filterbanks = num_filterbanks
#         self.ensemble = ensemble
#         self.trains = []
#         self.w = []
#         self.num_targs = 0
#         self.can_train = True

#     def fit(self, X, Y):
#         r"""
#         Construction of the spatial filter.

#         Args:
#             X : ndarray, shape (trial, channels, samples)
#                 3-dim signal data by trial
#             Y : ndarray, shape (trial)
#                 Target labels, targets are int, starts from 0
#         """
#         epochs, channels, samples = X.shape
#         targets_count = len(np.unique(Y))
#         idx = np.argsort(Y)
#         X = X[idx, :, :]

#         if isinstance(epochs/targets_count, int):
#             X = X.reshape((epochs/targets_count, targets_count,
#                            channels, samples), order='F')
#         else:
#             tr = np.floor(epochs/targets_count).astype(int)
#             X = X[0:tr*targets_count, :,
#                   :].reshape((tr, targets_count, channels, samples), order='F')

#         X = X.transpose((1, 2, 3, 0))
        
#         num_targs, num_chans, num_smpls, _ = X.shape
#         self.num_targs = num_targs
#         self.trains = np.zeros((num_targs, self.num_filterbanks, num_chans, num_smpls))
#         self.w = np.zeros((self.num_filterbanks, num_targs, num_chans))

#         for targ_i in range(num_targs):
#             eeg_tmp = X[targ_i, :, :, :]
#             for fb_i in range(self.num_filterbanks):
#                 eeg_tmp = self._filterbank(eeg_tmp, fb_i)
#                 self.trains[targ_i, fb_i, :, :] = np.mean(eeg_tmp, axis=-1)
#                 w_tmp = self._trca(eeg_tmp)
#                 self.w[fb_i, targ_i, :] = w_tmp[:, 0]

#     def predict(self, X):
#         r"""
#         Predict the label for each trial.

#         Args:
#             X : ndarray, shape (trial, channels, samples)
#                 3-dim signal data by trial
#         Returns:
#             results : array
#                 Predicted targets
#         """
        
#         epochs = X.shape[0]
        
#         fb_coefs = np.arange(1, self.num_filterbanks+1)**(-1.25) + 0.25
#         r = np.zeros((self.num_filterbanks, self.num_targs))
#         results = []
#         for trial_i in range(epochs):
#             test_tmp = X[trial_i, :, :]
#             for fb_i in range(self.num_filterbanks):
#                 testdata = self._filterbank(test_tmp, fb_i)
#                 for class_i in range(self.num_targs):
#                     traindata = self.trains[class_i, fb_i, :, :]
#                     if self.ensemble:
#                         w = self.w[fb_i, :, :].T
#                         # Follows corrcoef MATLAB function implementation
#                         r_tmp = np.corrcoef(np.matmul(testdata.T, w).flatten(), np.matmul(traindata.T, w).flatten())
#                     else:
#                         w = self.w[fb_i, class_i, :]
#                         r_tmp = np.corrcoef(np.matmul(testdata.T, w), np.matmul(traindata.T, w))
#                     r[fb_i, class_i] = r_tmp[0, 1]

#             rho = np.matmul(fb_coefs, r)
#             tau = np.argmax(rho)
#             results.append(tau)

#         results = np.array(results)
#         return results

#     def _trca(self, eeg):
#         num_chans, num_smpls, num_trials = eeg.shape

#         S = np.zeros((num_chans, num_chans))

#         for trial_i in range(num_trials-1):
#             x1 = eeg[:, :, trial_i]
#             x1 = x1 - np.mean(x1, axis=1)[:, None]
#             for trial_j in range(trial_i+1, num_trials):
#                 x2 = eeg[:, :, trial_j]
#                 x2 = x2 - np.mean(x2, axis=1)[:, None]
#                 S = S + np.matmul(x1, x2.T) + np.matmul(x2, x1.T)

#         UX = eeg.reshape((num_chans, num_smpls*num_trials))
#         UX = UX - np.mean(UX, axis=1)[:, None]
#         Q = np.matmul(UX, UX.T)
#         _, W = eigs(S, M=Q)
#         #_, W = np.linalg.eig(np.dot(np.linalg.inv(Q), S))
#         return np.real(W)

#     def _filterbank(self, eeg, idx_fbi):
#         r"""
#         We use the filterbank specification described in X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao, “Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface”, J. Neural Eng., 12: 046008, 2015.
#         """        
            
#         if eeg.ndim == 2:
#             num_chans = eeg.shape[0]
#             num_trials = 1
#         else:
#             num_chans, _, num_trials = eeg.shape
#         fs = self.sampling_rate / 2
#         passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
#         stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
#         Wp = [passband[idx_fbi]/fs, 90/fs]
#         Ws = [stopband[idx_fbi]/fs, 100/fs]
#         [N, Wn] = cheb1ord(Wp, Ws, 3, 40)
#         [B, A] = cheby1(N, 0.5, Wn, 'bp')
#         yy = np.zeros_like(eeg)
#         if num_trials == 1:
#             yy = filtfilt(B, A, eeg, axis=1)
#         else:
#             for trial_i in range(num_trials):
#                 for ch_i in range(num_chans):
#                     yy[ch_i, :, trial_i] = filtfilt(B, A, eeg[ch_i, :, trial_i], padtype='odd', padlen=3*(max(len(B),len(A))-1))
#         return yy


# if __name__ == "__main__":
#     from splearn.cross_decomposition.trca import TRCA
#     from splearn.data.sample_ssvep import SampleSSVEPData
#     from splearn.cross_validate.leave_one_out import leave_one_block_evaluation
    
#     data = SampleSSVEPData()
#     eeg = data.get_data()
#     labels = data.get_targets()
#     print("eeg.shape:", eeg.shape)
#     print("labels.shape:", labels.shape)

#     trca_classifier = TRCA(sampling_rate=data.sampling_rate)
#     test_accuracies = leave_one_block_evaluation(trca_classifier, eeg, labels)
