import os
cwd = os.getcwd()
import sys
path = os.path.join(cwd, "..\\..\\")
sys.path.append(path)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.functional import elu

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import logging
logging.getLogger('lightning').setLevel(0)

import warnings
warnings.filterwarnings('ignore')

import pytorch_lightning
pytorch_lightning.utilities.distributed.log.setLevel(logging.ERROR)

from splearn.data import MultipleSubjects, PyTorchDataset, PyTorchDataset2Views, HSSSVEP
from splearn.filter.butterworth import butter_bandpass_filter
from splearn.filter.notch import notch_filter
from splearn.filter.channels import pick_channels
from splearn.nn.models import CompactEEGNet
from splearn.utils import Logger, Config
from splearn.nn.base import LightningModelClassifier

####

config = {
    "run_name": "deep4net_normal",
    "data": {
        "load_subject_ids": np.arange(1,3),
        # "selected_channels": ["PO8", "PZ", "PO7", "PO4", "POz", "PO3", "O2", "Oz", "O1"], # AA paper
        "selected_channels": ["PZ", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"], # hsssvep paper
    },
    "training": {
        "num_epochs": 10,
        "num_warmup_epochs": 50,
        "learning_rate": 0.03,
        "gpus": [0],
        "batchsize": 256,
    },
    "model": {
        "optimizer": "adamw",
        "scheduler": "cosine_with_warmup",
    },
    "testing": {
        "test_subject_ids": np.arange(1,2),
        "kfolds": np.arange(0,3),
    },
    "seed": 1234
}

main_logger = Logger(filename_postfix=config["run_name"])
main_logger.write_to_log("Config")
main_logger.write_to_log(config)

config = Config(config)

seed_everything(config.seed)

####

# def func_preprocessing(data):
#     data_x = data.data
#     # selected_channels = ['P7','P3','PZ','P4','P8','O1','Oz','O2','P1','P2','POz','PO3','PO4']
#     selected_channels = config.data.selected_channels
#     data_x = pick_channels(data_x, channel_names=data.channel_names, selected_channels=selected_channels)
#     # data_x = notch_filter(data_x, sampling_rate=data.sampling_rate, notch_freq=50.0)
#     data_x = butter_bandpass_filter(data_x, lowcut=4, highcut=75, sampling_rate=data.sampling_rate, order=6)
#     start_t = 125
#     end_t = 125 + 250
#     data_x = data_x[:,:,:,start_t:end_t]
#     data.set_data(data_x)

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

####


def np_to_th(
    X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
    """
    Convenience function to transform numpy array to `torch.Tensor`.
    Converts `X` to ndarray using asarray if necessary.
    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor
    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor

def identity(x):
    return x

def transpose_time_to_spat(x):
    """Swap time and spatial dimensions.
    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 3, 2, 1)

def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.
    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class Expression(nn.Module):
    """Compute given expression on forward pass.
    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )


class AvgPool2dWithConv(nn.Module):
    """
    Compute average pooling using a convolution, to have the dilation parameter.
    Parameters
    ----------
    kernel_size: (int,int)
        Size of the pooling region.
    stride: (int,int)
        Stride of the pooling operation.
    dilation: int or (int,int)
        Dilation applied to the pooling filter.
    padding: int or (int,int)
        Padding applied before the pooling operation.
    """

    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        # don't name them "weights" to
        # make sure these are not accidentally used by some procedure
        # that initializes parameters or something
        self._pool_weights = None

    def forward(self, x):
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (
            in_channels,
            1,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape)) or
            (self._pool_weights.is_cuda != x.is_cuda) or
            (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_th(
                np.ones(weight_shape, dtype=np.float32) / float(n_pool)
            )
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self._pool_weights = weights

        pooled = F.conv2d(
            x,
            self._pool_weights,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=in_channels,
        )
        return pooled
    
class Ensure4d(nn.Module):
    def forward(self, x):
        while(len(x.shape) < 4):
            x = x.unsqueeze(-1)
        return 
        
        

class Deep4Net(nn.Sequential):
    """Deep ConvNet model from Schirrmeister et al 2017.
    Model described in [Schirrmeister2017]_.
    Parameters
    ----------
    in_chans : int
        XXX
    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples,
        final_conv_length,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=1,
        pool_time_stride=1,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        first_nonlin=elu,
        first_pool_mode="max",
        first_pool_nonlin=identity,
        later_nonlin=elu,
        later_pool_mode="max",
        later_pool_nonlin=identity,
        drop_prob=0.5,
        double_time_convs=False,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_nonlin = first_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_nonlin = later_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        self.add_module("ensuredims", Ensure4d())
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Expression(transpose_time_to_spat))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )
        self.add_module("conv_nonlin", Expression(self.first_nonlin))
        self.add_module(
            "pool",
            first_pool_class(
                kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        self.add_module("pool_nonlin", Expression(self.first_pool_nonlin))

        def add_conv_pool_block(
            model, n_filters_before, n_filters, filter_length, block_nr
        ):
            suffix = "_{:d}".format(block_nr)
            self.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            self.add_module(
                "conv" + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            if self.batch_norm:
                self.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm2d(
                        n_filters,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            self.add_module("nonlin" + suffix, Expression(self.later_nonlin))

            self.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            self.add_module(
                "pool_nonlin" + suffix, Expression(self.later_pool_nonlin)
            )

        add_conv_pool_block(
            self, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
        )
        add_conv_pool_block(
            self, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        add_conv_pool_block(
            self, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        # self.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            out = self(
                np_to_th(
                    np.ones(
                        (1, self.in_chans, self.input_window_samples, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.n_filters_4,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        param_dict = dict(list(self.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)


####

def train_test_subject_kfold(data, config, test_subject_id, kfold_k=0):
    
    ## init data
    
    # train_dataset, val_dataset, test_dataset = leave_one_subject_out(data, test_subject_id=test_subject_id, kfold_k=kfold_k)
    train_dataset, val_dataset, test_dataset = data.get_train_val_test_dataset(test_subject_id=test_subject_id, kfold_k=kfold_k)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batchsize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batchsize, shuffle=False)

    ## init model

    # eegnet = CompactEEGNet(num_channel=num_channel, num_classes=num_classes, signal_length=signal_length)
    base_model = Deep4Net(
        in_chans=num_channel,
        n_classes=num_classes,
        input_window_samples=signal_length,
        final_conv_length="auto"
    )

    model = LightningModelClassifier(
        optimizer=config.model.optimizer,
        scheduler=config.model.scheduler,
        optimizer_learning_rate=config.training.learning_rate,
        scheduler_warmup_epochs=config.training.num_warmup_epochs,
    )
    
    model.build_model(model=base_model)

    ## train

    sub_dir = "sub"+ str(test_subject_id) +"_k"+ str(kfold_k)
    logger_tb = TensorBoardLogger(save_dir="tensorboard_logs", name=config.run_name, sub_dir=sub_dir)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(max_epochs=config.training.num_epochs, gpus=config.training.gpus, logger=logger_tb, progress_bar_refresh_rate=0, weights_summary=None, callbacks=[lr_monitor])
    trainer.fit(model, train_loader, val_loader)
    
    ## test
    
    result = trainer.test(dataloaders=test_loader, verbose=False)
    test_acc = result[0]['test_acc_epoch']
    
    return test_acc

####

main_logger.write_to_log("Begin", break_line=True)

test_results_acc = {}
means = []

def k_fold_train_test_all_subjects():
    
    for test_subject_id in config.testing.test_subject_ids:
        print()
        print("running test_subject_id:", test_subject_id)
        
        if test_subject_id not in test_results_acc:
            test_results_acc[test_subject_id] = []
            
        for k in config.testing.kfolds:

            test_acc = train_test_subject_kfold(data, config, test_subject_id, kfold_k=k)
            
            test_results_acc[test_subject_id].append(test_acc)
        
        mean_acc = np.mean(test_results_acc[test_subject_id])
        means.append(mean_acc)
        
        this_result = {
            "test_subject_id": test_subject_id,
            "mean_acc": mean_acc,
            "acc": test_results_acc[test_subject_id],
        }        
        print(this_result)
        main_logger.write_to_log(this_result)

k_fold_train_test_all_subjects()

mean_acc = np.mean(means)
print()
print("mean all", mean_acc)
main_logger.write_to_log("Mean acc: "+str(mean_acc), break_line=True)
