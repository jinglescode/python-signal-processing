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

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning
pytorch_lightning.utilities.distributed.log.setLevel(logging.ERROR)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from splearn.data import MultipleSubjects, Benchmark, PyTorchDataset2Views
from splearn.utils import Logger, Config
from splearn.filter.butterworth import butter_bandpass_filter
from splearn.filter.notch import notch_filter
from splearn.filter.channels import pick_channels
from splearn.nn.models import ConvCA, ConvCaLighting
from splearn.cross_decomposition.reference_frequencies import generate_reference_signals

config = {
    "experiment_name": "convca_benchmark_nokfold",
    "data": {
        "load_subject_ids": np.arange(1,36),
        "root": "../data/hsssvep",
        "selected_channels": ["PZ", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"],
        "duration": 1,
    },
    "model": {
        "optimizer": "adamw",
        "scheduler": "cosine_with_warmup",
    },
    "training": {
        "num_epochs": 100,
        "num_warmup_epochs": 20,
        "learning_rate": 0.03,
        "gpus": [0],
        "batchsize": 256,
    },
    "testing": {
        "test_subject_ids": np.arange(1,36),
        "kfolds": np.arange(0,3),
    },
    "seed": 1234
}
main_logger = Logger(filename_postfix=config["experiment_name"])
main_logger.write_to_log("Config")
main_logger.write_to_log(config)
config = Config(config)

seed_everything(config.seed)

# define custom preprocessing steps
def func_preprocessing(data):
    data_x = data.data
    data_x = pick_channels(data_x, channel_names=data.channel_names, selected_channels=config.data.selected_channels)
    data_x = notch_filter(data_x, sampling_rate=data.sampling_rate, notch_freq=50.0)
    data_x = butter_bandpass_filter(data_x, lowcut=7, highcut=90, sampling_rate=data.sampling_rate, order=6)
    start_t = 35
    end_t = start_t + (config.data.duration * data.sampling_rate)
    data_x = data_x[:,:,:,start_t:end_t]
    data.set_data(data_x)


# prepare data loader
def leave_one_subject_out(data, **kwargs):
    test_subject_id = kwargs["test_subject_id"] if "test_subject_id" in kwargs else 1
    kfold_k = kwargs["kfold_k"] if "kfold_k" in kwargs else 0
    kfold_split = kwargs["kfold_split"] if "kfold_split" in kwargs else 3
    
    num_subjects = data.data.shape[0]
    num_trials = data.data.shape[1]
    num_channel = data.data.shape[2]
    size = data.data.shape[3]
    sampling_rate = data.sampling_rate
    target_frequencies = data.stimulus_frequencies

    ref = generate_reference_signals(target_frequencies, size, sampling_rate, num_harmonics=1)
    ref = ref[:, 0, :]
    ref = np.expand_dims(ref, axis=1)
    ref = np.repeat(ref, num_channel, axis=1)
    ref = np.transpose(ref, (1,2,0))
    ref = np.expand_dims(ref, axis=0)

    # get test data
    test_sub_idx = np.where(data.subject_ids == test_subject_id)[0][0]
    selected_subject_data = data.data[test_sub_idx]
    selected_subject_targets = data.targets[test_sub_idx]
    selected_subject_ref = np.repeat(ref, selected_subject_data.shape[0], axis=0)
    test_dataset = PyTorchDataset2Views(selected_subject_data, selected_subject_ref, selected_subject_targets)

    # get train val data
    indices = np.arange(data.data.shape[0])
    train_val_data = data.data[indices!=test_sub_idx, :, :, :]
    train_val_data = train_val_data.reshape((train_val_data.shape[0]*train_val_data.shape[1], train_val_data.shape[2], train_val_data.shape[3]))
    train_val_targets = data.targets[indices!=test_sub_idx, :]
    train_val_targets = train_val_targets.reshape((train_val_targets.shape[0]*train_val_targets.shape[1]))

    # train val split
    # (X_train, y_train), (X_val, y_val) = data.dataset_split_stratified(train_val_data, train_val_targets, k=kfold_k, n_splits=kfold_split)
    # train_ref = np.repeat(ref, X_train.shape[0], axis=0)
    # val_ref = np.repeat(ref, X_val.shape[0], axis=0)
    # train_dataset = PyTorchDataset2Views(X_train, train_ref, y_train)
    # val_dataset = PyTorchDataset2Views(X_val, val_ref, y_val)
    # return train_dataset, val_dataset, test_dataset

    # no kfold
    X_train = train_val_data
    train_ref = np.repeat(ref, X_train.shape[0], axis=0)
    y_train = train_val_targets
    train_dataset = PyTorchDataset2Views(X_train, train_ref, y_train)
    return train_dataset, test_dataset
    

# load data
data = MultipleSubjects(
    dataset=Benchmark, 
    root=os.path.join(path,config.data.root), 
    subject_ids=config.data.load_subject_ids, 
    func_preprocessing=func_preprocessing,
    func_get_train_val_test_dataset=leave_one_subject_out,
    verbose=True, 
)

num_channel = data.data.shape[2]
num_classes = data.stimulus_frequencies.shape[0]
signal_length = data.data.shape[3]

##### test data

test_subject_id = 1
train_dataset, test_dataset = data.get_train_val_test_dataset(test_subject_id=test_subject_id)
train_loader = DataLoader(train_dataset, batch_size=config.training.batchsize, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=config.training.batchsize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.training.batchsize, shuffle=False)

# print()
print("train_loader", train_loader.dataset.data_view1.shape, train_loader.dataset.data_view2.shape, train_loader.dataset.targets.shape)
# print("val_loader", val_loader.dataset.data_view1.shape, val_loader.dataset.data_view2.shape, val_loader.dataset.targets.shape)
print("test_loader", test_loader.dataset.data_view1.shape, test_loader.dataset.data_view2.shape, test_loader.dataset.targets.shape)

######

def train_test_subject_kfold(data, config, test_subject_id, kfold_k=0):
    
    ## init data
    # train_dataset, val_dataset, test_dataset = data.get_train_val_test_dataset(test_subject_id=test_subject_id, kfold_k=kfold_k)
    # train_loader = DataLoader(train_dataset, batch_size=config.training.batchsize, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.training.batchsize, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=config.training.batchsize, shuffle=False)
    # no kfold
    train_dataset, test_dataset = data.get_train_val_test_dataset(test_subject_id=test_subject_id)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batchsize, shuffle=False)

    ## init model
    base_model = ConvCA(num_channel=num_channel, num_classes=num_classes, signal_length=signal_length)
    model = ConvCaLighting(
        optimizer=config.model.optimizer,
        scheduler=config.model.scheduler,
        optimizer_learning_rate=config.training.learning_rate,
        scheduler_warmup_epochs=config.training.num_warmup_epochs,
    )
    model.build_model(model=base_model)

    ## train
    sub_dir = "sub"+ str(test_subject_id) +"_k"+ str(kfold_k)
    logger_tb = TensorBoardLogger(save_dir="tensorboard_logs", name=config.experiment_name, sub_dir=sub_dir)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(max_epochs=config.training.num_epochs, gpus=config.training.gpus, logger=logger_tb, progress_bar_refresh_rate=0, weights_summary=None, callbacks=[lr_monitor])
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, train_loader)
    
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
        
        # k-fold
        # for k in config.testing.kfolds:
        #     test_acc = train_test_subject_kfold(data, config, test_subject_id, kfold_k=k)
        #     test_results_acc[test_subject_id].append(test_acc)
        # mean_acc = np.mean(test_results_acc[test_subject_id])
        # means.append(mean_acc)

        # one fold:
        mean_acc = train_test_subject_kfold(data, config, test_subject_id)
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
