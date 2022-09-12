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

from splearn.data import MultipleSubjects, Benchmark
from splearn.utils import Logger, Config
from splearn.filter.butterworth import butter_bandpass_filter
from splearn.filter.notch import notch_filter
from splearn.filter.channels import pick_channels
from splearn.nn.models import TimeDomainBasedCNN
from splearn.nn.base import LightningModelClassifier

config = {
    "experiment_name": "tcnn_benchmark",
    "data": {
        "load_subject_ids": np.arange(1,36), # get first 5 subjects
        "root": "../data/hsssvep",
        "selected_channels": ["PZ", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"],
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
    start_t = 160
    end_t = start_t + 250
    data_x = data_x[:,:,:,start_t:end_t]
    data.set_data(data_x)

# load data
data = MultipleSubjects(
    dataset=Benchmark, 
    root=os.path.join(path,config.data.root), 
    subject_ids=config.data.load_subject_ids, 
    func_preprocessing=func_preprocessing,
    verbose=True, 
)

num_channel = data.data.shape[2]
num_classes = 40
signal_length = data.data.shape[3]


def train_test_subject_kfold(data, config, test_subject_id, kfold_k=0):
    
    ## init data
    
    # train_dataset, val_dataset, test_dataset = leave_one_subject_out(data, test_subject_id=test_subject_id, kfold_k=kfold_k)
    train_dataset, val_dataset, test_dataset = data.get_train_val_test_dataset(test_subject_id=test_subject_id, kfold_k=kfold_k)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batchsize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batchsize, shuffle=False)

    ## init model

    # eegnet = CompactEEGNet(num_channel=num_channel, num_classes=num_classes, signal_length=signal_length)
    base_model = TimeDomainBasedCNN(num_classes=num_classes, signal_length=signal_length)

    model = LightningModelClassifier(
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
