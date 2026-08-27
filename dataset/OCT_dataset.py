import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
from PIL import Image
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings

def create_dataset(opt,kwargs):
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    
    train_data = np.load("/home/lph/OCT2017/train_data.npy")
    train_labels= np.load("/home/lph/OCT2017/train_label.npy")
    normal_train_data = train_data[train_labels==3]
    normal_train_labels = train_labels[train_labels==3]
    anomaly_train_data = train_data[train_labels!=3]
    anomaly_train_labels = train_labels[train_labels!=3]
    print('all anomalous', anomaly_train_data.shape[0])
    normal_amount = normal_train_data.shape[0]
    print('normal_amount',normal_amount)
    anomaly_amount = int((normal_amount//(1-opt.gamma_p))-normal_amount)
    print(anomaly_amount,'anomaly_amount')

    randIdx_anomaly=np.load("dataset/OCT_randIdx_anomaly%d.npy"%(opt.dataset_ID,))
    
    train_unlabeled_data = np.concatenate((normal_train_data,anomaly_train_data[randIdx_anomaly[:anomaly_amount]]),axis=0)
    train_unlabeled_labels = np.concatenate((normal_train_labels,anomaly_train_labels[randIdx_anomaly[:anomaly_amount]]),axis=0)
    
    anomaly_train_data = anomaly_train_data[randIdx_anomaly[anomaly_amount:]]
    anomaly_train_labels = anomaly_train_labels[randIdx_anomaly[anomaly_amount:]]
    
    auxiliary_amount = int(opt.gamma_l*normal_amount/(1-opt.gamma_l))
    print('auxiliary_amount',auxiliary_amount)

    auxiliary_amounts=[auxiliary_amount-auxiliary_amount//opt.k *(opt.k-1) if i==opt.k-1 else auxiliary_amount//opt.k for i in range(opt.k)]
    auxiliary_data=[]
    auxiliary_labels=[]
    for i in range(opt.k):
        auxiliary_data.append(anomaly_train_data[anomaly_train_labels==i][:auxiliary_amounts[i]])
        auxiliary_labels.append(anomaly_train_labels[anomaly_train_labels==i][:auxiliary_amounts[i]])
    auxiliary_data=np.concatenate(auxiliary_data,axis=0)
    auxiliary_labels=np.concatenate(auxiliary_labels,axis=0)
    
    unlabeled_dataset = TrainDataset(train_unlabeled_data,train_unlabeled_labels)
    auxiliary_dataset = TrainDataset(auxiliary_data,auxiliary_labels)
    
    test_data = np.load("/home/lph/OCT2017/test_data.npy")
    test_labels = np.load("/home/lph/OCT2017/test_label.npy")
    normal_test_data = test_data[test_labels==3]
    normal_test_labels = test_labels[test_labels==3]
    
    anomaly_test_data = test_data[test_labels!=3]
    anomaly_test_labels = test_labels[test_labels!=3]
    
    test_random_normal = int(0.8*normal_test_data.shape[0])
    test_random_anomaly = int(0.8*anomaly_test_data.shape[0])
    test_random_anomalys=[test_random_anomaly//3 if i!=2 else test_random_anomaly-test_random_anomaly//3*2 for i in range(0,3)]
    randIdx_test_normal=np.load("dataset/OCT_randIdx_test_normal%d.npy"%(opt.dataset_ID,))
    randIdx_test_anomaly=np.load("dataset/OCT_randIdx_test_anomaly%d.npy"%(opt.dataset_ID,))
    test_data=[normal_test_data[randIdx_test_normal[:test_random_normal]]]
    test_labels=[normal_test_labels[randIdx_test_normal[:test_random_normal]]]
    anomaly_test_data=anomaly_test_data[randIdx_test_anomaly]
    anomaly_test_labels=anomaly_test_labels[randIdx_test_anomaly]
    for i in range(0,3):
        test_data.append(anomaly_test_data[anomaly_test_labels==i][:test_random_anomalys[i]])
        test_labels.append(anomaly_test_labels[anomaly_test_labels==i][:test_random_anomalys[i]])
    test_data=np.concatenate(test_data,axis=0)
    test_labels=np.concatenate(test_labels,axis=0)
    val_data=[normal_test_data[randIdx_test_normal[test_random_normal:]]]
    val_labels=[normal_test_labels[randIdx_test_normal[test_random_normal:]]]
    for i in range(opt.k):
        val_data.append(anomaly_test_data[anomaly_test_labels==i][test_random_anomalys[i]:])
        val_labels.append(anomaly_test_labels[anomaly_test_labels==i][test_random_anomalys[i]:])
    val_data=np.concatenate(val_data,axis=0)
    val_labels=np.concatenate(val_labels,axis=0)
    val_dataset = TrainDataset(val_data,val_labels)
    test_dataset = TrainDataset(test_data,test_labels)
    return unlabeled_dataset, auxiliary_dataset, val_dataset, test_dataset

class TrainDataset(Dataset):
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):        
        return (self.data[index],self.targets[index])

    def __len__(self):
        return len(self.data)