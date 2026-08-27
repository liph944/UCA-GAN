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
    
    train_data = np.load("/home/lph/CIFAR-10/train_data.npy")
    train_labels= np.load("/home/lph/CIFAR-10/train_label.npy")
    normal_train_data = train_data[train_labels==opt.normal_digit]
    normal_train_labels = train_labels[train_labels==opt.normal_digit]
    anomaly_train_data = train_data[train_labels!=opt.normal_digit]
    anomaly_train_labels = train_labels[train_labels!=opt.normal_digit]
    print('all anomalous', anomaly_train_data.shape[0])
    normal_amount = normal_train_data.shape[0]
    print('normal_amount',normal_amount)
    anomaly_amount = int((normal_amount*opt.gamma_p)/(1-opt.gamma_p))
    print(anomaly_amount,'anomaly_amount')
    anomaly_digits=[i for i in range(10) if i!=opt.normal_digit]
   
    randIdx_anomaly=np.load("dataset/randIdx%d.npy"%(opt.dataset_ID,))

    anomaly_amounts=[anomaly_amount //9 if i<8 else anomaly_amount-anomaly_amount//9*8 for i in range(9)]

    anomaly_train_data_2=anomaly_train_data[randIdx_anomaly]
    anomaly_train_labels_2=anomaly_train_labels[randIdx_anomaly]
    rst_data=[normal_train_data]
    rst_labels=[normal_train_labels]
    for i in range(9):
        temp=anomaly_train_labels_2==anomaly_digits[i]
        rst_data.append(anomaly_train_data_2[temp][:anomaly_amounts[i]])
        rst_labels.append(anomaly_train_labels_2[temp][:anomaly_amounts[i]])
    train_unlabeled_data=np.concatenate(rst_data,axis=0)
    train_unlabeled_labels=np.concatenate(rst_labels,axis=0)
    
    left_data=[]
    left_labels=[]
    for i in range(9):
        temp=anomaly_train_labels_2==anomaly_digits[i]
        left_data.append(anomaly_train_data_2[temp][anomaly_amounts[i]:])
        left_labels.append(anomaly_train_labels_2[temp][anomaly_amounts[i]:])
    
    unlabeled_amount=train_unlabeled_data.shape[0]
    auxiliary_amount = int(opt.gamma_l*unlabeled_amount/(1-opt.gamma_l))
    print('auxiliary_amount',auxiliary_amount)

    auxiliary_amounts=[auxiliary_amount-(auxiliary_amount//opt.k *(opt.k-1)) if i==opt.k-1 else auxiliary_amount//opt.k for i in range(opt.k)]
    rst_data=[]
    rst_labels=[]
    for i in range(opt.k):
        rst_data.append(left_data[i][:auxiliary_amounts[i]])
        rst_labels.append(left_labels[i][:auxiliary_amounts[i]])
        
    auxiliary_data=np.concatenate(rst_data,axis=0)
    auxiliary_labels=np.concatenate(rst_labels,axis=0)
    
    unlabeled_dataset = TrainDataset(train_unlabeled_data,train_unlabeled_labels)
    auxiliary_dataset = TrainDataset(auxiliary_data,auxiliary_labels)
    
    
    test_data = np.load("/home/lph/CIFAR-10/test_data.npy")
    test_labels = np.load("/home/lph/CIFAR-10/test_label.npy")
    normal_test_data = test_data[test_labels==opt.normal_digit]
    normal_test_labels = test_labels[test_labels==opt.normal_digit]
    
    anomaly_test_data = test_data[test_labels!=opt.normal_digit]
    anomaly_test_labels = test_labels[test_labels!=opt.normal_digit]
    
    test_random_normal = int(0.8*normal_test_data.shape[0]) #划分测试集和验证集
    test_random_anomaly = int(0.8*anomaly_test_data.shape[0])
    test_random_anomalys=[test_random_anomaly//9 if i!=8 else test_random_anomaly-test_random_anomaly//9*8 for i in range(0,9)]
    randIdx_test_normal=np.load("dataset/randIdx_normal%d.npy"%(opt.dataset_ID,))
    randIdx_test_anomaly=np.load("dataset/randIdx_abnormal%d.npy"%(opt.dataset_ID,))
    test_data=[normal_test_data[randIdx_test_normal[:test_random_normal]]]
    test_labels=[normal_test_labels[randIdx_test_normal[:test_random_normal]]]
    anomaly_test_data=anomaly_test_data[randIdx_test_anomaly]
    anomaly_test_labels=anomaly_test_labels[randIdx_test_anomaly]
    for i in range(0,9):
        test_data.append(anomaly_test_data[anomaly_test_labels==anomaly_digits[i]][:test_random_anomalys[i]])
        test_labels.append(anomaly_test_labels[anomaly_test_labels==anomaly_digits[i]][:test_random_anomalys[i]])
    test_data=np.concatenate(test_data,axis=0)
    test_labels=np.concatenate(test_labels,axis=0)
    val_data=[normal_test_data[randIdx_test_normal[test_random_normal:]]]
    val_labels=[normal_test_labels[randIdx_test_normal[test_random_normal:]]]
    for i in range(opt.k):
        val_data.append(anomaly_test_data[anomaly_test_labels==anomaly_digits[i]][test_random_anomalys[i]:])
        val_labels.append(anomaly_test_labels[anomaly_test_labels==anomaly_digits[i]][test_random_anomalys[i]:])
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
    
    
