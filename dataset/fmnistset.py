import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings



def create_loader(opt,kwargs):
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    dataset1 = datasets.FashionMNIST(root ='data-fashion-mnist', train=True, download=True,transform=data_transform)
    data1 = dataset1.data
    target1 = np.array(dataset1.targets)
   
    abnormal_digits=[i for i in range(0,10) if i != opt.normal_digit]

    data1_p = data1[target1==opt.normal_digit]
    data1_n = data1[target1!=opt.normal_digit]
    target1_p = target1[target1==opt.normal_digit]
    target1_n = target1[target1!=opt.normal_digit]

    randIdx=np.load("dataset/fmnist_randIdx%d.npy"%(opt.dataset_ID,))

    normal_num = data1_p.shape[0]
    abnormal_num = int((normal_num*opt.gamma_p)/(1-opt.gamma_p))

    abnormal_counts=[abnormal_num //9 if i<8 else abnormal_num-abnormal_num//9*8 for i in range(9)]
    target1_n2=target1_n[randIdx]
    data1_n2=data1_n[randIdx]
    rst_data=data1_p
    rst_target=target1_p
    for i in range(9):
        temp=target1_n2==abnormal_digits[i]
        rst_data=np.concatenate((rst_data,data1_n2[temp][:abnormal_counts[i]]),axis=0)
        rst_target=np.concatenate((rst_target,target1_n2[temp][:abnormal_counts[i]]),axis=0)
    dataset1.data=torch.tensor(rst_data)
    dataset1.targets=torch.tensor(rst_target)
    train_pos = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    
    dataset2 = datasets.FashionMNIST(root ='data-fashion-mnist', train=True, download=True,transform=data_transform)
    
    rst_data=[]
    rst_target=[]
    for i in range(9):
        temp=target1_n2==abnormal_digits[i]
        rst_data.append(data1_n2[temp][abnormal_counts[i]:])
        rst_target.append(target1_n2[temp][abnormal_counts[i]:])
    
    unlabeled_num = dataset1.data.shape[0]
    auxiliary_num = int((unlabeled_num*opt.gamma_l)/(1-opt.gamma_l))
    auxiliary_nums=[auxiliary_num // opt.k if i<opt.k-1 else auxiliary_num-auxiliary_num // opt.k *(opt.k-1) for i in range(opt.k)]
    for i in range(9):
        if i<opt.k:
            rst_data[i]=rst_data[i][:auxiliary_nums[i]]
            rst_target[i]=rst_target[i][:auxiliary_nums[i]]
        else:
            rst_data[i]=None
            rst_target[i]=None
    rst_data=np.concatenate(rst_data[:opt.k],axis=0)
    rst_target=np.concatenate(rst_target[:opt.k],axis=0)

    dataset2.data = torch.tensor(rst_data)
    dataset2.targets = torch.tensor(rst_target)

    if(opt.gamma_l == 0.1):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//9, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_l == 0.05):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//19, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_l == 0.2):
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//4, shuffle=True, drop_last = False,**kwargs)
    else:
        train_neg = torch.utils.data.DataLoader(dataset2, batch_size=50, shuffle=True, drop_last = False, **kwargs)
    
    dataset_val = datasets.FashionMNIST(root = 'data-fashion-mnist', train=False, download=True,transform=data_transform)
    data_val = dataset_val.data
    target_val = np.array(dataset_val.targets)
    data_val_normal = data_val[target_val==opt.normal_digit]
    target_val_normal = target_val[target_val==opt.normal_digit]
    data_val_abnormal = data_val[target_val!=opt.normal_digit]
    target_val_abnormal = target_val[target_val!=opt.normal_digit]

    randIdx_normal=np.load("dataset/fmnist_randIdx_normal%d.npy"%(opt.dataset_ID,))
    randIdx_abnormal=np.load("dataset/fmnist_randIdx_abnormal%d.npy"%(opt.dataset_ID,))
    abnormal_digits=[i for i in range(0,10) if i!=opt.normal_digit]
    rst_data=data_val_normal[randIdx_normal[:200]]
    rst_target=target_val_normal[randIdx_normal[:200]]
    data_val_abnormal2=data_val_abnormal[randIdx_abnormal]
    target_val_abnormal2=target_val_abnormal[randIdx_abnormal]
    for i in range(opt.k):
        temp=target_val_abnormal2==abnormal_digits[i]
        rst_data=np.concatenate((rst_data,data_val_abnormal2[temp][:200]),axis=0)
        rst_target=np.concatenate((rst_target,target_val_abnormal2[temp][:200]),axis=0)
    dataset_val.data=torch.tensor(rst_data)
    dataset_val.targets=torch.tensor(rst_target)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)

    dataset_test = datasets.FashionMNIST('data-fashion-mnist', train=False, download=True,transform=data_transform)
    rst_data=data_val_normal[randIdx_normal[200:]]
    rst_target=target_val_normal[randIdx_normal[200:]]
    for i in range(len(abnormal_digits)):
        temp=target_val_abnormal2==abnormal_digits[i]
        rst_data=np.concatenate((rst_data,data_val_abnormal2[temp][200:]),axis=0)
        rst_target=np.concatenate((rst_target,target_val_abnormal2[temp][200:]),axis=0)
    dataset_test.data=torch.tensor(rst_data)
    dataset_test.targets=torch.tensor(rst_target)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    return train_pos,train_neg,val_loader,test_loader