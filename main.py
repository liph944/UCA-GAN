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

a = 1
b = 0 
c = 0.75
bn = (a+b)/2
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--normal_digit", type=int, default=0, help="noraml class") # 
parser.add_argument("--auxiliary_digit", type=int, default=1, help="abnormal aviliable during training process") # 
parser.add_argument("--gpu", type=str, default='3', help="gpu_num")
parser.add_argument("--dataset", type=str, default='MNIST', help="choice of dataset(CIFAR,F-MNIST,MNIST)")
parser.add_argument("--dir", type=str, default='/summary//', help="save dir")
parser.add_argument("--name", type=str, default='result', help="file name")
parser.add_argument("--gamma_l", type=float, default=0.2, help="ratio of auxiliary data") #
parser.add_argument("--gamma_p", type=float, default=0, help="ratio of pollution data") #
parser.add_argument("--k", type=int, default=1, help="the number of categories of the anomalous data") # 
parser.add_argument("--p_a_max",type=float,default=0.1)
parser.add_argument("--p_a_0",type=float,default=0.01)
parser.add_argument("--dataset_ID",type=int,default=1)
parser.add_argument('--pretrain', action='store_true', help='to pretrain')
parser.add_argument("--pretrain_n_epochs",type=int,default=0,help="number of epochs of training.")

opt = parser.parse_args()
pi = 1-opt.gamma_p
bn = (2*pi-1)/(1+pi)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}
if(opt.k<=1):
    seed = 12
else:
    seed = opt.auxiliary_digit
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
cuda = True if torch.cuda.is_available() else False

if(opt.dataset=='CIFAR'):
    import model.arch_cifar as arch
elif(opt.dataset=='F-MNIST'):
    import model.arch_fmnist as arch

adversarial_loss = torch.nn.MSELoss()

generator = arch.Generator()
discriminator = arch.Discriminator()
encoder = arch.Encoder()
mu_a=torch.tensor(np.zeros((opt.latent_dim,)),dtype=torch.float,requires_grad=True)
s_a=torch.tensor(1/opt.p_a_0-1/opt.p_a_max,dtype=torch.float,requires_grad=True)


if(opt.dataset == 'CIFAR'):
    from dataset.cifarset import create_dataset
    unlabeled_dataset, auxiliary_dataset, val_dataset, test_dataset=create_dataset(opt,kwargs)
    train_pos = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    if(opt.gamma_l == 0.1):
        train_neg = torch.utils.data.DataLoader(auxiliary_dataset, batch_size=opt.batch_size//9, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_l == 0.05):
        train_neg = torch.utils.data.DataLoader(auxiliary_dataset, batch_size=opt.batch_size//19, shuffle=True, drop_last = False,**kwargs)
    elif(opt.gamma_l == 0.2):
        train_neg = torch.utils.data.DataLoader(auxiliary_dataset, batch_size=opt.batch_size//4, shuffle=True, drop_last = False,**kwargs)
    else:
        train_neg = torch.utils.data.DataLoader(auxiliary_dataset, batch_size=50, shuffle=True, drop_last = False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
elif(opt.dataset == 'F-MNIST'):
    from dataset.fmnistset import create_loader
    train_pos, train_neg, val_loader, test_loader = create_loader(opt,kwargs)

if cuda:
    generator = generator.cuda('cuda')
    encoder = encoder.cuda('cuda')
    discriminator = discriminator.cuda('cuda')
    adversarial_loss = adversarial_loss.cuda('cuda')
    mu_a=torch.tensor(mu_a,device=device,requires_grad=True)
    s_a=torch.tensor(s_a,device=device,requires_grad=True)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_G_2=torch.optim.Adam([mu_a],lr=0.001,betas=(0.9,0.999))
optimizer_G_3=torch.optim.Adam([s_a],lr=0.001,betas=(0.9,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.000025, betas=(0.5,0.9))
optimizer_E = torch.optim.Adam(encoder.parameters(),lr=0.0001,betas=(0.5,0.9))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
StepLR_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.98)
StepLR_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.98)
StepLR_E = torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=100, gamma=0.98)

from testing import test_eva
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
auc_re = pd.DataFrame()
best_val_recon = 0
best_test_recon = 0
best_val_zs = 0
best_test_zs = 0
import time 

if opt.pretrain:
    for epoch in range(opt.pretrain_n_epochs):
        i=0
        for (batch_pos,batch_neg) in zip(train_pos,cycle(train_neg)):
            generator.train()
            encoder.train()
            i+=1
            img_pos=batch_pos[0]
            img_neg=batch_neg[0]
            img_pos = img_pos.to(device)  
            img_neg = img_neg.to(device)
            target_pos=batch_pos[1]
            target_neg=batch_neg[1]
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            img=torch.cat([img_pos,img_neg])
            z_out=encoder(img)
            gen=generator(z_out)
            criterion = nn.MSELoss()
            loss = criterion(gen, img)
            loss.backward()
            optimizer_G.step()
            optimizer_E.step()
            print("[Pretrain Epoch %d/%d] [Batch %d/%d]"%(epoch, opt.pretrain_n_epochs, i, len(train_pos)))

for epoch in range(opt.n_epochs):
    start = time.time()
    i = 0
    StepLR_G.step()
    StepLR_E.step()
    StepLR_D.step()
    dxz_list = []
    for (batch_pos,batch_neg) in zip(train_pos,cycle(train_neg)):
        discriminator.train()
        generator.train()
        encoder.train()
    
        i+=1
        
        img_pos = batch_pos[0]
        img_neg = batch_neg[0]
        
        target_pos = batch_pos[1]
        target_neg = batch_neg[1]        

        optimizer_D.zero_grad()   
        valid = torch.ones([img_pos.size(0), 1])
        fake = torch.zeros([img_pos.size(0), 1])       
        img_pos = img_pos.to(device)  
        img_neg = img_neg.to(device)
        valid = valid.to(device)
        valid2=torch.ones([img_pos.shape[0]//4,1]).to(device)
        fake = fake.to(device)
    
        pos_imgs = img_pos.type(Tensor)
        neg_imgs = img_neg.type(Tensor)
    
        z_out_fake = Variable(Tensor(np.random.normal(0, 1, (img_pos.shape[0], opt.latent_dim))))
        z_out_fake = z_out_fake.to(device)
        
        anomaly_z_fake_rand=torch.tensor(np.random.normal(0, 1, (img_pos.shape[0]//4, opt.latent_dim)),dtype=torch.float)
        anomaly_z_fake_rand=anomaly_z_fake_rand.to(device)
        anomaly_z_fake=anomaly_z_fake_rand*10+mu_a
        p_a=1/(1/opt.p_a_max+s_a)
        
        
        img = torch.cat([img_pos,img_neg])
        z_out = encoder(img)
        z_out_real = z_out[:img_pos.shape[0]]
        z_out_neg = z_out[img_pos.shape[0]:]
        
        z = torch.cat([z_out_real,z_out_fake,anomaly_z_fake])
        gen = generator(z)
        gen_imgs_real = gen[:img_pos.shape[0]]
        gen_imgs_fake = gen[img_pos.shape[0]:img_pos.shape[0]*2]
        gen_imgs_anomaly_fake=gen[img_pos.shape[0]*2:]
        
        D_pos_xz = adversarial_loss(discriminator(pos_imgs,z_out_real,'xz')[0], a*valid)
        D_fake_xz = adversarial_loss(discriminator(gen_imgs_fake,z_out_fake,'xz')[0], b*valid)
        D_neg_xz = adversarial_loss(discriminator(neg_imgs,z_out_neg,'xz')[0],bn*(torch.ones([img_neg.size(0), 1])).to(device))
        D_anomaly_fake_xz=adversarial_loss(discriminator(gen_imgs_anomaly_fake,anomaly_z_fake,'xz')[0], b*valid2)
        
        d_loss_xz = D_pos_xz+(1-p_a)*D_fake_xz+p_a*D_anomaly_fake_xz+D_neg_xz
       
        dxz_list.append(d_loss_xz.data.cpu().numpy())
        
        d_loss = d_loss_xz
        
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        
        
        optimizer_G.zero_grad()
        optimizer_G_2.zero_grad()
        optimizer_G_3.zero_grad()
        z_out = encoder(img)
        z_out_real = z_out[:img_pos.shape[0]]
        anomaly_z_fake=anomaly_z_fake_rand*10+mu_a
        z = torch.cat([z_out_real,z_out_fake,anomaly_z_fake])
        gen = generator(z)
        gen_imgs_anomaly_fake=gen[img_pos.shape[0]*2:]
        gen_imgs_fake = gen[img_pos.shape[0]:img_pos.shape[0]*2]
        p_a=1/(1/opt.p_a_max+s_a)
        g_loss = (1-p_a)*adversarial_loss(discriminator(gen_imgs_fake,z_out_fake,'xz')[0], c*valid)+p_a*adversarial_loss(discriminator(gen_imgs_anomaly_fake,anomaly_z_fake,'xz')[0], c*valid2)
        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        optimizer_G_2.step()
        optimizer_G_3.step()
        with torch.no_grad():
            if s_a.item()<0:
                s_a.fill_(0)
        
        optimizer_E.zero_grad()
        z_out = encoder(img)
        z_out_neg = z_out[img_pos.shape[0]:]
        z_out_real = z_out[:img_pos.shape[0]]
        e_loss = adversarial_loss(discriminator(pos_imgs,z_out_real,'xz')[0],c*valid)+ adversarial_loss(discriminator(neg_imgs,z_out_neg,'xz')[0], c*(torch.ones([img_neg.size(0), 1])).to(device))
        e_loss.backward()
        optimizer_E.step()
        
        discriminator.eval()
        generator.eval()
        encoder.eval()
        recon_pos = torch.mean(torch.sum((generator(encoder(img_pos))-img_pos)**2,dim=(1,2,3)))

        print(
                "[Epoch %d/%d] [Batch %d/%d] [recon_pos:%.3f]"
                % (epoch, opt.n_epochs, i, len(train_pos), recon_pos.item())
            )
   
    if((np.mean(dxz_list)<0.015) and epoch>300):
        break
    eva_dic = test_eva(generator,encoder,discriminator,epoch,val_loader,test_loader,device,opt)
    auc_re=pd.concat([auc_re,pd.DataFrame([eva_dic])],ignore_index=True)
    end = time.time()
    time_epoch = end-start
    
    if(eva_dic['val_recon']>best_val_recon):
        best_test_recon = eva_dic['test_recon']
        best_val_recon = eva_dic['val_recon']
    if(eva_dic['val_zs']>best_val_zs):
        best_test_zs = eva_dic['test_zs']
        best_val_zs = eva_dic['val_zs']

    print(
                "[Epoch %d/%d] [val_recon:%.3f][test_recon:%.3f] [val_zs:%.3f][test_zs:%.3f] [best_recon:%.3f][best_zs:%.3f][epoch_time:%.3f]"
                % (epoch, opt.n_epochs,eva_dic['val_recon'],eva_dic['test_recon'],eva_dic['val_zs'],eva_dic['test_zs'],best_test_recon,best_test_zs,time_epoch)
            )

if not os.path.exists(PACK_PATH +opt.dir):
    os.makedirs(PACK_PATH+opt.dir)
auc_re.to_csv(PACK_PATH+opt.dir+opt.name+str(opt.normal_digit)+'vs'+str(opt.auxiliary_digit)+".csv")