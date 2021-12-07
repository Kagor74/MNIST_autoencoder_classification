import torch 
from torch.utils.data import Dataset, DataLoader 
import random
import torch.autograd as autograd         
from torch import Tensor                 
import torch.nn as nn                     
import torch.nn.functional as F           
import torch.optim as optim               

import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms  
import numpy as np  
import matplotlib.pyplot as plt        


def train_step_per_epoch(model, train_loader, device, criterion, optimizer):
    train_loss = []
    model.train()
    for image, label in train_loader:
        image = image.view(image.size(0), -1).to(device) #rescale to [batch_size, 784] and send it to GPU
        #forward
        out = model(image)
        loss = criterion(out, image)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    return np.mean(train_loss)


def train_clf_per_epoch(model, train_loader, device, criterion, optimizer):
    model.train()
    train_loss = []
    for image, label in train_loader:
        image = image.view(image.size(0), -1).to(device)
        label = label.to(device)
        #forward
        out = model.classify(image)
        loss = criterion(out, label)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())

    return np.mean(train_loss)