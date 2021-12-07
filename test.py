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

from Utils import *


def test_step_per_epoch(model, valid_loader, device, criterion, epoch, folder_name, draw_images=True):
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for image, label in valid_loader:
            image = image.view(image.size(0), -1).to(device) #rescale to [batch_size, 784]
            out = model(image)
            loss = criterion(out, image)
            valid_loss.append(loss.item())

        if epoch % 10 == 0:
            ae_img, original_img = save_images(folder_name, out, image, epoch)
            if draw_images:
                draw_output(5, ae_img, original_img)

    return np.mean(valid_loss)


def test_clf_per_epoch(model, valid_loader, device, criterion):

    model.eval()
    val_loss = []
    with torch.no_grad():
        for image, label in valid_loader:
            image = image.view(image.size(0), -1).to(device)
            label = label.to(device)
            out = model.classify(image)
            loss = criterion(out, label)
            
            val_loss.append(loss.item())

    return np.mean(val_loss)