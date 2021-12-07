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

class Encoder(nn.Module):
    def __init__(self, latent_space):
        super().__init__()

        self.latent_space = latent_space
        self.encoder = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_space)
            )
    def forward(self, x):
        encoded = self.encoder(x)

        return encoded


class Decoder(nn.Module):
    def __init__(self, latent_space):
        super().__init__()

        self.latent_space = latent_space
        self.decoder = nn.Sequential(
                nn.Linear(latent_space, 64),
                nn.ReLU(),
                nn.Linear(64, 128), 
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 784)
            )

    def forward(self, x):

        decoded = self.decoder(x)
        return decoded


class Classifier(nn.Module):
    def __init__(self, latent_space):
        super().__init__() 


        self.classifier = nn.Sequential(
                nn.Linear(latent_space, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
                nn.LogSoftmax(dim=1)
            )

    def forward(self, x):

        cls = self.classifier(x)
        return cls

class AutoEncoderClassifier(nn.Module):
    def __init__(self, latent_space):
        super().__init__()

        self.latent_space = latent_space
        self.encoder = Encoder(latent_space)
        self.decoder = Decoder(latent_space)
        self.clf = Classifier(latent_space)

    def TurnOffEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
                    
    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)

        return out
  
    def classify(self,x):
        encoded = self.encoder(x)
        cls_out = self.clf(encoded)

        return cls_out