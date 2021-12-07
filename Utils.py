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


def draw_output(n_images, ae_img, original_img):
    """

    This function just for draw image from the output of AE and original image.

    n_images: umber of images ( <= batch_size).
    ae_img: outputs of AE. 
    original_img: original images.

    """
    plt.figure(figsize=(10,4.5))
    for i in range(5):
        ax = plt.subplot(2,5,i+1)
        plt.imshow(original_img[i].cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 5//2:
            ax.set_title('Original images')        
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(ae_img[i].cpu().squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == 5//2:
            ax.set_title('Reconstructed images')
    plt.show()


def save_images(folder_to_save, ae_img, original_img, epoch):
    """
    Function for saving images

    """
    ae_img = ae_img.view(ae_img.size(0), 1, 28, 28)
    save_image(ae_img, './' + folder_to_save + '/ae_img_{}.png'.format(epoch)) 
    original_img = original_img.view(original_img.size(0), 1, 28, 28)
    save_image(original_img,'./' + folder_to_save + '/original_img_{}.png'.format(epoch))

    return ae_img, original_img


def check_accuracy_classification(model, test_loader, device,criterion):

    test_loss = []
    model.eval() 
    with torch.no_grad():
        for image, target in test_loader:
            image = image.view(image.size(0), -1).to(device)
            target = target.to(device)
            output = model.classify(image)
            # print(output.shape)
            loss = criterion(output, target)

            test_loss.append(loss.item())

            _, pred = torch.max(output, 1)
            # compare predictions to label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # ctest accuracy for each object class
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    return np.mean(test_loss), class_correct, class_total