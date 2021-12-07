import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt        

def train_step_per_epoch(model, train_loader, device, criterion, optimizer, epoch, folder_name, plot_train=False):
  train_loss = 0
  model.train()
  for image, label in train_loader:
    image = image.view(image.size(0), -1).to(device) #rescale to [batch_size, 784]
    #forward
    out = model(image)
    loss = criterion(out, image)
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*image.size(0)

  if epoch % 10 == 0:
    ae_img = out.view(out.size(0), 1, 28, 28)
    save_image(ae_img, './Images/ae_img_{}.png'.format(epoch)) 
    original_img = image.view(image.size(0), 1, 28, 28)
    save_image(original_img,'./Images/original_img_{}.png'.format(epoch))

    if plot_train:

      plt.figure(figsize=(10,4.5))
      for i in range(5):
        ax = plt.subplot(2,5,i+1)
        plt.imshow(original_img[i].detach().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 5//2:
            ax.set_title('Original images')        
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(ae_img[i].detach().squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == 5//2:
          ax.set_title('Reconstructed images')
      plt.show()

  return train_loss/len(train_loader)