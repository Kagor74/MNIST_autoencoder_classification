import numpy as np
import torch

def test_step_per_epoch(model, valid_loader, device, criterion):
  valid_loss = 0
  model.eval()
  with torch.no_grad():
    for image, label in valid_loader:
      image = image.view(image.size(0), -1).to(device) #rescale to [batch_size, 784]
      out = model(image)
      loss = criterion(out, image)
      valid_loss += loss.item()*image.size(0)
  return valid_loss/len(valid_loader)