"""
Basic implementation of Defense GAN strategy.
https://arxiv.org/abs/1805.06605
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

def get_z_list(model, data, z_num=10, num_gd_steps=50, z_size=10, device='cuda'):
    """
    Return noise that results in the closest reconstruction to the given data.
    Parameters:
        model   : the Generator model
        data    : the input data (images)
        z_num   : number of trials to get the best noise vector
        num_gd_steps    : number of steps of gradient descent to get to the vector
        z_size  : size of noise vector 
    """
    z_initial = torch.randn(z_num, data.shape[0], z_size)
    lr = 0.1

    # try different criteria and optimizers
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD([z_hat], lr=lr, momentum=0.7)

    model.eval()
    best_z = None
    best_loss = 1e5
    
    for i in range(z_num):
        # get each noise vector
        z_hat = z_initial[i].to(device)
        z_hat = z_hat.detach()
        z_hat.requires_grad = True
        
        temp_loss = 1e5

        for gd_step in range(num_gd_steps):
            optimizer.zero_grad()
            fake_image = model(z_hat)    #assume same shape as input data

            loss = criterion(fake_image, data)
            loss.backward()
            optimizer.step()

            temp_loss = loss.item()
        
        if temp_loss < best_loss:
            best_loss = temp_loss
            best_z = z_hat

    return best_z





