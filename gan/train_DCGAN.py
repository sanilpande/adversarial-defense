"""
Train a DCGAN to generate images using the CIFAR10 Dataset.
References:
    https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from DCGAN import Generator, Discriminator

DATA_DIR = "/home/sanil/deeplearning/adversarial-detection/cifar10"


def train():
    """Train DCGAN and save the generator and discrinator."""
    
    epochs = 200    
    z_dim = 10
    batch_size = 256
    
    lr = 0.0002 # A learning rate of 0.0002 works well on DCGAN
    beta_1 = 0.5 
    beta_2 = 0.999

    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

    dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    gen = Generator(z_dim).to(device)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    disc = Discriminator().to(device) 
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr/2.0, betas=(beta_1, beta_2))

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    best_loss = 1e5

    for epoch in range(epochs):
        print("Epoch:   ", epoch + 1, end='\n')
        total_discriminator_loss = 0
        total_generator_loss = 0
        display_real = None
        display_fake = None

        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)
            display_real = real


            # UPDATE DISCRIMINATOR
            disc_optimizer.zero_grad()
            noise = torch.randn(batch_size, z_dim, device=device)
            fake = gen(noise)
            
            disc_fake_pred = disc(fake.detach())
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_real_pred = disc(real)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            disc_loss = disc_fake_loss + disc_real_loss

            total_discriminator_loss += disc_loss.item()
            disc_loss.backward(retain_graph=True)
            # if i % 2 == 0:
            #     disc_optimizer.step()


            # UPDATE GENERATOR
            gen_optimizer.zero_grad()
            
            noise = torch.randn(batch_size, z_dim, device=device)
            fake = gen(noise)
            display_fake = fake
            disc_fake_pred = disc(fake)   # Notice no detach
            
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_optimizer.step()

            total_generator_loss += gen_loss.item()

            print('Discriminator Loss: {:.4f} \t Generator Loss: {:.4f} \t Done: {:.4f}'.format(total_discriminator_loss/(i+1),
                total_generator_loss/(i+1), i/len(dataloader)), end='\r')
        
        if (epoch + 1) % 5 == 0 and total_generator_loss < best_loss:
            best_loss = total_generator_loss
            show_tensor_images(display_fake, epoch=epoch)
        # show_tensor_images(display_real)
        # print('Discriminator Loss: {:.4f} \t Generator Loss: {:.4f}'.format(total_discriminator_loss/iteration, total_generator_loss/iteration), end='\r')


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def show_tensor_images(image_tensor, num_images=25, epoch=0):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(str(epoch)+".png")


if __name__ == "__main__":
    train()