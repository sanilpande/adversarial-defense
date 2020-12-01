"""
Train a GAN to generate images using the CIFAR10 Dataset using W-Loss and Gradient Penalty.
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

    torch.manual_seed(7)
    epochs = 200
    z_dim = 100
    batch_size = 256

    lr = 0.0003 # A learning rate of 0.0002 works well on DCGAN
    beta_1 = 0.5
    beta_2 = 0.999

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

    dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    gen = Generator(z_dim).to(device)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))

    disc = Discriminator().to(device)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr * 1.5, betas=(beta_1, beta_2))

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    c_lambda = 10   # weight of gradient penalty in loss

    for epoch in range(epochs):
        print("Epoch:   ", epoch + 1, end='\n')
        total_discriminator_loss = 0
        total_generator_loss = 0
        display_fake = None

        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)

            # UPDATE DISCRIMINATOR
            for _ in range(5):
                disc_optimizer.zero_grad()
                noise = torch.randn(batch_size, z_dim, device=device)
                fake = gen(noise)

                disc_fake_pred = disc(fake.detach())
                disc_real_pred = disc(real)

                # ratio of real to fake in calculating gp
                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)

                gradient = get_gradient(disc, real, fake.detach(), epsilon)
                gp = grad_penalty(gradient)

                # value of real should go up, fake should go down : so loss is opposite
                disc_loss = -torch.mean(disc_real_pred) + torch.mean(disc_fake_pred) + c_lambda*gp

                total_discriminator_loss += disc_loss.item()
                disc_loss.backward(retain_graph=True)
                # if i % 2 == 0:
                disc_optimizer.step()


            # UPDATE GENERATOR
            gen_optimizer.zero_grad()

            noise = torch.randn(batch_size, z_dim, device=device)
            fake = gen(noise)
            display_fake = fake
            disc_fake_pred = disc(fake)   # Notice no detach

            # for generator, critic prediction should be higher
            gen_loss = -torch.mean(disc_fake_pred)
            gen_loss.backward()
            gen_optimizer.step()

            total_generator_loss += gen_loss.item()

            print('Discriminator Loss: {:.4f} \t Generator Loss: {:.4f} \t Done: {:.4f}'.format(total_discriminator_loss/(i+1),
                total_generator_loss/(i+1), i/len(dataloader)), end='\r')

        if (epoch + 1) % 5 == 0:
            show_tensor_images(display_fake, epoch=epoch)
            torch.save(gen.state_dict, "saved_gen/wgan_gp_gen_{}.pth".format(epoch))


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def get_gradient(model, real, fake, epsilon):
    """
    Return gradient of the model's scores with respect to mixed real and fake images.
    """
    mixed_images = real * epsilon + fake * (1 - epsilon)
    score = model(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=score,
        grad_outputs=torch.ones_like(score), 
        create_graph=True,
        retain_graph=True,
    )[0]
    
    return gradient


def grad_penalty(gradient):
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)
    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty

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
