"""
DCGAN Model to get Generator for DefenseGAN Baseline Implementation.
References:
    https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
from torch import nn


class Generator(nn.Module):
    """Generator in the DCGAN."""
    def __init__(self, z_dim=10, image_channels=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen_1 = self.make_gen(z_dim, hidden_dim*8, kernel_size=4, stride=1)
        # Size 1024x4x4
        self.gen_2 = self.make_gen(hidden_dim*8, hidden_dim*4, kernel_size=4, stride=2, padding=1)
        # Size 512x8x8
        self.gen_3 = self.make_gen(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding=1)
        # Size 256x16x16
        self.gen_4 = self.make_gen(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1)
        # Size 128x32x32
        self.gen_final = self.make_gen(hidden_dim, image_channels, kernel_size=4, stride=2, padding=1, final_layer=True)
        # Size 3x64x64

    def make_gen(self, input_channels, output_channels, kernel_size=4, stride=1, padding=0, final_layer=False):
        """Make a generator block with transpose convolutions."""
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        x = self.gen_1(x)
        x = self.gen_2(x)
        x = self.gen_3(x)
        x = self.gen_4(x)
        out = self.gen_final(x)

        return out


class Discriminator(nn.Module):
    def __init__(self, image_channels=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc_1 = self.make_disc(image_channels, hidden_dim, first_layer=True)
        # Size 16x32x32
        self.disc_2 = self.make_disc(hidden_dim, hidden_dim * 2)
        # Size 32x16x16
        self.disc_3 = self.make_disc(hidden_dim * 2, hidden_dim * 4)
        # Size 64x8x8
        self.disc_4 = self.make_disc(hidden_dim * 4, hidden_dim * 8)
        # Size 128x4x4
        self.disc_final = self.make_disc(hidden_dim * 8, 1, final_layer=True)
        # Size 1x1x1

    def make_disc(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False, first_layer=False):
        """Make a discriminator block using convolutions to downsample."""
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=1, bias=False),
                nn.Identity() if first_layer else nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=1, bias=False),
            )

    def forward(self, image):
        x = self.disc_1(image)
        x = self.disc_2(x)
        x = self.disc_3(x)
        x = self.disc_4(x)
        pred = self.disc_final(x)
        pred = pred.flatten()

        return pred
