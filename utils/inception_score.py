"""
Calculates the inception score for a trained generator.
Improved Techniques for Training GANs : https://arxiv.org/abs/1606.03498
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from defensegan import Generator


def inception_score(generator, data_size=1e3, batch_size=2, splits=5, z_dim=100, device='cuda'):
    """
    Calculate Inception Score for a trained generator.
    Params:
        generator   : Generator model
        data_size   : Number of iteration to generate for
        batch_size  : batch size (default 2 to run on small gpu)
        splits      : splits to calculate Inception Score over
        z_dim       : dimension of noise vector
    """
    
    model = inception_v3(pretrained=True)
    model.eval()
    model.to(device)

    total_size = data_size * batch_size
    predictions = np.zeros((int(total_size), 1000))

    for i in range(int(data_size)):
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)

        fake_im = generator(z)
        fake_im = F.interpolate(fake_im, size=(299, 299), mode='bilinear')
        
        output = model(fake_im)
        pred = F.softmax(output).data.cpu().numpy()

        predictions[i*batch_size : i*batch_size + batch_size] = pred

    splits = 5

    split_scores = []
    for k in range(splits):
        subset = predictions[int(k * (total_size // splits)): int((k+1) * (total_size // splits)), :]

        mean = np.mean(subset, axis=0)
        scores = []
        for sub in subset:
            scores.append(entropy(sub, mean))
        
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)


def load_generator():
    gen = Generator(1)
    gen.main[12] = nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=False)
    state_dict = torch.load("../defensegan/pretrained_generator/netG_epoch_199.pth")
    gen.load_state_dict(state_dict)
    gen.eval()

    return gen


if __name__ == "__main__":
    gen = load_generator()
    gen.to('cuda')

    score = inception_score(gen)

    print("Inception Score:\t", score)