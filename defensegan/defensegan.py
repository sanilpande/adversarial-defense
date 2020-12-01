"""
Implementation of DefenseGAN strategy for defence against adversarial attacks.
https://arxiv.org/abs/1805.06605
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from generator import Generator


DATA_DIR = "/home/sanil/deeplearning/adversarial-detection/fgsm_cifar10"

def main():
    torch.manual_seed(7)

    batch_size = 64
    z_dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = load_generator()
    gen.to(device)

    # Test that generated images are acceptable
    noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
    gen_images = gen(noise)
    show_tensor_images(gen_images)

    classifier = get_classifier()
    classifier.to(device)
    classifier.eval()

    #TODO The normalization vlaues might need to be changed
    transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    # Load the adversarial dataset
    dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform)
    print(len(dataset))
    dataset = torch.utils.data.Subset(dataset, indices=range(0, len(dataset), 100))
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    accuracy = 0.0

    # Loop through dataset
    for i, (adv_images, true_labels) in enumerate(dataloader):
        adv_images = adv_images.to(device)
        true_labels = true_labels.to(device)

        # Get noise vectors to generate defended images.
        defense_z = get_z_list(gen, adv_images)
        defense_z = defense_z.reshape(batch_size, z_dim, 1, 1)

        with torch.no_grad():
            # get defended images
            defended_images = gen(defense_z)
            
            # classify the generated (defended) images
            output = classifier(defended_images)
            pred = torch.argmax(output, dim=-1)

            # evaluate accuracy after defense
            correct = torch.sum(pred == true_labels)
            accuracy += correct.item() / pred.shape[0]
            
            #TODO evaluate per-class accuracy
            print(correct.item() / pred.shape[0])

    accuracy /= len(dataloader)

    print("Accuracy after defense: {:.4f}".format(accuracy))


def get_z_list(model, data, z_num=10, num_gd_steps=50, z_size=100, lr=0.1, device='cuda'):
    """
    Return noise that results in the closest reconstruction to the given data.
    Parameters:
        model   : the Generator model
        data    : the input data (images)
        z_num   : number of trials to get the best noise vector
        num_gd_steps    : number of steps of gradient descent to get to the vector
        z_size  : size of noise vector. 
    """
    batch_size = data.shape[0]
    z_initial = torch.randn(batch_size, z_num, z_size, 1, 1)

    # TODO try different criteria and optimizers
    criterion = nn.MSELoss()

    model.eval()
    best_z_list = torch.zeros(batch_size, z_size).to(device)
    
    for batch in range(batch_size):
        best_z = None
        best_loss = 1e5
        for i in range(z_num):
            # get each noise vector
            z_hat = z_initial[batch, i].to(device)
            z_hat = z_hat.detach().unsqueeze(0)
            z_hat.requires_grad = True
            
            temp_loss = 1e5

            optimizer = torch.optim.SGD([z_hat], lr=lr, momentum=0.7)

            for _ in range(num_gd_steps):
                optimizer.zero_grad()
                fake_image = model(z_hat)    #assume same shape as input data

                loss = criterion(fake_image[0], data[batch])
                loss.backward()
                optimizer.step()

                temp_loss = loss.item()
            
            if temp_loss < best_loss:
                best_loss = temp_loss
                best_z = z_hat
        
        # from IPython import embed; embed()
        best_z_list[batch] = torch.squeeze(best_z)

    return best_z_list


def get_classifier():
    """Get Pretrained Classifier."""
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("../adv_generation/saved_models/res80.pth"))
    return model


def load_generator():
    """Get generator and load with pre-trained weights."""
    gen = Generator(1)
    gen.main[12] = nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=False)
    state_dict = torch.load("netG_epoch_199.pth")
    gen.load_state_dict(state_dict)
    gen.eval()
    return gen


def show_tensor_images(image_tensor, num_images=50, epoch=0):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


if __name__ == "__main__":
    main()