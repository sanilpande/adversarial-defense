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

from pytorch_generator import Generator

import wandb


DATA_DIR = "/home/sanil/deeplearning/adversarial-detection/fgsm_cifar10"


def train(config=None):
    """Perform the defense against the adversarial dataset."""
    torch.manual_seed(7)

    if config is not None:
        wandb.init(config=config)
        config = wandb.config

    # Define hyperparameters that are not part of hyperparam search
    batch_size  = 512
    z_dim       = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = load_generator()
    gen.to(device)

    # Manually verify that generated images are acceptable
    # noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
    # gen_images = gen(noise)
    # show_tensor_images(gen_images)

    classifier = get_classifier()
    classifier.to(device)
    classifier.eval()

    #Ensure that normalization values are the same for Generator and Classifier
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    # Load the adversarial dataset
    dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform)
    # Subsample dataset for faster hyperparam search
    dataset = torch.utils.data.Subset(dataset, indices=range(0, len(dataset), 10))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    accuracy = 0.0
    adv_accuracy = 0.0

    # Loop through dataset
    for i, (adv_images, true_labels) in enumerate(dataloader):
        print("Step {} of {}".format(i, len(dataloader)), end='\r')
        adv_images  = adv_images.to(device)
        true_labels = true_labels.to(device)

        # Get noise vectors to generate defended images.
        defense_z = get_z_list(gen, adv_images, config)
        defense_z = defense_z.reshape(batch_size, z_dim, 1, 1)

        with torch.no_grad():
            # get defended images
            defended_images = gen(defense_z)

            # TODO Modify function to save after showing
            # show_tensor_images(adv_images)
            # show_tensor_images(defended_images)
            
            # classify the generated (defended) images
            output = classifier(defended_images)
            pred = torch.argmax(output, dim=-1)
            # evaluate accuracy after defense
            correct = torch.sum(pred == true_labels)
            accuracy += correct.item() / pred.shape[0]

            # classify adversarial (undefended) images
            adv_output = classifier(adv_images)
            adv_pred = torch.argmax(adv_output, dim=-1)
            # evaluate accuracy before defense
            adv_correct = torch.sum(adv_pred == true_labels)
            adv_accuracy += adv_correct.item() / adv_pred.shape[0]
            
            #TODO evaluate per-class accuracy

    accuracy /= len(dataloader)
    adv_accuracy /= len(dataloader)

    wandb.log({"accuracy": accuracy})
    wandb.log({"adv_accuracy": adv_accuracy})

    print("Accuracy before defense:\t{:.6f}".format(adv_accuracy))
    print("Accuracy after defense:\t{:.6f}".format(accuracy))


def get_z_list(model, data, config=None, z_size=100, device='cuda'):
    """
    Return noise that results in the closest reconstruction to the given data.
    Parameters:
        model   : the Generator model
        data    : the input data (images)
        z_size  : size of noise vector

        config:        
            z_num       : number of trials to get the best noise vector
            gd_steps    : number of steps of gradient descent to get to the vector
            lr          : learning rate for SGD on
            criterion   : which criterion to evaluate image reconstruction similarity on
    """
    if config is not None:
        z_num = config.z_num
        num_gd_steps = config.gd_steps
        lr = config.learning_rate
    else:   # defenseGAN default
        z_num = 10
        num_gd_steps = 50
        lr = 1e-3

    batch_size = data.shape[0]
    z_initial = torch.randn(batch_size, z_num, z_size, 1, 1, requires_grad=True, device="cuda")
    # z_original = z_initial.detach().clone()

    losses = torch.empty(batch_size, z_num)

    if config.criterion == 'l1':
        criterion = nn.L1Loss(reduction='none')
    elif config.criterion == 'kl':
        criterion = nn.KLDivLoss(reduction='none')
    else:
        criterion = nn.MSELoss(reduction='none')

    model.eval()
    best_z_list = torch.empty(batch_size, z_size).to(device)
    
    for i in range(z_num):
        # get each noise vector
        z_hat = z_initial[:, i] #.to(device)
        
        optimizer = torch.optim.SGD([z_initial], lr=lr, momentum=0.7)
        # loss saturates around ~500 iteration
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.1)

        for g in range(num_gd_steps):
            optimizer.zero_grad()
            fake_image = model(z_hat)    #assume same shape as input data
            
            loss = criterion(fake_image, data)  #evaluate similarity
            losses[:, i] = loss.sum((1, 2, 3))
            # print("Loss ", loss.mean().item(), end='\r')

            loss.mean().backward()
            optimizer.step()
            scheduler.step()
            # print("Difference ", ((z_hat - z_hat_orig)**2).mean().item(), end='\r')

    best_idx = torch.argmax(losses, dim=-1)
    
    for batch, idx in enumerate(best_idx):
        best_z_list[batch] = z_initial[batch, idx, :, :, :].squeeze()

    return best_z_list


def get_classifier():
    """Get Pretrained Classifier."""
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("../adv_generation/saved_models/res80.pth"))
    
    return model


def load_generator():
    """
    Get generator and load with pre-trained weights.
    https://github.com/csinva/gan-vae-pretrained-pytorch/tree/master/cifar10_dcgan/weights
    """
    gen = Generator(1)
    # The Generator example in PyTorch is for 64x64, weights are for 32x32
    gen.main[12] = nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=False)
    state_dict = torch.load("pretrained_generator/netG_epoch_199.pth")
    gen.load_state_dict(state_dict)
    gen.eval()

    return gen


def show_tensor_images(image_tensor, num_images=50, id=0):
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
    # main()

    # Define sweep configuration
    sweep_config = {
		'method': 'grid', #grid, random
		'metric': {
		'name': 'accuracy',
		'goal': 'maximize',  
		},
		'parameters': {
			'criterion': {
				'values': ['l1', 'l2', 'kl']
			},
			'z_num': {
				'values': [10, 20, 50]
			},
			'gd_steps': {
				'values': [50, 500, 1000, 100]
			},
			'learning_rate': {
				'values': [1e-3, 3e-4, 3e-3]
			},
		}
	}

    #setup sweep controller
    sweep_id = wandb.sweep(sweep_config, project="defense_gan")
    wandb.agent(sweep_id, train, count=20)