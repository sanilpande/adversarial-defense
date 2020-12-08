import numpy as np
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision import transforms
import pandas as pd
from io import BytesIO
import base64
import infer
import adversarial_detector
from tqdm import tqdm

from defensegan import get_z_list, load_generator

to_pil = ToPILImage()
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

real_train_loader, real_dev_loader = infer.get_cifar_loader()

adv_train_loader, adv_dev_loader = infer.adv_loader()


def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, "png")
        return base64.b64encode(buffer.getvalue()).decode()


def run():
    # dataloader = adv_dev_loader
    dataloader = real_dev_loader
    basemodel = infer.get_base_model()
    basemodel.eval()

    gen = load_generator()
    
    results_names = ["image", "actual", "prediction"]   #, "dfgan_prediction"
    results = []
    for idx, (image, target) in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            image = image.cuda()

        #assume pre-trained generator is loaded (gen)
        defense_z = get_z_list(gen, image)
        defense_z = defense_z.reshape(defense_z.shape[0], defense_z.shape[1], 1, 1)  #might be unnecessary

        defended_images = gen(defense_z)

        #classifier
        output = basemodel(defended_images)

        prediction = torch.argmax(output, dim=1).detach().cpu().numpy()
        batch_size = image.shape[0]

        image = image.detach().cpu()
        target = target.cpu().numpy()
        for i in range(batch_size):
            results.append(
                [
                    image_base64(to_pil(unnormalize(image[i]))),
                    target[i],
                    prediction[i],
                ]
            )
    print("Generating and Saving Dataframe")
    df = pd.DataFrame(results, columns=results_names)
    df.to_csv("dev_real_dataset_results.csv", index=False)


if __name__ == "__main__":
    run()