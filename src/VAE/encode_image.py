import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt

from src.VAE.VAEModel import ConvVAE
from src.unet_shape.Dataloader import TCGAImageLoader


folder = "Metastatic_data"
image_type = "193x193Image"
predictor_column = 3
response_column = 8

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = TCGAImageLoader("../../data/{}/{}/meta_data.csv".format(folder, image_type),
                          folder, image_type, predictor_column, response_column, filter_by_type=['OV', 'COAD', 'UCEC', 'KIRC','STAD', 'BLCA'])
torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainLoader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)


model = ConvVAE()
model.load_state_dict(torch.load("../saved_models/vae_model_ep_396.pt"))
model.eval()


for images in trainLoader:
    images = images.to(device)
    recon_images, mu, logvar = model(images)
    for i in range(60):
        im = np.array(recon_images.squeeze(dim=0).cpu().detach())
        plt.imshow(im[i, :, :], cmap='gray')
        plt.show()
    break
