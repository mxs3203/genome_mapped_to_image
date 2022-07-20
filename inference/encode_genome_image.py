import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pandas import DataFrame
from torch.utils.data import DataLoader

from src.AutoEncoder.AE_Square import AE
from src.modeling.Dataloader import TCGAImageLoader


def makeImages(x):
    img_cin_g = x[0, 0, :, :]
    img_cin_l = x[0, 1, :, :]
    img_mut = x[0, 2, :, :]
    img_exp = x[0, 3, :, :]
    img_meth = x[0, 4, :, :]
    img_exp = Image.fromarray(img_exp, 'L')
    img_mut = Image.fromarray(img_mut, 'L')
    img_cin_g = Image.fromarray(img_cin_g, 'L')
    img_cin_l = Image.fromarray(img_cin_l, 'L')
    img_meth = Image.fromarray(img_meth, 'L')
    total_img = np.dstack((img_cin_g, img_cin_l, img_mut, img_exp, img_meth))
    total_img = Image.fromarray(total_img, 'RGB')
    return img_cin_g, img_cin_l, img_mut, img_exp, img_meth, total_img

def show_data(x, y):
    img_cin_g, img_cin_l, img_mut, img_exp, img_meth, total_img = makeImages(x.cpu().detach().numpy())
    f, axarr = plt.subplots(3, 2)
    axarr[0, 0].imshow(img_cin_l, cmap='Blues', vmin=0, vmax=1)
    axarr[0, 1].imshow(img_cin_g, cmap='Reds', vmin=0, vmax=1)
    axarr[1, 0].imshow(img_mut, cmap='Greens', vmin=0, vmax=1)
    axarr[1, 1].imshow(img_exp, cmap='seismic', vmin=0, vmax=1)
    axarr[2, 0].imshow(img_meth, cmap='seismic', vmin=0, vmax=1)
    axarr[2, 1].imshow(total_img, cmap='plasma', vmin=0, vmax=1)
    f.show()

all_genes = pd.read_csv("../data/raw_data/all_genes_ordered_by_chr.csv")
#all_genes = all_genes[all_genes['name2'] != "TP53"]
# Script Params
cancer_types =['OV', 'COAD', 'UCEC', 'KIRC','STAD', 'BLCA'] # ['KIRC','STAD', 'UCEC','BLCA', , 'OV', 'DLBC', 'COAD']
# Read this from Metadata!!
image_type = "SquareImg"
folder = "Metastatic_data"
predictor_column = 3 # 3=n_dim_img,4=flatten
response_column = 10 # 5=met,6=wgii,7=tp53, 9= type

# Model Params
net = AE(output_size=6)
LR = 9.900000000000001e-05
checkpoint = torch.load("../src/modeling/models/v1/SquareImg_RandomCancerType.pb")
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay= 1e-6, lr=LR, weight_decay= 1e-6)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()

dataset = TCGAImageLoader("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/main_meta_data.csv",
                          folder, image_type, predictor_column, response_column,
                          filter_by_type=['OV', 'COAD', 'UCEC', 'KIRC','STAD', 'BLCA'] )
trainLoader = DataLoader(dataset, batch_size=120, num_workers=10, shuffle=False)

arr = np.empty((0, 128), float)
ids = []
for x, y_dat, id in trainLoader:
    encoded_genome = net.encode(x)
    id = np.array(id)
    ids = np.concatenate((ids, id))
    arr = np.append(arr, encoded_genome.detach().numpy(), axis=0)

df = DataFrame(arr)
df['sampleid'] = ids
df.to_csv("../Results/V2/RandomCancerType/encoded_genomes.csv")


