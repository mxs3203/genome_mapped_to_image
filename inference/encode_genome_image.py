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
cancer_types = ['LUSC','LUAD', 'UCEC', 'THCA', 'COAD', 'SKCM', 'BLCA', 'KIRC', 'STAD', 'BRCA', 'OV', 'HNSC']
# Read this from Metadata!!
image_type = "SquareImg"
folder = "TCGA_Square_Imgs/Metastatic_data"
predictor_column = 0
response_column = 9

# Model Params
net = AE(output_size=12)
LR = 9.700000e-5
checkpoint = torch.load("../src/modeling/models/v3/56_SquareImg_TCGA_Square_Imgs_RandCancerType.pb")
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=1e-6, lr=LR, weight_decay=1e-5)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()

dataset = TCGAImageLoader("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/corrected_metastatic_based_on_stages.csv",
                          folder, image_type, predictor_column, response_column)
trainLoader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

arr = np.empty((0, 4608), float)
ids = []
true_y = []
pred_y = []
for x, y_dat, id,type in trainLoader:
    encoded_genome = net.encode(x)
    id = np.array(id)
    ids = np.concatenate((ids, id))
    arr = np.append(arr, encoded_genome.detach().numpy(), axis=0)

    # y = net(x)
    # probs = torch.softmax(y, dim=1)
    # winners = probs.argmax(dim=1)
    # true_y.append(y_dat.detach().numpy())
    # pred_y.append(winners.detach().numpy())


# df = DataFrame()
# df['sampleid'] = ids
# df['true_y'] = true_y
# df['pred_y'] = pred_y
# print(df)
# df.to_csv("../Results/V3/CancerType/confusion_matrix.csv")


df = DataFrame(arr)
df['sampleid'] = ids
df.to_csv("../Results/V3/RandCancerType/encoded_genomes.csv")


