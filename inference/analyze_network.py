from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from TCGA_GenomeImage.inference.Dataloader import TCGAImageLoader
from TCGA_GenomeImage.src.classic_cnn.Network_Softmax import ConvNetSoftmax
from TCGA_GenomeImage.src.image_to_picture.utils import make_image

all_genes = pd.read_csv("../data/raw_data/all_genes_ordered_by_chr.csv")

cancer_type = "BLCA"
image_type = "193x193Image"
response = "TP53"

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("../data/{}_data/{}/meta_data.csv".format(response, image_type), filter_by_type=cancer_type)
trainLoader = DataLoader(dataset, batch_size=1, num_workers=10, shuffle=False)
checkpoint = torch.load("../src/classic_cnn/models/tp53_193x193_auc_81.pt")
LR = 0.0001
net = ConvNetSoftmax()
print(len(trainLoader))

optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=0.01, lr=LR, weight_decay=0.001)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()
print(len(trainLoader))
occlusion = IntegratedGradients(net)


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

heatmaps_meth = []
heatmaps_loss = []
heatmaps_gains = []
heatmaps_mut = []
heatmaps_exp = []
cnt = 1
for x, type, id, met_1_2_3 in trainLoader:
    print("ID: ", id)
    for d in range(1, 6):
        print("\t",d)
        #show_data(x, met_1_2_3)
        baseline = torch.zeros((1, 5, 197, 197))
        attribution = occlusion.attribute(x, baseline, target=1)
        attribution = attribution.squeeze().cpu().detach().numpy()
        for_heatmap = np.abs(attribution[d - 1, :, :])
        if d == 1:
            heatmaps_gains.append(for_heatmap)
        if d == 2:
            heatmaps_loss.append(for_heatmap)
        if d == 3:
            heatmaps_mut.append(for_heatmap)
        if d == 4:
            heatmaps_exp.append(for_heatmap)
        if d == 5:
            heatmaps_meth.append(for_heatmap)



heatmaps_loss = np.array(heatmaps_loss)
mean_loss_matrix = heatmaps_loss.mean(axis=0)
ax = sns.heatmap(mean_loss_matrix, cmap="YlGnBu")
plt.show()
heatmaps_gains = np.array(heatmaps_gains)
mean_gain_matrix = heatmaps_gains.mean(axis=0)
ax = sns.heatmap(mean_gain_matrix, cmap="YlGnBu")
plt.show()
heatmaps_mut = np.array(heatmaps_mut)
mean_mut_matrix = heatmaps_mut.mean(axis=0)
ax = sns.heatmap(mean_mut_matrix, cmap="YlGnBu")
plt.show()

heatmaps_exp = np.array(heatmaps_exp)
mean_exp_matrix = heatmaps_exp.mean(axis=0)
ax = sns.heatmap(mean_exp_matrix, cmap="YlGnBu")
plt.show()
heatmaps_meth = np.array(heatmaps_meth)
mean_meth_matrix = heatmaps_meth.mean(axis=0)
ax = sns.heatmap(mean_meth_matrix, cmap="YlGnBu")
plt.show()

number_of_genes_returned = 20

image = make_image("ID", 1, all_genes)
exp_att = image.analyze_attribution(mean_exp_matrix, number_of_genes_returned, "Expression")
mut_att = image.analyze_attribution(mean_mut_matrix, number_of_genes_returned, "Mutation")
gain_att = image.analyze_attribution(mean_gain_matrix, number_of_genes_returned, "Gain")
loss_att = image.analyze_attribution(mean_loss_matrix, number_of_genes_returned, "Loss")
meth_att = image.analyze_attribution(mean_meth_matrix, number_of_genes_returned, "Methylation")

total_df = pd.concat([exp_att,mut_att,gain_att,loss_att, meth_att])
total_df.to_csv("../Results/{}_{}_{}_top_{}.csv".format(cancer_type, image_type, response, number_of_genes_returned))

number_of_genes_returned = 38000

image = make_image("ID", 1, all_genes)
exp_att = image.analyze_attribution(mean_exp_matrix, number_of_genes_returned, "Expression")
mut_att = image.analyze_attribution(mean_mut_matrix, number_of_genes_returned, "Mutation")
gain_att = image.analyze_attribution(mean_gain_matrix, number_of_genes_returned, "Gain")
loss_att = image.analyze_attribution(mean_loss_matrix, number_of_genes_returned, "Loss")
meth_att = image.analyze_attribution(mean_meth_matrix, number_of_genes_returned, "Methylation")

total_df = pd.concat([exp_att,mut_att,gain_att,loss_att, meth_att])
total_df.to_csv("../Results/{}_{}_{}_top_{}.csv".format(cancer_type, image_type, response, number_of_genes_returned))

