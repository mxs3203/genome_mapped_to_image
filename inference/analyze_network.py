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

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("../data/meta_data.csv", filter_by_type="OV")
trainLoader = DataLoader(dataset, batch_size=1, num_workers=10, shuffle=False)
checkpoint = torch.load("../src/classic_cnn/models/tp53_auc_80.pt")
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
    img_exp = Image.fromarray(img_exp, 'L')
    img_mut = Image.fromarray(img_mut, 'L')
    img_cin_g = Image.fromarray(img_cin_g, 'L')
    img_cin_l = Image.fromarray(img_cin_l, 'L')
    total_img = np.dstack((img_cin_g, img_cin_l, img_mut, img_exp))
    total_img = Image.fromarray(total_img, 'RGB')
    return img_cin_g, img_cin_l, img_mut, img_exp, total_img


def show_data(x, y):
    img_cin_g, img_cin_l, img_mut, img_exp, total_img = makeImages(x.cpu().detach().numpy())
    f, axarr = plt.subplots(3, 2)
    axarr[0, 0].imshow(img_cin_l, cmap='Blues', vmin=0, vmax=1)
    axarr[0, 1].imshow(img_cin_g, cmap='Reds', vmin=0, vmax=1)
    axarr[1, 0].imshow(img_mut, cmap='Greens', vmin=0, vmax=1)
    axarr[1, 1].imshow(img_exp, cmap='seismic', vmin=0, vmax=1)
    axarr[2, 1].imshow(img_exp, cmap='seismic', vmin=0, vmax=1)
    f.show()
    plt.imshow(total_img, cmap='plasma', vmin=0, vmax=1)
    plt.title('Total')

    plt.show()


def plot_attribution(att):
    f, axarr = plt.subplots(3, 2)
    axarr[0, 0].imshow(att[0, :, :], cmap='Blues', vmin=0, vmax=np.max(att[0, :, :]))
    axarr[0, 1].imshow(att[1, :, :], cmap='Reds', vmin=0, vmax=np.max(att[1, :, :]))
    axarr[1, 0].imshow(att[2, :, :], cmap='Greens', vmin=0, vmax=np.max(att[2, :, :]))
    axarr[1, 1].imshow(att[3, :, :], cmap='seismic', vmin=0, vmax=np.max(att[3, :, :]))
    axarr[2, 1].imshow(att[3, :, :], cmap='seismic', vmin=0, vmax=np.max(att[4, :, :]))
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
        #plot_attribution(attribution)
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

image = make_image("ID", 1, all_genes)
exp_att = image.analyze_attribution(mean_exp_matrix, 20)
mut_att = image.analyze_attribution(mean_mut_matrix, 20)
gain_att = image.analyze_attribution(mean_gain_matrix, 20)
loss_att = image.analyze_attribution(mean_loss_matrix, 20)
meth_att = image.analyze_attribution(mean_meth_matrix, 20)

for i in exp_att:
    print(i, exp_att[i])
print("\n")
for i in mut_att:
    print(i, mut_att[i])
print("\n")
for i in gain_att:
    print(i, gain_att[i])
print("\n")
for i in loss_att:
    print(i, loss_att[i])
for i in meth_att:
    print(i, meth_att[i])


wordcloud = WordCloud(prefer_horizontal=1,background_color="white",include_numbers=True, width=400, height=400).generate_from_frequencies(gain_att)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Gain")
plt.show()
wordcloud = WordCloud(prefer_horizontal=1,background_color="white",include_numbers=True, width=400, height=400).generate_from_frequencies(loss_att)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Loss")
plt.show()
wordcloud = WordCloud(prefer_horizontal=1,background_color="white",include_numbers=True, width=400, height=400).generate_from_frequencies(mut_att)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Mutation")
plt.show()
wordcloud = WordCloud(prefer_horizontal=1,background_color="white",include_numbers=True, width=400, height=400).generate_from_frequencies(exp_att)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Expression")
plt.show()
wordcloud = WordCloud(prefer_horizontal=1,background_color="white",include_numbers=True, width=400, height=400).generate_from_frequencies(meth_att)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Methylation")
plt.show()