from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader

from inference.Dataloader import TCGAImageLoader
from src.classic_cnn.Network_Softmax import ConvNetSoftmax
from src.image_to_picture.utils import make_image


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

# Script Params
cancer_types = [ 'OV','COAD', 'UCEC', 'KIRC','STAD', 'BLCA']# ['KIRC','STAD', 'UCEC','BLCA', 'HNSC', 'OV', 'DLBC', 'COAD']
# Read this from Metadata!!
image_type = "193x193Image"
folder = "Metastatic"
response_var = "type"
meta_data_response_column_index = 8
predictor_column_index = 3

# Model Params
net = ConvNetSoftmax()
LR = 0.0001
checkpoint = torch.load("../src/classic_cnn/models/cancertype_f1_86_auc_98.pt")
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=0.01, lr=LR, weight_decay=0.001)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()
# make IG Instance of a model
occlusion = IntegratedGradients(net)

# Run for every cancer type specified in a list
for type in cancer_types:

    type = str(type)
    # load the data
    dataset = TCGAImageLoader("../data/{}_data/{}/meta_data.csv".format(folder, image_type),
                              filter_by_type=type, response_var=response_var, image_type=image_type,
                              response_column_index=meta_data_response_column_index,
                              predictor_column_index=predictor_column_index)
    trainLoader = DataLoader(dataset, batch_size=1, num_workers=10, shuffle=False)
    print(type, "Samples: ", len(trainLoader))
    # prepare empty lists
    heatmaps_meth = []
    heatmaps_loss = []
    heatmaps_gains = []
    heatmaps_mut = []
    heatmaps_exp = []

    # iterate sample by samples
    for x, type, id, y_dat in trainLoader:
        #print("ID: ", id)
        #print("\t",d)
        #show_data(x, met_1_2_3)
        baseline = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3]))
        attribution = occlusion.attribute(x, baseline, target=int(y_dat))
        attribution = attribution.squeeze().cpu().detach().numpy()
        heatmaps_gains.append(np.abs(attribution[0, :, :]))
        heatmaps_loss.append(np.abs(attribution[1, :, :]))
        heatmaps_mut.append(np.abs(attribution[2, :, :]))
        heatmaps_exp.append(np.abs(attribution[3, :, :]))
        heatmaps_meth.append(np.abs(attribution[4, :, :]))

    # make a mean value for every gene for loses, gains, etc...
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

    number_of_genes_returned = all_genes.shape[0]-1
    folder_for_res = "CancerType"
    image = make_image("ID", 1, all_genes)
    exp_att = image.analyze_attribution(mean_exp_matrix, number_of_genes_returned, "Expression")
    mut_att = image.analyze_attribution(mean_mut_matrix, number_of_genes_returned, "Mutation")
    gain_att = image.analyze_attribution(mean_gain_matrix, number_of_genes_returned, "Gain")
    loss_att = image.analyze_attribution(mean_loss_matrix, number_of_genes_returned, "Loss")
    meth_att = image.analyze_attribution(mean_meth_matrix, number_of_genes_returned, "Methylation")

    total_df = pd.concat([exp_att,mut_att,gain_att,loss_att, meth_att])
    total_df.to_csv("../Results/{}/{}_{}_{}_top_{}.csv".format(folder_for_res,type, image_type, folder_for_res, number_of_genes_returned))

