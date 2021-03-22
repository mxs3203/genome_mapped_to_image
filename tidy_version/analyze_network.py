import seaborn as sns
from captum.attr import Occlusion, IntegratedGradients
from captum.attr import visualization as viz
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

from TCGA_GenomeImage.tidy_version.v2.Dataloader import TCGAImageLoader
from TCGA_GenomeImage.tidy_version.v2.Network_Softmax import ConvNetSoftmax

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("../data/v2/DSS_labels_with_extrastuff.csv",
                          "../data/v2/",
                          filter_by_type="OV",
                          transform=transform)
trainLoader = DataLoader(dataset, batch_size=1, num_workers=10, shuffle=True)
checkpoint = torch.load("v2/models/model_met_0_74.pt")
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
    img_cin_l = x[0, 0, :, :]*255.0
    img_cin_l = ((img_cin_l - img_cin_l.min()) * (1/(img_cin_l.max()+1e-3 - img_cin_l.min()) * 255)).astype('uint8')
    img_cin_g = x[0, 1, :, :]*255.0
    img_cin_g = ((img_cin_g - img_cin_g.min()) * (1/(img_cin_g.max()+1e-3 - img_cin_g.min()) * 255)).astype('uint8')
    img_mut = x[0, 2, :, :] * 255.0
    img_mut = ((img_mut - img_mut.min()) * (1 / (img_mut.max()+1e-3 - img_mut.min()) * 255)).astype('uint8')
    img_mut = Image.fromarray(img_mut, 'P')
    img_cin_g = Image.fromarray(img_cin_g, 'P')
    img_cin_l = Image.fromarray(img_cin_l, 'P')
    total_img = np.dstack((img_cin_l, img_cin_g, img_mut))
    #total_img = Image.fromarray(total_img, 'RGB')
    return total_img
def plot_channels(W):
    # number of output channels
    n_out = W.shape[0]
    # number of input channels
    n_in = W.shape[1]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(n_out, n_in)
    fig.subplots_adjust(hspace=0.001)
    out_index = 0
    in_index = 0
    # plot outputs as rows inputs as columns
    for ax in axes.flat:

        if in_index > n_in - 1:
            out_index = out_index + 1
            in_index = 0
        plt.imshow(W[out_index, in_index, :, :], vmin=w_min, vmax=w_max, cmap='seismic')
        plt.show()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        in_index = in_index + 1

    plt.show()
def show_data(x, y):

    plt.imshow(makeImages(x.cpu().detach().numpy()), cmap='gray')
    plt.title('y='+str(y))
    plt.show()
def plot_activations(A, dim1, dim2, number_rows=1, name=""):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]

    print(n_activations)
    A_min = A.min().item()
    A_max = A.max().item()

    if n_activations == 1:

        # Plot the image.
        plt.imshow(A[0, :], vmin=A_min, vmax=A_max, cmap='seismic')

    else:
        fig, axes = plt.subplots(dim1, dim2)
        fig.set_size_inches(10, 10)
        fig.subplots_adjust(hspace=0.1)
        for i, ax in enumerate(axes.flat):
            if i < n_activations:
                # Set the label for the sub-plot.
                ax.set_xlabel("activation:{0}".format(i + 1))

                # Plot the image.
                ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
                ax.set_xticks([])
                ax.set_yticks([])

heatmaps_loss = []
heatmaps_gains = []
heatmaps_mut = []
cnt = 1
for x, dss, type, id, met_1_2, met_1_2_3, GD in trainLoader:
    for d in range(1,4):
        print("\tID: ", id)
        baseline = torch.zeros((1,3, 192, 193))
        attribution = occlusion.attribute(x, baseline, target=1)
        attribution = attribution.squeeze().cpu().detach().numpy()
        for_heatmap = np.abs(attribution[d-1, :, :])
        if d == 1:
            heatmaps_gains.append(for_heatmap)
        if d == 2:
            heatmaps_loss.append(for_heatmap)
        if d == 3:
            heatmaps_mut.append(for_heatmap)

print(len(heatmaps_loss))
print(len(heatmaps_gains))
print(len(heatmaps_mut))
heatmaps_loss = np.array(heatmaps_loss)
mean_loss_matrix = heatmaps_loss.mean(axis=0)
ax = sns.heatmap(mean_loss_matrix)
plt.show()
heatmaps_gains = np.array(heatmaps_gains)
mean_gain_matrix = heatmaps_gains.mean(axis=0)
ax = sns.heatmap(mean_gain_matrix)
plt.show()
heatmaps_mut = np.array(heatmaps_mut)
mean_mut_matrix = heatmaps_mut.mean(axis=0)
ax = sns.heatmap(mean_mut_matrix)
plt.show()
np.savetxt('mut.csv', mean_mut_matrix, delimiter=',')
np.savetxt('gain.csv', mean_gain_matrix, delimiter=',')
np.savetxt('loss.csv', mean_loss_matrix, delimiter=',')