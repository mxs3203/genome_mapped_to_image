import seaborn as sns

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from Dataloader import TCGAImageLoader
from Network_Softmax import ConvNetSoftmax
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.1,0.1,0.1], [0.1,0.1,0.1])])
dataset = TCGAImageLoader("../data/DSS_labels.csv", "../data/", transform)
trainLoader = DataLoader(dataset, batch_size=1, num_workers=10, shuffle=False)
checkpoint = torch.load("checkpoints/ep_454_model.pt")
LR = 0.0001
net = ConvNetSoftmax()
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=1e-2, lr=LR, weight_decay=1e-3)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()

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

print(len(trainLoader))

cnt = 1
for x,y,type,id in trainLoader:
    if y == 1 and type[0] == "PRAD":
        print(type)
        print(id)
        print(y.item())
        show_data(x,y)
        #plot_channels(net.state_dict()['conv1.weight'])
        #plot_channels(net.state_dict()['conv2.weight'])
        # out = net.activations(x)
        # plot_activations(out[0], number_rows=1, name="1st feature map", dim1=4, dim2=5)
        # plt.savefig("/home/mateo/Desktop/PRAD_1/1_feature.png",dpi=100)
        # plot_activations(out[1],  number_rows=1, name="1st activation map",dim1=4,dim2=5)
        # plt.savefig("/home/mateo/Desktop/PRAD_1/1_activation.png",dpi=100)
        # plot_activations(out[2], number_rows=1, name="2nd feature map", dim1=5, dim2=6)
        # plt.savefig("/home/mateo/Desktop/PRAD_1/2_feature.png",dpi=100)
        # plot_activations(out[3], number_rows=1, name="2nd activation map",dim1=5,dim2=6)
        # plt.savefig("/home/mateo/Desktop/PRAD_1/2_activation.png",dpi=100)
        break
    cnt = cnt + 1