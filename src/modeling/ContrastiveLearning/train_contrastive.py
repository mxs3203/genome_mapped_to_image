#!/usr/bin python3
import json
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
# Training Params
import umap
import wandb
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.models import vgg11, resnet50, resnet18, vgg19
from torchvision.transforms import transforms

from src.modeling.ContrastiveLearning.ContrastiveLoss import SupConLoss
# os.environ["WANDB_MODE"]="offline"
from src.modeling.ContrastiveLearning.Dataloader_contrastive import TCGAImageLoaderContrastive
from src.modeling.ContrastiveLearning.genius_network import GENIUS

sys.argv.append("/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/config/metastatic_square")
if len(sys.argv) == 1:
    print("You have to provide a path to a config file")
    quit(1)
else:
    config_path = sys.argv[1]

with open(config_path, "r") as jsonfile:
    config = json.load(jsonfile)
    print("Read successful")


LR = config['LR'] #9.900000000000001e-05
batch_size = config['batch_size']
lr_decay = config['lr_decay']  # 1e-5
weight_decay = config['weight_decay'] # 1e-5
epochs = config['epochs'] #200
start_of_lr_decrease = config['start_of_lr_decrease']#60
# Dataset Params
folder = config['folder'] #"Metastatic_data"
image_type = config['image_type']# "SquereImg"
predictor_column = config['predictor_column'] #
response_column = config['response_column'] #11
# Genome_As_Image_v2
wandb.init(project="Genome_As_Image_contrastive", entity="mxs3203", name="{}_{}".format(config['run_name'],folder),reinit=True)
wandb.save(config_path)
wandb.save("/src/ContrastiveLearning/train_contrastive.py")

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoaderContrastive(config['meta_data'],
                          folder,
                          image_type,
                          predictor_column,
                          response_column)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = int(len(dataset) * 0.6)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)
# resnet50= 224
criterion = SupConLoss(temperature=0.07, base_temperature=0.07)
extractor = vgg19(pretrained=False, num_classes=5000)
input_layer = torch.nn.Sequential(
    # (w-1)s - 2p + k-1 + 1
    torch.nn.ConvTranspose2d(in_channels=5, out_channels=3,padding=1, stride=1, kernel_size=33, bias=False),
    torch.nn.ReLU()
)
net = GENIUS(backbone=extractor, input=input_layer)
net.to(device)

wandb.watch(net)
optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=weight_decay)


def show_res_umap(epoch):
    print("making umap...")
    for x1, x2, y_dat in valLoader:
        images = torch.cat([x1, x2], dim=0)
        y_dat = torch.cat([y_dat, y_dat], dim=0)
        embedding = net(images.to(device))
        break
    umap_res = umap.UMAP().fit_transform(embedding.detach().cpu())
    tsne_res = TSNE(n_components=2).fit_transform(embedding.detach().cpu())
    plt.scatter(
        umap_res[:, 0],
        umap_res[:, 1],
        c=[sns.color_palette()[x] for x in y_dat.detach().cpu()])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection: Ep {}'.format(epoch), fontsize=24)
    plt.show()
    plt.scatter(
        tsne_res[:, 0],
        tsne_res[:, 1],
        c=[sns.color_palette()[x] for x in y_dat.detach().cpu()])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('TSNE projection: Ep {}'.format(epoch), fontsize=24)
    plt.show()

best_loss = float("+inf")
best_back_bone = None
best_input_layer = None

for ep in range(epochs):
    net.train()
    train_loss_epoch = 0
    val_loss_epoch = 0

    for x1,x2,y_dat in trainLoader:
        bsz = y_dat.shape[0]
        optimizer.zero_grad()
        images = torch.cat([x1, x2], dim=0)
        embedding = net(images.to(device))
        f1, f2 = torch.split(embedding, [bsz,bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, y_dat)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    net.eval()
    with torch.no_grad():
        for x1,x2, y_dat in valLoader:
            bsz = y_dat.shape[0]
            images = torch.cat([x1, x2], dim=0)
            embedding = net(images.to(device))
            f1, f2 = torch.split(embedding, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, y_dat)
            val_loss_epoch += loss.item()
    if val_loss_epoch/(len(valLoader)) < best_loss and ep > 5:
        path_backbone,backbone, path_input,input_l,optimizer_state = net.store_backbone(ep=ep, optimizer=optimizer,loss=val_loss_epoch/(len(valLoader)), image_type=image_type,folder=folder)
        #wandb.save(path_backbone)
        #wandb.save(path_input)
        best_back_bone = backbone
        best_input_layer = input_l
        best_loss = val_loss_epoch/(len(valLoader))
        print("Best loss... saving models state_dict")
    if ep % 5 == 0:
        show_res_umap(ep)
    print("Epoch: ", ep)
    print("\tTrain loss: ", train_loss_epoch / (len(trainLoader)))
    print("\tVal loss: ", val_loss_epoch/(len(valLoader)))
    wandb.log({"Train/loss":   train_loss_epoch / (len(trainLoader)),
                   "Test/loss":  val_loss_epoch/(len(valLoader))})

