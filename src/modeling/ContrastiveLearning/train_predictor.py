#!/usr/bin python3
import json
import sys
import os
import torch
# Training Params
import umap as umap
import wandb
import torch.nn.functional as F
from pytorch_metric_learning.losses import ContrastiveLoss
from sklearn.manifold import TSNE
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.transforms import transforms
import numpy as np

from src.ResNet50.resnet50_with_predictionlayer import ResNetPred
from src.modeling.ContrastiveLearning.ContrastiveLoss import SimCLR_Loss

os.environ["WANDB_MODE"]="offline"
from src.ResNet50.resblock import ResBottleneckBlock
from src.ResNet50.resnet50 import ResNet
from src.modeling.ContrastiveLearning.Dataloader_contrastive import TCGAImageLoaderContrastive

sys.argv.append("/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/config/cancer_type_square")
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
wandb.save("/src/AutoEncoder/AE_Square.py")
criteria = CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoaderContrastive(config['meta_data'],
                          folder,
                          image_type,
                          predictor_column,
                          response_column)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)

cost_func = SimCLR_Loss(batch_size)
extractor = ResNet(5, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=False, outputs=512)
net = ResNetPred(5, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=False, outputs=512,
                 path_to_pth="../checkpoints/40_SquareImg_TCGA_Square_ImgsGainLoss_harsh_Metastatic_data.pb")
net.to(device)
#print(summary(net, (5, 193, 193)))


wandb.watch(net)
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)


def show_res_umap():
    print("making umap...")
    pass


for ep in range(epochs):
    net.train()
    train_loss_epoch = 0
    val_loss_epoch = 0

    for x1,x2,y_dat in trainLoader:
        optimizer.zero_grad()
        embedding1, projection1 = net(x1.to(device))
        embedding2, projection2 = net(x2.to(device))
        loss = cost_func(projection1, projection2)
        if loss != -1:
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
    net.eval()
    with torch.no_grad():
        for x1, x2, y_dat in valLoader:
            embedding1, projection1 = net(x1.to(device))
            embedding2, projection2 = net(x2.to(device))
            loss = cost_func(projection1,projection2)
            if loss != -1:
                val_loss_epoch += loss.item()
    if ep % 10 == 0:
        show_res_umap()
        torch.save({
            'epoch': ep,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/{}_{}_{}.pb".format(ep, image_type,
                                                                                                     folder.replace("/",
                                                                                                                    "_")))
        wandb.save(
            "/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/{}_{}_{}.pb".format(ep, image_type,
                                                                                                      folder.replace(
                                                                                                          "/", "_")))

    print("Epoch: ", ep)
    print("\tTrain loss: ", train_loss_epoch / (len(trainLoader)))
    print("\tVal loss: ", val_loss_epoch/(len(valLoader)))
    wandb.log({"Train/loss":   train_loss_epoch / (len(trainLoader)),
                   "Test/loss":  val_loss_epoch/(len(valLoader))})

