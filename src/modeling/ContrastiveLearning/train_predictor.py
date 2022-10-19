#!/usr/bin python3
import json
import os
import sys

import torch
# Training Params
import wandb
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import DataLoader
from torchvision.models import vgg11, vgg19
from torchvision.transforms import transforms

#os.environ["WANDB_MODE"]="offline"
from src.modeling.ContrastiveLearning.genius_network import GENIUS
from src.modeling.Dataloader import TCGAImageLoader

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
dataset = TCGAImageLoader(config['meta_data'],
                          folder,
                          image_type,
                          predictor_column,
                          response_column)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)
# resnet50= 224
criterion = torch.nn.CrossEntropyLoss()
extractor = vgg19(pretrained=False, num_classes=5000)
checkpoint = torch.load("/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/backbone__SquareImg_TCGA_Square_ImgsGainLoss_harsh_Metastatic_data.pb")
extractor.load_state_dict(checkpoint['model_state_dict'])
for p in extractor.parameters():
           p.requires_grad = False
input_layer = torch.nn.Sequential(
    # (w-1)s - 2p + k-1 + 1
    torch.nn.ConvTranspose2d(in_channels=5, out_channels=3,padding=1, stride=1, kernel_size=33, bias=False),
    torch.nn.ReLU()
)
input_layer.load_state_dict(torch.load("/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/input__SquareImg_TCGA_Square_ImgsGainLoss_harsh_Metastatic_data.pb")['model_state_dict'])
for p in input_layer.parameters():
           p.requires_grad = False
net = GENIUS(backbone=extractor, input=input_layer)
net.to(device)

wandb.watch(net)
optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=weight_decay)


best_loss = float("+inf")
best_back_bone = None
best_input_layer = None

def acc(y_hat, y):
    probs = y_hat
    winners = probs.argmax(dim=1)
    corrects = (winners == y)
    accuracy = corrects.sum().float() / float(y.size(0))
    return accuracy, winners

for ep in range(epochs):
    net.train()
    train_loss_epoch = 0
    val_loss_epoch = 0

    for x,y_dat in trainLoader:
        optimizer.zero_grad()
        y_hat = net(x.to(device))
        loss = criterion(y_hat, y_dat.cuda())
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    net.eval()
    with torch.no_grad():
        total_auc,total_f1 = 0,0
        for x, y_dat in valLoader:
            y_hat = net(x.to(device))
            loss = criterion(y_hat, y_dat.cuda())
            accuracy, pred_classes = acc(y_hat.detach().cpu(), y_dat.detach().cpu())

            if config['trainer'] != "multi-class":
                auc = roc_auc_score(y_true=y_dat.cpu().detach(), y_score=pred_classes.cpu().detach())
                total_auc += auc
            f1 = classification_report(
                digits=6,
                y_true=y_dat.cpu().detach().numpy(),
                y_pred=pred_classes.cpu().detach().numpy(),
                output_dict=True,
                zero_division=0)['macro avg']['f1-score']
            val_loss_epoch += loss.item()
            total_f1 += f1

    if val_loss_epoch/(len(valLoader)) < best_loss :
        path_backbone = net.store_predictor(ep=ep, optimizer=optimizer,loss=val_loss_epoch/(len(valLoader)), image_type=image_type,folder=folder)
        wandb.save(path_backbone)
        best_loss = val_loss_epoch/(len(valLoader))
        print("Best loss... saving models state_dict")

    print("Epoch: ", ep)
    print("\tTrain loss: ", train_loss_epoch / (len(trainLoader)))
    print("\tVal loss: ", val_loss_epoch/(len(valLoader)))
    print("\tVal AUC and F1: ", total_auc/len(valLoader), total_f1/ len(valLoader))
    wandb.log({"Train/loss":   train_loss_epoch / (len(trainLoader)),
                "Test/loss":  val_loss_epoch/(len(valLoader)),
                "Val/AUC": total_auc/len(valLoader)
               })

