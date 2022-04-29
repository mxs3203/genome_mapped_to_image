import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import classification_report

# Training Params
import wandb
import json

from Dataloader import TCGAImageLoader
from train_util import return_model_and_cost_func

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

wandb.init(project="Test", entity="mxs3203", name="{}_{}-{}".format(config['run_name'],image_type,folder),reinit=True)
wandb.save(config_path)

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/meta_data_new.csv",
                          folder,
                          image_type,
                          predictor_column,
                          response_column)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)


net, cost_func = return_model_and_cost_func(config, dataset)
net.to(device)


wandb.watch(net)
wandb.save("/src/AutoEncoder/AE_Square.py") #"AutoEncoder/AE.py")
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
lambda1 = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

best_loss = float("+Inf")

def acc(y_hat, y):
    probs = torch.softmax(y_hat, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == y)
    accuracy = corrects.sum().float() / float(y.size(0))
    return accuracy, winners

def batch_train(x, y):
    net.train()
    cost_func.zero_grad()
    y_hat = net(x)
    y_probs = net.predict(x)
    loss = cost_func(y_hat, y)
    accuracy, pred_classes = acc(y_hat, y)
    auc = 0
    auc = roc_auc_score(y_true=y.cpu().detach(), y_score=pred_classes.cpu().detach())
    report = classification_report(
        digits=6,
        y_true=y.cpu().detach().numpy(),
        y_pred=pred_classes.cpu().detach().numpy(),
        output_dict=True,
        zero_division=0)
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy.item(), report['macro avg']['precision'],report['macro avg']['recall'],report['macro avg']['f1-score'], auc

def batch_valid(x, y):
    with torch.no_grad():
        net.eval()
        y_hat = net(x)
        y_probs = net.predict(x)
        loss = cost_func(y_hat, y)
        accuracy, pred_classes = acc(y_hat, y)
        auc = 0
        auc = roc_auc_score(y_true=y.cpu().detach(), y_score=pred_classes.cpu().detach())
        report = classification_report(
            digits=6,
            y_true=y.cpu().detach().numpy(),
            y_pred=pred_classes.cpu().detach().numpy(),
            output_dict=True,
            zero_division=0)

        return loss.item(), accuracy.item(), report['macro avg']['precision'],report['macro avg']['recall'],report['macro avg']['f1-score'], auc


train_losses = []
val_losses = []
for ep in range(epochs):
    batch_train_f1,batch_val_auc, batch_train_auc,\
    batch_train_loss, batch_val_f1, batch_val_loss = [],[],[],[],[],[]
    for x, y_dat,id in trainLoader:
        loss, acc_train, precision,recall,f1,train_auc = batch_train(x.cuda(), y_dat.cuda())
        batch_train_loss.append(loss)
        batch_train_f1.append(f1)
        batch_train_auc.append(train_auc)

    for x, y_dat,id in valLoader:
        loss, acc_val,  precision,recall,f1,val_auc = batch_valid(x.cuda(), y_dat.cuda())
        batch_val_loss.append(loss)
        batch_val_f1.append(f1)
        batch_val_auc.append(val_auc)
    if ep >= start_of_lr_decrease:
        scheduler.step()
    print(
        "Epoch {}: \n\tTrain loss: {} Train F1: {}, Train AUC: {} \n\tValidation loss: {} Val F1: {}, Val AUC: {} , \n\tLR : {}".format(ep,
            np.mean(batch_train_loss),np.mean(batch_train_f1),np.mean(batch_train_auc),
            np.mean( batch_val_loss),np.mean(batch_val_f1),np.mean(batch_val_auc),
            optimizer.param_groups[0]["lr"])
    )
    if np.mean(batch_val_loss) < best_loss:
        best_loss = np.mean(batch_val_loss)
        print("Best loss! ")
    wandb.log({"Train/loss":  np.mean(batch_train_loss),
               "Train/AUC": np.mean(train_auc),
               "Test/loss": np.mean(batch_val_loss),
               "Test/AUC": np.mean(batch_val_auc)})

    if (np.mean(batch_train_auc) >= config['save_model_score'] and np.mean(batch_val_auc) >= config['save_model_score']):
            torch.save({
                'epoch': ep,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(batch_val_loss)
            }, "checkpoints/{}_{}_{}.pb".format(ep,image_type,folder))
            wandb.save("checkpoints/{}_{}_{}.pb".format(ep, image_type,folder))
