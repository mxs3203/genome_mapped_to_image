#!/usr/bin python3

import json
import sys

import numpy as np
import torch
import wandb
from sklearn.metrics import  mean_squared_error
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.modeling.Dataloader import TCGAImageLoader
from src.modeling.train_util import return_model_and_cost_func_numeric

sys.argv.append("config/wGII_square")
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
wandb.init(project="Genome_As_Image_v3", entity="mxs3203", name="{}_{}".format(config['run_name'],folder),reinit=True)
wandb.save(config_path)

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader(config['meta_data'],
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

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


net, cost_func = return_model_and_cost_func_numeric(config)
cost_func_reconstruct = torch.nn.MSELoss()
net.to(device)

wandb.watch(net)
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
lambda1 = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

best_loss = float("+Inf")
best_model = None
trigger_times = 0
last_loss = 0
patience = config['early_stop_patience']


def batch_train(x, y):
    net.train()
    cost_func.zero_grad()
    cost_func_reconstruct.zero_grad()
    y_hat, reconstructed_img = net(x)
    loss = cost_func(y_hat.squeeze(), y.float())
    total_loss = loss
    mse = mean_squared_error(y_true=y.cpu().detach(), y_pred=y_hat.cpu().detach())

    total_loss.backward()
    optimizer.step()
    return total_loss.item(), mse, np.sqrt(mse)

def saveModel(ep, optimizer, loss):
    torch.save({
        'epoch': ep,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, "/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/{}_{}_{}.pb".format(ep, image_type,
                                                                                                 folder.replace("/",
                                                                                                                "_")))
    wandb.save("/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/{}_{}_{}.pb".format(ep, image_type,
                                                                                                         folder.replace(
                                                                                                             "/", "_")))


def batch_valid(x, y):
    with torch.no_grad():
        net.eval()
        y_hat,reconstructed_img = net(x)
        loss = cost_func(y_hat.squeeze(), y.float())
        total_loss = loss
        mse = mean_squared_error(y_true=y.cpu().detach(), y_pred=y_hat.cpu().detach())
        return total_loss.item(), mse, np.sqrt(mse)


train_losses = []
val_losses = []
for ep in range(epochs):
    batch_train_mse, batch_val_mse,batch_train_loss,  batch_val_loss , batch_val_rmse, batch_train_rmse = [],[],[],[],[],[]
    for x,y_dat, id,type  in trainLoader:
        loss,mse,rmse = batch_train(x.cuda(), y_dat.cuda())
        batch_train_loss.append(loss)
        batch_train_mse.append(mse)
        batch_train_rmse.append(rmse)
    for x,y_dat, id ,type in valLoader:
        loss,mse,rmse = batch_valid(x.cuda(), y_dat.cuda())
        batch_val_loss.append(loss)
        batch_val_mse.append(mse)
        batch_val_rmse.append(rmse)
    if ep >= start_of_lr_decrease:
        scheduler.step()
    print(
        "Epoch {}: \n\tTrain loss: {} Train MSE: {}, Train RMSE: {} \n\tValidation loss: {} Val MSE: {}, Val RMSE: {} \n\tLR : {}".format(ep,
            np.mean(batch_train_loss),np.mean(batch_train_mse),np.mean(batch_train_rmse),
            np.mean( batch_val_loss),np.mean(batch_val_mse), np.mean(batch_val_rmse),
            optimizer.param_groups[0]["lr"]))
    wandb.log({"Train/loss": np.mean(batch_train_rmse),
               "Train/RMSE": np.mean(batch_train_mse),
               "Train/MSE": np.mean(batch_train_mse),
               "Test/loss": np.mean(batch_val_loss),
               "Test/RMSE":np.mean(batch_val_rmse),
               "Test/MSE": np.mean(batch_val_mse),
               }
              )
    if np.mean(batch_val_loss) < best_loss:
        best_loss = np.mean(batch_val_loss)
        best_model = net
        print("Best loss! ")
    if (np.mean(batch_train_rmse) <= config['save_model_score'] and np.mean(batch_val_rmse) <= config['save_model_score']):
        saveModel(ep, optimizer, np.mean(batch_val_loss))
    if np.mean(batch_val_loss) > last_loss:
        trigger_times += 1
        print('Trigger Times:', trigger_times)
        if trigger_times >= patience:
            print("Early Stopping!")
            saveModel(ep, optimizer, np.mean(batch_val_loss))
            break
    else:
        print('trigger times: 0')
        trigger_times = 0
    last_loss = np.mean(batch_val_loss)