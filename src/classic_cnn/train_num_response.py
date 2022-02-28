import numpy as np
import torch
import wandb
from sklearn.metrics import  mean_squared_error
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.AutoEncoder.AE import AE
from src.FlattenFeatures.Network_Softmax_Flatten import NetSoftmax
from src.classic_cnn.Dataloader import TCGAImageLoader

LR = 1e-6
batch_size = 64
lr_decay = 1e-3
weight_decay = 1e-3
epochs = 200
start_of_lr_decrease = 20
# Dataset Params
folder = "Metastatic_data"
image_type = "ChrImg"
predictor_column = 3 # 3=n_dim_img,4=flatten
response_column = 10  # 5=met,6=wgii,7=tp53,8=type, 9 = age

wandb.init(project="genome_as_image", entity="mxs3203", name="Age_{}".format(image_type),reinit=True)
wandb.config = {
    "learning_rate": LR,
    "epochs": epochs,
    "batch_size": batch_size,
    "folder": folder,
    "image_type": image_type,
    "weight_decay": weight_decay,
    "lr_decay": lr_decay
}

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("/media/mateo/data1/genome_mapped_to_image/data/main_meta_data.csv",
                          folder,
                          image_type,
                          predictor_column,
                          response_column,
                          filter_by_type=['OV', 'COAD', 'UCEC', 'KIRC','STAD', 'BLCA'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)
net = AE()
net.to(device)
cost_func = torch.nn.MSELoss()

wandb.watch(net)
wandb.save("/media/mateo/data1/genome_mapped_to_image/src/AutoEncoder/AE_Squere.py")
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
lambda1 = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

best_loss = 12345678909

def batch_train(x, y):
    net.train()
    cost_func.zero_grad()
    y_hat = net(x)
    loss = cost_func(y_hat.squeeze(), y.float())
    mse = mean_squared_error(y_true=y.cpu().detach(), y_pred=y_hat.cpu().detach())

    loss.backward()
    optimizer.step()
    return loss.item(), mse, np.sqrt(mse)


def batch_valid(x, y):
    with torch.no_grad():
        net.eval()
        y_hat = net(x)
        loss = cost_func(y_hat.squeeze(), y.float())
        mse = mean_squared_error(y_true=y.cpu().detach(), y_pred=y_hat.cpu().detach())
        return loss.item(), mse, np.sqrt(mse)


train_losses = []
val_losses = []
for ep in range(epochs):
    batch_train_mse, batch_val_mse,batch_train_loss,  batch_val_loss , batch_val_rmse, batch_train_rmse = [],[],[],[],[],[]
    for x,y_dat, id  in trainLoader:
        loss,mse,rmse = batch_train(x.cuda(), y_dat.cuda())
        batch_train_loss.append(loss)
        batch_train_mse.append(mse)
        batch_train_rmse.append(rmse)
    for x,y_dat, id  in valLoader:
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
    if ( np.mean(batch_train_mse) <= 0.1 and np.mean(batch_val_mse) <= 0.1):
        if np.mean(batch_val_loss) < best_loss:
            best_loss = np.mean(batch_val_loss)
            torch.save({
                'epoch': ep,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(batch_val_loss)
            }, "checkpoints/Age_{}_{}.pb".format(image_type, folder))
            wandb.save("checkpoints/Age_{}_{}.pb".format(image_type, folder))
