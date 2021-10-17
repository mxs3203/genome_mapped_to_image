import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from sklearn.metrics import classification_report
# Training Params
from src.classic_cnn.Dataloader import TCGAImageLoader
from src.classic_cnn.Network_Softmax import ConvNetSoftmax

LR = 0.0001
batch_size = 200
lr_decay = 1e-5
weight_decay = 1e-5
epochs = 400
start_of_lr_decrease = 200
# Dataset Params
folder = "Metastatic_data"
image_type = "193x193Image"
predictor_column = 3 # ID
response_column = 8

writer = SummaryWriter(flush_secs=1)
transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("../../data/{}/{}/meta_data_wgii.csv".format(folder, image_type),
                          folder, image_type, predictor_column, response_column)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)
net = ConvNetSoftmax()
net.to(device)
cost_func = torch.nn.MSELoss()

optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
lambda1 = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
writer.add_text("Hyperparams",
                "LR={}, batchSize={},lr_decay={},weight_decay={}".format(LR, batch_size, lr_decay, weight_decay))
writer.add_text("Model", str(net.__dict__['_modules']))


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
    for x, type, id, y_dat in trainLoader:
        loss,mse,rmse = batch_train(x.cuda(), y_dat.cuda())
        batch_train_loss.append(loss)
        batch_train_mse.append(mse)
        batch_train_rmse.append(rmse)
    for x, type, id, y_dat in valLoader:
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
    writer.add_scalar('Loss/test', np.mean(batch_val_loss), ep)
    writer.add_scalar('Loss/train', np.mean(batch_train_loss), ep)
    writer.add_scalar('MSE/test', np.mean(batch_val_mse), ep)
    writer.add_scalar('MSE/train', np.mean(batch_train_mse), ep)
    if (np.mean(batch_train_mse) <= 0.038 and np.mean(batch_val_mse) <= 0.038):
        torch.save({
            'epoch': ep,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(batch_val_loss)
        }, "checkpoints/wgii_ep_{}_model.pt".format(ep))
    writer.flush()
