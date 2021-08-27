import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from TCGA_GenomeImage.src.FlattenFeatures.Dataloader_flatten import TCGAImageLoader
from TCGA_GenomeImage.src.FlattenFeatures.Network_Softmax_Flatten import NetSoftmax

LR = 0.0001
batch_size = 100
lr_decay = 1e-5
weight_decay = 1e-5

writer = SummaryWriter(flush_secs=1)
transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("../../data/meta_data.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)
net = NetSoftmax()
net.to(device)
cost_func = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
lambda1 = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
writer.add_text("Hyperparams",
                "LR={}, batchSize={},lr_decay={},weight_decay={}".format(LR, batch_size, lr_decay, weight_decay))
writer.add_text("Model", str(net.__dict__['_modules']))
epochs = 500


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
    loss = cost_func(y_hat, y)
    accuracy, pred_classes = acc(y_hat, y)
    auc = roc_auc_score(y.cpu().detach().numpy(), pred_classes.cpu().detach().numpy())
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy.item(), auc


def batch_valid(x, y):
    with torch.no_grad():
        net.eval()
        y_hat = net(x)
        loss = cost_func(y_hat, y)
        accuracy, pred_classes = acc(y_hat, y)
        auc = roc_auc_score(y.cpu().detach().numpy(), pred_classes.cpu().detach().numpy())
        return loss.item(), accuracy.item(), auc


train_losses = []
val_losses = []
for ep in range(epochs):
    batch_train_auc, batch_train_loss, batch_val_auc, batch_val_loss = [], [], [], []
    for x, type, id, met_1_2_3 in trainLoader:
        loss, acc_train, auc = batch_train(x.cuda(), met_1_2_3.cuda())
        batch_train_loss.append(loss)
        batch_train_auc.append(auc)
    for x, type, id, met_1_2_3 in valLoader:
        loss, acc_val, auc = batch_valid(x.cuda(), met_1_2_3.cuda())
        batch_val_loss.append(loss)
        batch_val_auc.append(auc)
    if ep >= 90:
        scheduler.step()
    print(
        "Epoch {}: Train loss: {} Train AUC: {}, Validation loss: {} Val AUC: {}, LR : {}".format(ep, np.mean(batch_train_loss),
                                                                                                np.mean(batch_train_auc),
                                                                                         np.mean(batch_val_loss),
                                                                                         np.mean(batch_val_auc), optimizer.param_groups[0]["lr"]))
    writer.add_scalar('Loss/test', np.mean(batch_val_loss), ep)
    writer.add_scalar('Loss/train', np.mean(batch_train_loss), ep)
    writer.add_scalar('AUC/train', np.mean(batch_train_auc), ep)
    writer.add_scalar('AUC/test', np.mean(batch_val_auc), ep)
    if (np.mean(batch_val_auc) >= 0.7 and np.mean(batch_train_auc) >= 0.7):
        torch.save({
            'epoch': ep,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(batch_val_loss)
        }, "checkpoints/ep_{}_model.pt".format(ep))
    writer.flush()
