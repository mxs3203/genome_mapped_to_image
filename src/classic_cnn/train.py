import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from sklearn.metrics import classification_report
# Training Params
from src.classic_cnn.Dataloader import TCGAImageLoader
from src.classic_cnn.Network_Softmax import ConvNetSoftmax

LR = 0.0001
batch_size = 100
lr_decay = 1e-5
weight_decay = 1e-5
epochs = 200
start_of_lr_decrease = 120
# Dataset Params
folder = "Metastatic_data"
image_type = "193x193Image"
predictor_column = 3
response_column = 8

writer = SummaryWriter(flush_secs=1)
transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("../../data/{}/{}/meta_data.csv".format(folder, image_type),
                          folder, image_type, predictor_column, response_column, filter_by_type=['OV', 'COAD', 'UCEC', 'KIRC','STAD', 'BLCA'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)
net = ConvNetSoftmax()
net.to(device)
cost_func = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
lambda1 = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
writer.add_text("Hyperparams",
                "LR={}, batchSize={},lr_decay={},weight_decay={}".format(LR, batch_size, lr_decay, weight_decay))
writer.add_text("Model", str(net.__dict__['_modules']))


def acc(y_hat, y):
    probs = torch.softmax(y_hat, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == y)
    accuracy = corrects.sum().float() / float(y.size(0))
    return accuracy, winners


def batch_train(x, y, auc_type=None):
    net.train()
    cost_func.zero_grad()
    y_hat = net(x)
    y_probs = net.predict(x)
    loss = cost_func(y_hat, y)
    accuracy, pred_classes = acc(y_hat, y)
    auc = roc_auc_score(y_true=y.cpu().detach(), y_score=y_probs.cpu().detach(), multi_class=auc_type)
    report = classification_report(
        digits=6,
        y_true=y.cpu().detach().numpy(),
        y_pred=pred_classes.cpu().detach().numpy(),
        output_dict=True,
        zero_division=0)
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy.item(), report['macro avg']['precision'],report['macro avg']['recall'],report['macro avg']['f1-score'], auc


def batch_valid(x, y, auc_type=None):
    with torch.no_grad():
        net.eval()
        y_hat = net(x)
        y_probs=net.predict(x)
        loss = cost_func(y_hat, y)
        accuracy, pred_classes = acc(y_hat, y)
        auc = roc_auc_score(y_true=y.cpu().detach(), y_score=y_probs.cpu().detach(), multi_class=auc_type)
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
    batch_train_loss, batch_val_f1, batch_val_loss = [], [], [],[],[],[]
    for x, type, id, y_dat in trainLoader:
        loss, acc_train, precision,recall,f1,train_auc = batch_train(x.cuda(), y_dat.cuda(), auc_type='ovr')
        batch_train_loss.append(loss)
        batch_train_f1.append(f1)
        batch_train_auc.append(train_auc)
    for x, type, id, y_dat in valLoader:
        loss, acc_val,  precision,recall,f1,val_auc = batch_valid(x.cuda(), y_dat.cuda(), auc_type='ovr')
        batch_val_loss.append(loss)
        batch_val_f1.append(f1)
        batch_val_auc.append(val_auc)
    if ep >= start_of_lr_decrease:
        scheduler.step()
    print(
        "Epoch {}: \n\tTrain loss: {} Train F1: {}, Train AUC: {} \n\tValidation loss: {} Val F1: {}, Val AUC: {} , \n\tLR : {}".format(ep,
            np.mean(batch_train_loss),np.mean(batch_train_f1),np.mean(batch_train_auc),
            np.mean( batch_val_loss),np.mean(batch_val_f1),np.mean(batch_val_auc),
            optimizer.param_groups[0]["lr"]))
    writer.add_scalar('F1/train', np.mean(batch_train_f1), ep)
    writer.add_scalar('F1/test', np.mean(batch_val_f1), ep)
    writer.add_scalar('Loss/test', np.mean(batch_val_loss), ep)
    writer.add_scalar('Loss/train', np.mean(batch_train_loss), ep)
    writer.add_scalar('AUC/test', np.mean(batch_val_auc), ep)
    writer.add_scalar('AUC/train', np.mean(batch_train_auc), ep)
    if (np.mean(batch_val_auc) >= 0.78 and np.mean(batch_val_auc) >= 0.78):
        torch.save({
            'epoch': ep,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(batch_val_loss)
        }, "checkpoints/ep_{}_model.pt".format(ep))
    writer.flush()
