import numpy as np
import torch
from torchsummary import summary
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from TCGA_GenomeImage.tidy_version.v2.Dataloader import TCGAImageLoader
from TCGA_GenomeImage.tidy_version.v2.Network_Softmax import ConvNetSoftmax

LR = 0.0001
batch_size = 200
lr_decay = 1e-2
weight_decay = 1e-3

writer = SummaryWriter(flush_secs=1)
transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader("../../data/v2/DSS_labels_with_extrastuff.csv", "../../data/v2/", transform=transform)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_size = int(len(dataset) * 0.8)
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
writer.add_text("Hyperparams","LR={}, batchSize={},lr_decay={},weight_decay={}".format(LR, batch_size, lr_decay, weight_decay))
writer.add_text("Model", str(net.__dict__['_modules']))
epochs = 500

def acc(y_hat, y):
    probs = torch.softmax(y_hat, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == y)
    accuracy = corrects.sum().float() / float(y.size(0))
    return accuracy, winners


def makeImages(x):
    img_cin_g = x[0, 0, :, :]
    img_cin_g = img_cin_g.astype('uint8')
    img_cin_l = x[0, 1, :, :]
    img_cin_l = img_cin_l.astype('uint8')
    img_mut = x[0, 2, :, :]
    img_mut = img_mut.astype('uint8')
    img_mut = Image.fromarray(img_mut, 'P')
    img_cin_g = Image.fromarray(img_cin_g, 'P')
    img_cin_l = Image.fromarray(img_cin_l, 'P')
    total_img = np.dstack((img_cin_g, img_cin_l, img_mut))
    total_img = Image.fromarray(total_img, 'RGB')
    img_cin_g.save("cin_gain.png")
    img_cin_l.save("cin_loss.png")
    img_mut.save("mut.png")
    total_img.save("total.png")
    return img_cin_l, img_cin_g, img_mut


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
        img_cin_l, img_cin_g, img_mut = makeImages(x=x.cpu().detach().numpy())
        return loss.item(), accuracy.item(), auc


train_losses = []
val_losses = []
for ep in range(epochs):
    batch_train_auc, batch_train_loss, batch_val_auc, batch_val_loss = [], [], [], []
    for x, dss, type, id, met_1_2, met_1_2_3, GD in trainLoader:
        loss, acc_train, auc = batch_train(x.cuda(), met_1_2_3.cuda())
        batch_train_loss.append(loss)
        batch_train_auc.append(auc)
    for x, dss, type, id, met_1_2, met_1_2_3, GD in valLoader:
        loss, acc_val, auc = batch_valid(x.cuda(), met_1_2_3.cuda())
        batch_val_loss.append(loss)
        batch_val_auc.append(auc)


    print(
        "Epoch {}: Train loss: {} Train AUC: {}, Validation loss: {} Val AUC: {}".format(ep, np.mean(batch_train_loss),
                                                                                         np.mean(batch_train_auc),
                                                                                         np.mean(batch_val_loss),
                                                                                         np.mean(batch_val_auc)))
    writer.add_scalar('Loss/test', np.mean(batch_val_loss), ep)
    writer.add_scalar('Loss/train', np.mean(batch_train_loss), ep)
    writer.add_scalar('AUC/train', np.mean(batch_train_auc), ep)
    writer.add_scalar('AUC/test', np.mean(batch_val_auc), ep)
    for name, weight in net.named_parameters():
        writer.add_histogram(name, weight, ep)
        writer.add_histogram(f'{name}.grad', weight.grad, ep)
    if (np.mean(batch_val_auc) >= 0.7 and np.mean(batch_train_auc) >= 0.7):
        torch.save({
            'epoch': ep,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(batch_val_loss)
        }, "checkpoints/ep_{}_model.pt".format(ep))
    writer.flush()