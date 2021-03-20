import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from Dataloader import TCGAImageLoader
from Network_Softmax import ConvNetSoftmax

writer = SummaryWriter(flush_secs=1)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.1,0.1,0.1], [0.1,0.1,0.1])])
dataset = TCGAImageLoader("../data/DSS_labels.csv", "../data/", transform)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=300, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=100, num_workers=10, shuffle=True)
net = ConvNetSoftmax()
net.to(device)
cost_func = torch.nn.NLLLoss()
LR = 0.0001
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=1e-2, lr=LR, weight_decay=1e-3)

epochs = 500

def adjust_learning_rate(optimizer, epoch,curr_lr, every_n_ep = 5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = curr_lr * (0.1 ** (epoch // every_n_ep))
    print("Setting LR from ", curr_lr ," to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def plot_activations(A, number_rows=1, name="", i=0):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()
    fig, axes = plt.subplots(number_rows, 1)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Set the label for the sub-plot.
            ax.set_xlabel("activation:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

def acc(y_hat, y):
    probs = torch.softmax(y_hat, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == y)
    accuracy = corrects.sum().float() / float(y.size(0))
    return accuracy,winners

def makeImages(x):
    img_cin_l = x[0, 0, :, :]*255.0
    img_cin_l = ((img_cin_l - img_cin_l.min()) * (1/(img_cin_l.max()+1e-3 - img_cin_l.min()) * 255)).astype('uint8')
    img_cin_g = x[0, 1, :, :]*255.0
    img_cin_g = ((img_cin_g - img_cin_g.min()) * (1/(img_cin_g.max()+1e-3 - img_cin_g.min()) * 255)).astype('uint8')
    img_mut = x[0, 2, :, :] * 255.0
    img_mut = ((img_mut - img_mut.min()) * (1 / (img_mut.max()+1e-3 - img_mut.min()) * 255)).astype('uint8')
    img_mut = Image.fromarray(img_mut, 'P')
    img_cin_g = Image.fromarray(img_cin_g, 'P')
    img_cin_l = Image.fromarray(img_cin_l, 'P')
    total_img = np.dstack((img_cin_l, img_cin_g, img_mut))
    total_img = Image.fromarray(total_img, 'RGB')
    img_cin_g.save("cin_gain.png")
    img_cin_l.save("cin_loss.png")
    img_mut.save("mut.png")
    total_img.save("total.png")
    return img_cin_l, img_cin_g, img_mut,total_img

def batch_train(x, y):
    net.train()
    cost_func.zero_grad()
    y_hat = net(x)
    loss = cost_func(y_hat, y)
    accuracy,pred_classes = acc(y_hat, y)
    auc = roc_auc_score(y.cpu().detach().numpy(), pred_classes.cpu().detach().numpy())
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy.item(), auc

def batch_valid(x,y):
    with torch.no_grad():
        net.eval()
        y_hat = net(x)
        loss = cost_func(y_hat, y)
        accuracy, pred_classes = acc(y_hat, y)
        auc = roc_auc_score(y.cpu().detach().numpy(), pred_classes.cpu().detach().numpy())
        img_cin_l,img_cin_g, img_mut,total_img = makeImages(x=x.cpu().detach().numpy())
        return loss.item(), accuracy.item(), auc

train_losses = []
val_losses = []
for ep in range(epochs):
    batch_train_auc ,batch_train_loss, batch_val_auc, batch_val_acc, batch_train_acc, batch_val_loss= [],[],[],[],[],[]
    for x,y,type,id in trainLoader:
        loss, acc_train, auc = batch_train(x.cuda(), y.cuda())
        batch_train_loss.append(loss)
        batch_train_acc.append(acc_train)
        batch_train_auc.append(auc)
    for x,y,type,id  in valLoader:
        loss, acc_val, auc = batch_valid(x.cuda(), y.cuda())
        batch_val_loss.append(loss)
        batch_val_acc.append(acc_val)
        batch_val_auc.append(auc)

    print("Epoch {}: Train loss: {} Train acc: {} Train AUC: {}, Validation loss: {} validation acc: {} Val AUC: {}".format(ep, np.mean(batch_train_loss),np.mean(batch_train_acc), np.mean(batch_train_auc), np.mean(batch_val_loss), np.mean(batch_val_acc), np.mean(batch_val_auc)))
    train_losses.append(np.mean(batch_train_loss))
    val_losses.append(np.mean(batch_val_loss))
    writer.add_scalar('Loss/test',np.mean(batch_val_loss) , ep)
    writer.add_scalar('Loss/train', np.mean(batch_train_loss), ep)
    writer.add_scalar('Accuracy/train', np.mean(batch_train_acc), ep)
    writer.add_scalar('Accuracy/test', np.mean(batch_val_acc), ep)
    writer.add_scalar('AUC/train', np.mean(batch_train_auc), ep)
    writer.add_scalar('AUC/test', np.mean(batch_val_auc), ep)
    if(np.mean(batch_val_auc) >= 0.7 and np.mean(batch_train_auc) >= 0.7):
        torch.save({
            'epoch': ep,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(batch_val_loss)
        }, "checkpoints/ep_{}_model.pt".format(ep))