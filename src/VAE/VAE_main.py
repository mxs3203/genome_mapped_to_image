import pandas
import torch
import numpy as np
import torch.optim as optim
import matplotlib
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Pad

from src.VAE.VAEModel import ConvVAE
from src.classic_cnn.Dataloader import TCGAImageLoader
from torch.nn.functional import pad
import wandb



lr = 0.00001
batch_size = 600
lr_decay = 1e-5
weight_decay = 1e-5
epochs = 1000
start_of_lr_decrease = 600
# Dataset Params
folder = "Metastatic_data"
image_type = "SquereImg"
predictor_column = 3
response_column = 7
torch.multiprocessing.set_start_method('spawn', force=True)

wandb.init(project="genome_as_image", entity="mxs3203", name="VAE_{}-{}".format(image_type,folder),reinit=True)
wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size,
    "folder": folder,
    "image_type": image_type,
    "weight_decay": weight_decay,
    "lr_decay": lr_decay
}


matplotlib.style.use('ggplot')

tsne = TSNE(n_components=2, verbose=0, random_state=123)

my_transforms = transforms.Compose([
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set the learning parameters
L = 1024
H = 1024
model = ConvVAE(z_dim=L, h_dim=H).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss(reduction='sum')

dataset = TCGAImageLoader("/media/mateo/data1/genome_mapped_to_image/data/{}/{}/meta_data.csv".format(folder, image_type),
                          folder, image_type, predictor_column, response_column, filter_by_type=['OV', 'COAD', 'UCEC', 'KIRC','STAD', 'BLCA'])
torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=True)


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2) / depth
    return torch.exp(-numerator)


def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()


def loss_function(pred, true, latent):
    return (pred - true).pow(2).mean(), MMD(torch.randn(512, L, requires_grad=False).to(device), latent)


def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, KLD


def train(ep):
    model.train()
    running_loss = []
    running_kld = []
    for images, type, id, y_dat in trainLoader:
        images = images.to(device)
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        # loss, kld = loss_function(recon_images, images, mu)
        # loss = loss + kld
        loss = criterion(recon_images, images)
        loss,kld = final_loss(loss,mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.cpu().detach())
        running_kld.append(kld.cpu().detach())

    to_print = "Epoch[{}/{}] \n\tTraining Loss: {:.8f} KLD: {:.8f}".format(epoch + 1,
                                                                           epochs,
                                                                           np.mean(running_loss),
                                                                           np.mean(running_kld))
    print(to_print)

    return np.mean(running_loss), np.mean(running_kld)


def validate(ep):
    # Validation
    running_loss = []
    running_kld = []
    with torch.no_grad():
        model.eval()
        for images, type, id, y_dat in valLoader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            # loss, kld = loss_function(recon_images, images, mu)
            # loss = loss + kld
            loss = criterion(recon_images, images)
            loss, kld = final_loss(loss, mu, logvar)

            running_loss.append(loss.cpu().detach())
            running_kld.append(kld.cpu().detach())
        to_print = "\tValidation Loss: {:.8f} KLD: {:.8f}".format(
            np.mean(running_loss),
            np.mean(running_kld))
        print(to_print)

    return recon_images[0, :, :, :], np.mean(running_loss), np.mean(running_kld)


def plot_latent(ep):
    print("\tCalculating TSNE of validation data latent vectors")
    total = pandas.DataFrame()
    Y = []
    for x, type, id, y_dat in valLoader:
        x, z, z_mean, z_log_var = model.encode_img(x.to(device))
        z = z.to('cpu').detach().numpy()
        Y = np.append(Y, y_dat.to('cpu').detach().numpy())
        z = pandas.DataFrame(z)
        total = pandas.concat([total, z], ignore_index=True)

    tsne_results = tsne.fit_transform(total)
    tsne_results = pandas.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    tsne_results['y'] = Y
    cdict = {0: 'noTP53', 1: 'TP53'}
    plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=tsne_results['y'])
    plt.legend(loc='lower left', title='TP53')
    plt.title("TSNE of Validation DF latent vectors")
    wandb.log({"tsne": plt})
    # writer.add_figure('Valid/tsne', plt, ep)
    #plt.savefig("Res/tsne{}.png".format(ep))
    plt.close()
    #plt.show()


def visualize_recon(randomimg, ep, channels=5):
    num_row = 2
    num_col = 3 # plot images
    fig, axes = plt.subplots(num_row, num_col)
    for i in range(channels):
        ax = axes[i // num_col, i % num_col]
        im = np.array(randomimg[i, :, :])
        ax.imshow(im, cmap='gray')
    plt.tight_layout()
    # writer.add_figure('Valid/reconstruction', plt.show(), ep)
    #plt.savefig("Res/reconstructions{}.png".format(ep))
    wandb.log({"reconstruction": plt})
    plt.close()
    #plt.show()


loss = []
for epoch in range(epochs):
    train_loss, train_kld = train(epoch)
    loss.append(train_loss)
    recon_image, val_loss, val_kld = validate(epoch)
    if epoch % 10 == 1:
        visualize_recon(recon_image.cpu().detach(), epoch)
        plot_latent(epoch)
        #torch.save(model.state_dict(), "../saved_models/vae_model_ep_{}.pt".format(epoch))
    wandb.log({"Train/loss":train_loss,
               "Train/KLD": train_kld,
               "Test/loss":val_loss,
               "Test/KLD": val_kld})
torch.save(model.state_dict(), "../saved_models/vae_model_last.pt")
plt.plot(range(epochs), loss)
plt.title("Total Loss")
plt.show()

print('TRAINING COMPLETE')
