import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UnFlatten(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)



class ConvVAE(nn.Module):
    def __init__(self, h_dim, z_dim,  image_channels=5):
        super(ConvVAE, self).__init__()
        self.z_size = z_dim
        self.encoder = nn.Sequential(
            #  (B, 8, 197+3, 197+3)
            nn.Conv2d(image_channels, 16, kernel_size=2, stride=2, padding=3),
            nn.ReLU(),
            #  (B, 8, 100, 100)
            nn.Conv2d(16, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            #  (B, 32, 25, 25)
            nn.Conv2d(32, 64, kernel_size=5,stride=5),
            nn.ReLU(),
            #  (B, 64, 5, 5)
            nn.Conv2d(64, h_dim, kernel_size=5,stride=5),
            nn.ReLU(),
            #nn.Dropout2d(0.1),
            nn.Flatten(),
            nn.Linear(h_dim, h_dim), nn.ReLU()
        )

        self.z_mean = nn.Linear(h_dim, z_dim)
        self.z_log_var = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(),
            UnFlatten(-1, h_dim, 1, 1),
            nn.ConvTranspose2d(h_dim, 64, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5,stride=5),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=2, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        z_std = torch.exp(logvar/2.0)
        q = torch.distributions.Normal(mu, z_std)
        z = q.rsample()
        return z

    def encode_img(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        return x,z, z_mean, z_log_var

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z), z_mean, z_log_var
