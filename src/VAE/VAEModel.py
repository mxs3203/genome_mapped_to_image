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
    def __init__(self, image_channels=5, h_dim=512, z_dim=1024):
        super(ConvVAE, self).__init__()
        self.z_size = z_dim
        self.encoder = nn.Sequential(
            # 512/4=128, (B, 64, 128, 128)
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            # 128/4=32 (B, 80, 32, 32)
            nn.Conv2d(64, 80, kernel_size=4,stride=4),
            nn.ReLU(),
            # 32/4=8 (B, 100, 8, 8)
            nn.Conv2d(80, 100, kernel_size=4,stride=4),
            nn.ReLU(),
            # 8/8=1 (B, 120, 1, 1)
            nn.Conv2d(100, h_dim, kernel_size=8,stride=8),
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
            nn.ConvTranspose2d(h_dim, 100, kernel_size=8,stride=8),
            nn.ReLU(),
            nn.ConvTranspose2d(100, 80, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(80, 64, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=4),
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
