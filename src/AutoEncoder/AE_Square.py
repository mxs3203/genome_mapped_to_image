import torch.nn as nn
import torch

from src.AutoEncoder.AE_util import UnFlatten


class AE(nn.Module):
    def __init__(self, output_size, image_channels=5):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, 16, kernel_size=2, stride=2, padding=3), # 512/4=128, (B, 64, 128, 128)
            nn.LeakyReLU(1e-6),
            nn.Conv2d(16, 32, kernel_size=4, stride=4), # 128/4=32 (B, 80, 32, 32)
            nn.LeakyReLU(1e-6),
            nn.Conv2d(32, 64, kernel_size=5,stride=5), #32/4=8 (B, 100, 8, 8)
            nn.LeakyReLU(1e-6),
            nn.Conv2d(64, 128, kernel_size=5,stride=5), #8/8=1 (B, 120, 1, 1)
            nn.LeakyReLU(1e-6),
            UnFlatten(-1, 128, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5,stride=5),
            nn.LeakyReLU(1e-6),
            nn.ConvTranspose2d(64, 32, kernel_size=5,stride=5),
            nn.LeakyReLU(1e-6),
            nn.ConvTranspose2d(32, 16, kernel_size=4,stride=4),
            nn.LeakyReLU(1e-6),
            nn.ConvTranspose2d(16, image_channels, kernel_size=2, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(1e-6),
        )

        self.extractor = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, 64, kernel_size=(3, 3), stride=1, padding=1), nn.LeakyReLU(1e-6), nn.MaxPool2d((3,3)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1), nn.LeakyReLU(1e-6), nn.MaxPool2d((3, 3)),
            nn.Dropout2d(0.35),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1),nn.LeakyReLU(1e-6), nn.MaxPool2d((3, 3)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1), nn.LeakyReLU(1e-6),
            nn.BatchNorm2d(128),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2176, 1024),nn.LeakyReLU(1e-6),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),nn.LeakyReLU(1e-6),
            nn.Linear(512,output_size)
        )

    def predict(self, x):
        x = self.forward(x)
        return x

    def encode(self, x):
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        x_dec = self.decoder(x_enc)
        x = self.extractor(x_dec)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x_enc_flat], dim = 1)
        x = self.predictor(x)
        return x, x_dec