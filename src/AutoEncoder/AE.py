import torch.nn as nn
import torch

from src.VAE.VAEModel import UnFlatten


class AE(nn.Module):
    def __init__(self, image_channels=5):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, 16, kernel_size=(2, 5), stride=(2, 5)), # 752, 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 8), stride=(2, 8)), # 94, 6
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2)), # 47, 3
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 4), stride=(1, 4), padding=1), # 12,3
            nn.ReLU(),
            nn.Conv2d(128, 200, kernel_size=(3, 3), stride=(3, 3)),  # 12,3
            nn.ReLU(),
            nn.Conv2d(200, 256, kernel_size=(1, 4), stride=(1, 4)),  # 4,3
            nn.ReLU(),
            UnFlatten(-1, 256, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 200, kernel_size=(1, 4), stride=(1, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(200, 128,  kernel_size=(3, 3), stride=(3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(1,4),stride=(1,4)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32,  kernel_size=(2,2),stride=(2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(2,8), stride=(2,8)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=(2,5), stride=(2,5)),
            nn.ReLU()
        )

        self.extractor = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, 64, kernel_size=(2, 5), stride=1), nn.ReLU(), nn.MaxPool2d((2, 5)),
            nn.Conv2d(64, 128, kernel_size=(2, 8), stride=1), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Dropout2d(0.05),
            nn.Conv2d(128, 128, kernel_size=(2, 4), stride=1), nn.ReLU(), nn.MaxPool2d((2, 4)),
            nn.Conv2d(128, 128, kernel_size=(2, 4), stride=1), nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.predictor = nn.Sequential(
            nn.Linear(5760, 1024), nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 2)
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
        x = self.decoder(x_enc)
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x_enc_flat], dim = 1)
        x = self.predictor(x)
        return x