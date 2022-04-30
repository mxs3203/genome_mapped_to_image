import torch.nn as nn
import torch

from src.AutoEncoder.AE_util import UnFlatten


class AE(nn.Module):
    def __init__(self, output_size, image_channels=5):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, 8, kernel_size=(2, 5), stride=(2, 5)), # 751, 11
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(1, 7), stride=(2, 8)), # 93, 5
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(2, 5)), # 18, 2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 5), padding=(1, 0)), # 3,3
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(3, 3)),  # 1,1
            nn.ReLU(),
            UnFlatten(-1, 128, 1, 1),
        )
        self.decoder = nn.Sequential( # output=(input-1)*stride - 2*padding + (kernel-1) + 1
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=1), # 3, 3
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32,  kernel_size=(2, 8), stride=(1, 5), padding=(1,0)), # 18, 2
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,8),stride=(2,5)), # 93, 5
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8,  kernel_size=(3,15),stride=(2, 8)), # 751, 11
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_channels, kernel_size=(4,10), stride=(2, 5)), #3760,24
            nn.ReLU(),
        )

        self.extractor = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, 32, kernel_size=(3, 3), stride=1), nn.ReLU(), nn.MaxPool2d((2, 3)),
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=1), nn.ReLU(), nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.05),
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=1), nn.ReLU(), nn.MaxPool2d((2, 3)),
            nn.Conv2d(32, 32, kernel_size=(1, 4), stride=1), nn.ReLU(), nn.MaxPool2d((1, 2)),
            nn.BatchNorm2d(32)
        )
        self.predictor = nn.Sequential(
            nn.Linear(4288+128, 1024), nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 512), nn.ReLU(),
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
        return x,x_dec