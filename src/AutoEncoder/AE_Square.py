import torch.nn as nn
import torch

from src.AutoEncoder.AE_util import UnFlatten


class AE(nn.Module):
    def __init__(self, output_size, image_channels=5):
        super(AE, self).__init__()
        base_scale_channel = 32
        img_size = 193

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, base_scale_channel, kernel_size=(3,2), stride=3,padding=1),  # 65
            nn.AdaptiveMaxPool2d((65,65)),
            nn.ReLU(),
            nn.Conv2d(base_scale_channel, base_scale_channel*2, kernel_size=(3,2), stride=3),  # 32
            nn.AdaptiveMaxPool2d((32, 32)),
            nn.ReLU(),
            nn.Conv2d(base_scale_channel*2, base_scale_channel*4, kernel_size=(4,2), stride=4),  # 8
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_scale_channel*4, base_scale_channel*2, kernel_size=5, stride=5), # 5
            nn.ReLU(),
            nn.ConvTranspose2d(base_scale_channel*2, base_scale_channel, kernel_size=5, stride=5), # 25
            nn.ReLU(),
            nn.ConvTranspose2d(base_scale_channel, base_scale_channel, kernel_size=4, stride=4), # 100
            nn.ReLU(),
            nn.ConvTranspose2d(base_scale_channel, image_channels, kernel_size=2, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(image_channels),
        )

        extractor_channels = 8
        self.extractor = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, extractor_channels, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(), nn.MaxPool2d((3, 3)),
            nn.Conv2d(extractor_channels, extractor_channels, kernel_size=(3, 3), stride=1), nn.ReLU(), nn.MaxPool2d((3, 3)),
            nn.Dropout2d(0.45),
            nn.Conv2d(extractor_channels, extractor_channels, kernel_size=(3, 3), stride=1), nn.ReLU(), nn.MaxPool2d((3, 3)),
            nn.Conv2d(extractor_channels, extractor_channels, kernel_size=(3, 3), stride=1), nn.ReLU(),
            nn.BatchNorm2d(extractor_channels),
        )
        self.predictor = nn.Sequential(  # (extractor_channels * 20)+(base_scale_channel*16)
            nn.Linear((extractor_channels * 16) + (base_scale_channel * 4), 1024), nn.ReLU(),
            # from extractor + L + number of cancer types 10 cancer types
            nn.Dropout(0.50),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, output_size)
        )
        self.softmax = nn.Softmax(dim=1)

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
        x_dec_3d = self.decoder(x_enc)
        x = self.extractor(x_dec_3d)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x_enc_flat], dim=1)
        x = self.predictor(x)
        return self.softmax(x) , x_dec_3d, x_enc_flat