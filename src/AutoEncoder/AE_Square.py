import torch.nn as nn
import torch

from src.AutoEncoder.AE_util import UnFlatten


class AE(nn.Module):
    def __init__(self, output_size, image_channels=5):
        super(AE, self).__init__()
        base_scale_channel = 32

        self.denosiser = nn.Sequential(
            nn.AdaptiveAvgPool3d((5, 66, 66))
        )
        self.encoder = nn.Sequential(# [(W-K+2P)/Stride ] + 1
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, base_scale_channel, kernel_size=2, stride=2, padding=2), # 512/4=128, (B, 34, 34, 34)
            nn.ReLU(),
            nn.Conv2d(base_scale_channel,base_scale_channel*4, kernel_size=2, stride=2, padding=1), # 128/4=32 (B, 17, 17, 17)
            nn.ReLU(),
            nn.Conv2d(base_scale_channel*4, base_scale_channel*8, kernel_size=3,stride=3), #32/4=8 (B, 100, 5, 5)
            nn.ReLU(),
            nn.Conv2d(base_scale_channel*8, base_scale_channel*16, kernel_size=5,stride=5), #8/8=1 (B, 120, 1, 1)
            nn.ReLU(),
            #UnFlatten(-1, base_scale_channel*16, 1, 1),
        )
        self.decoder = nn.Sequential( # (input-2) * stride - 2padding + (kernel-1) + 1
            nn.ConvTranspose2d(base_scale_channel*16, base_scale_channel*8, kernel_size=5, stride=5), # 5,5
            nn.ReLU(),
            nn.ConvTranspose2d(base_scale_channel*8, base_scale_channel*4, kernel_size=5, stride=3), # 17
            nn.ReLU(),
            nn.ConvTranspose2d(base_scale_channel*4, base_scale_channel, kernel_size=2,stride=2), # 34
            nn.ReLU(),
            nn.ConvTranspose2d(base_scale_channel, image_channels, kernel_size=2, stride=2, padding=1), # 66
            nn.ReLU(),
        )

        self.denosiser3d = nn.Sequential(
            #nn.MaxPool3d((image_channels, 1, 1))
            nn.AdaptiveAvgPool3d((1, 66, 66))
        )

        extractor_channels = 8
        self.extractor = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, extractor_channels, kernel_size=(3, 3), stride=3, padding=2),
            nn.PReLU(extractor_channels),
            nn.Dropout2d(0.43),
            nn.Conv2d(extractor_channels, extractor_channels, kernel_size=(3, 3), stride=3),
            nn.PReLU(extractor_channels),
            nn.Dropout2d(0.43),
            nn.Conv2d(extractor_channels, extractor_channels, kernel_size=(2, 2), stride=2),
            nn.PReLU(extractor_channels),
            nn.Dropout2d(0.43),
            nn.BatchNorm2d(extractor_channels)
        )
        self.predictor = nn.Sequential( #(extractor_channels * 20)+(base_scale_channel*16)
            nn.Linear((base_scale_channel * 16) + (extractor_channels * 3 * 3), 128),nn.PReLU(), # from extractor + L + number of cancer types 10 cancer types
            nn.Dropout(0.43),
            nn.Linear(128, 128), nn.PReLU(),
            nn.Dropout(0.43),
            nn.Linear(128, 128), nn.PReLU(),
            nn.Dropout(0.43),
            nn.Linear(128, output_size)
        )

    def predict(self, x):
        x = self.forward(x)
        return x

    def encode(self, x):
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        x = self.denosiser(x)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        x_dec = self.decoder(x_enc)
        x_dec = self.denosiser3d(x_dec)
        x = self.extractor(x_dec)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x_enc_flat], dim=1)
        x = self.predictor(x)
        return x #, x_dec