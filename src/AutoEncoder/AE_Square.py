import torch.nn as nn
import torch

class AE(nn.Module):
    def __init__(self, output_size, image_channels=1):
        super(AE, self).__init__()
        base_scale_channel = 32

        self.encoder = nn.Sequential(  # [(W-K+2P)/Stride ] + 1
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, base_scale_channel, kernel_size=2, stride=2, padding=2),  # W=98
            nn.PReLU(base_scale_channel),
            nn.Conv2d(base_scale_channel, base_scale_channel * 2, kernel_size=2, stride=2, padding=1),  # W = 50
            nn.PReLU(base_scale_channel * 2),
            nn.Conv2d(base_scale_channel * 2, base_scale_channel * 4, kernel_size=2, stride=2),  # W = 24
            nn.PReLU(base_scale_channel * 4),
            nn.Conv2d(base_scale_channel * 4, base_scale_channel * 8, kernel_size=4, stride=4),  # W = 5
            nn.PReLU(base_scale_channel * 8),
            nn.Conv2d(base_scale_channel * 8, base_scale_channel * 16, kernel_size=4, stride=4),  # W = 1
            nn.PReLU(base_scale_channel * 16),
            # UnFlatten(-1, base_scale_channel*16, 1, 1),
        )
        self.decoder = nn.Sequential(  # (input-1) * stride - 2padding + (kernel-1) + 1
            nn.ConvTranspose2d(base_scale_channel * 16, base_scale_channel * 8, kernel_size=4, stride=4, output_padding=1),  # W = 4
            nn.PReLU(base_scale_channel * 8),
            nn.ConvTranspose2d(base_scale_channel * 8, base_scale_channel * 4, kernel_size=5, stride=5),  # 24
            nn.PReLU(base_scale_channel * 4),
            nn.ConvTranspose2d(base_scale_channel * 4, base_scale_channel * 2, kernel_size=2, stride=2),  # 50
            nn.PReLU(base_scale_channel * 2),
            nn.ConvTranspose2d(base_scale_channel * 2, base_scale_channel, kernel_size=2, stride=2),  # 100
            nn.PReLU(base_scale_channel),
            nn.ConvTranspose2d(base_scale_channel, 3, kernel_size=2, stride=2, padding=3),  # 194
            nn.PReLU(3),
        )

        resulting_img_channels = 1
        self.denosiser3d = nn.Sequential(
            nn.AdaptiveMaxPool3d((resulting_img_channels, 194, 194))
        )

        extractor_channels = 32
        self.extractor = nn.Sequential(
            nn.BatchNorm2d(resulting_img_channels),
            nn.Conv2d(resulting_img_channels, extractor_channels, kernel_size=(3, 3), stride=3, padding=2),
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
        self.predictor = nn.Sequential(  # (extractor_channels * 20)+(base_scale_channel*16)
            nn.Linear((extractor_channels * 121) + (base_scale_channel * 16), 512), nn.PReLU(),
            # from extractor + L + number of cancer types 10 cancer types
            nn.Dropout(0.45),
            nn.Linear(512, 512), nn.PReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, 512), nn.PReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, output_size)
        )

    def predict(self, x):
        x = self.forward(x)
        return x

    def encode(self, x):
        x = self.denosiser3d(x)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        x = self.denosiser3d(x)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        x_dec_3d = self.decoder(x_enc)
        x_dec = self.denosiser3d(x_dec_3d)
        x = self.extractor(x_dec)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x_enc_flat], dim=1)
        x = self.predictor(x)

        return x #, x_dec_3d, x_enc_flat