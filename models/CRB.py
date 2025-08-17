import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

        if self.in_channels != self.out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        input_residuals = x
        x = self.conv_block(x) + self.identity(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


class Encoder(nn.Module):  # [32,64,128,256]
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], residual_blocks_per_layer=2):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=1, padding=1)

        self.layers = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.layers.append(ResidualBlock(in_channels=channels[i], out_channels=channels[i + 1]))
            self.layers.append(nn.ReLU())

            for _ in range(residual_blocks_per_layer - 1):
                self.layers.append(ResidualBlock(in_channels=channels[i + 1], out_channels=channels[i + 1]))
                self.layers.append(nn.ReLU())
                self.layers.append(Downsample(in_channels=channels[i + 1], out_channels=channels[i + 1]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layers(x)
        return x


class Decoder(nn.Module):  # [256,128,64,32]
    def __init__(self, out_channels=3, channels=[256, 128, 64, 32], residual_blocks_per_layer=2):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.layers.append(ResidualBlock(in_channels=channels[i], out_channels=channels[i + 1]))
            self.layers.append(nn.ReLU())

            for _ in range(residual_blocks_per_layer - 1):
                self.layers.append(ResidualBlock(in_channels=channels[i + 1], out_channels=channels[i + 1]))
                self.layers.append(nn.ReLU())
                self.layers.append(Upsample(in_channels=channels[i + 1], out_channels=channels[i + 1]))

        self.layers = nn.Sequential(*self.layers)

        self.out_conv = nn.Conv2d(in_channels=channels[-1], out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.out_conv(x)
        x = F.tanh(x)
        return x
