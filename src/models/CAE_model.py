import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, out_channels=64):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
      #3X128X128 => 64x64x64
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
      #64x64x64 => 128x32x32
      nn.BatchNorm2d(out_channels*2),
      nn.ReLU(),
      nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=3, stride=2, padding=1, bias=False),
      #128x32x32 => 256x16x16
      nn.BatchNorm2d(out_channels*4),
      nn.ReLU(),
      nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*8, kernel_size=3, stride=2, padding=1, bias=False),
      #256x16x16 => 512x8x8
      nn.BatchNorm2d(out_channels*8),
      nn.ReLU(),
      nn.Conv2d(in_channels=out_channels*8, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
      #512x8x8 => 64x4x4
      nn.ReLU()
    )

  def forward(self, x):
    encoded = self.encoder(x)
    return encoded

class Decoder(nn.Module):
  def __init__(self, out_channels=64):
    super().__init__()

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1),
      #64x4x4 => 512x8x8
      nn.BatchNorm2d(out_channels*8),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=out_channels*8, out_channels=out_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
      #512x8x8 => 256x16x16
      nn.BatchNorm2d(out_channels*4),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=out_channels*4, out_channels=out_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
      #256x16x16 => 128x32x32
      nn.BatchNorm2d(out_channels*2),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
      #128x32x32 => 64x64x64
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=out_channels, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
      #64x64x64 => 3X128x128
    )

  def forward(self, x):
    decoded = self.decoder(x)
    decoded = F.tanh(decoded)
    return decoded