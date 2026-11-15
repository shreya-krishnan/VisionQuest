"""Exact U-Net architecture used to train best_unet.pth."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 256, 256)
        self.dec3 = DoubleConv(256 + 128, 128)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.dec1 = DoubleConv(64 + 32, 32)
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = F.interpolate(
            b, size=e4.shape[2:], mode="bilinear", align_corners=False
        )
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = F.interpolate(
            d4, size=e3.shape[2:], mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = F.interpolate(
            d3, size=e2.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = F.interpolate(
            d2, size=e1.shape[2:], mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.output_conv(d1)
