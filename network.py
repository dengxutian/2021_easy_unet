import torch
import torch.nn as nn

class unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_conv_layers = double_conv_layers(in_channel, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.out_conv_layers = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv_layers(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        x = self.out_conv_layers(x)
        return x

class double_conv_layers(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        return self.conv_layers(x)

class down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                double_conv_layers(in_channel, out_channel)
                )

    def forward(self, x):
        return self.maxpool_conv(x)

class up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_layers = double_conv_layers(in_channel, out_channel)
        

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        return self.conv_layers(torch.cat([x1, x2], dim = 1))
