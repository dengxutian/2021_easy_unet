# @Time : 2021/9/4 16:27
# @Author : Deng Xutian
# @Email : dengxutian@126.com

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_conv_layers = DoubleConvLayers(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out_conv_layers = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv_layers(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.out_conv_layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DoubleConvLayers(nn.Module):
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


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.MaxPoolConv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            DoubleConvLayers(in_channel, out_channel)
        )

    def forward(self, x):
        return self.MaxPoolConv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_layers = DoubleConvLayers(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        return self.conv_layers(torch.cat([x1, x2], dim=1))


# if __name__ == '__main__':
#     net = UNet(in_channel=1, out_channel=1)
#     x = torch.rand(1, 1, 224, 224)
#     y = net(x)
#     print(y)
#     print('Finish!')