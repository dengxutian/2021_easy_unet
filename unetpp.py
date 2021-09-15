# @Time : 2021/9/5 17:26
# @Author : Deng Xutian
# @Email : dengxutian@126.com

import torch
import torch.nn as nn


class UNetPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_00 = DoubleConvLayers(in_channel, 64)
        self.conv_10 = DoubleConvLayers(64, 64)
        self.conv_20 = DoubleConvLayers(64, 64)
        self.conv_30 = DoubleConvLayers(64, 64)
        self.conv_01 = DoubleConvLayers(128, 64)
        self.conv_11 = DoubleConvLayers(128, 64)
        self.conv_21 = DoubleConvLayers(128, 64)
        self.conv_02 = DoubleConvLayers(192, 64)
        self.conv_12 = DoubleConvLayers(192, 64)
        self.conv_03 = DoubleConvLayers(256, 64)
        self.down_00 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down_10 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down_20 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.up_00 = nn.Upsample(scale_factor=2)
        self.up_10 = nn.Upsample(scale_factor=2)
        self.up_20 = nn.Upsample(scale_factor=2)
        self.up_01 = nn.Upsample(scale_factor=2)
        self.up_11 = nn.Upsample(scale_factor=2)
        self.up_02 = nn.Upsample(scale_factor=2)
        self.conv_l1 = nn.Conv2d(64, out_channel, kernel_size=1, padding=0)
        self.conv_l2 = nn.Conv2d(64, out_channel, kernel_size=1, padding=0)
        self.conv_l3 = nn.Conv2d(64, out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        x_00 = self.conv_00(x)
        down_00 = self.down_00(x_00)
        x_10 = self.conv_10(down_00)
        down_10 = self.down_10(x_10)
        x_20 = self.conv_20(down_10)
        down_20 = self.down_20(x_20)
        x_30 = self.conv_30(down_20)

        up_00 = self.up_00(x_10)
        up_10 = self.up_10(x_20)
        up_20 = self.up_20(x_30)

        x_01 = self.conv_01(torch.cat([x_00, up_00], dim=1))
        x_11 = self.conv_11(torch.cat([x_10, up_10], dim=1))
        x_21 = self.conv_21(torch.cat([x_20, up_20], dim=1))

        up_01 = self.up_01(x_11)
        up_11 = self.up_11(x_21)

        x_02 = self.conv_02(torch.cat([x_00, x_01, up_01], dim=1))
        x_12 = self.conv_12(torch.cat([x_10, x_11, up_11], dim=1))

        up_02 = self.up_02(x_12)

        x_03 = self.conv_03(torch.cat([x_00, x_01, x_02, up_02], dim=1))

        l1 = self.conv_l1(x_01)
        l2 = self.conv_l2(x_02)
        l3 = self.conv_l3(x_03)

        return (l1 + l2 + l3) / 3


class DoubleConvLayers(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.ConvLayers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ConvLayers(x)


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
        self.UpSample = nn.Upsample(scale_factor=2)
        self.ConvLayers = DoubleConvLayers(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.UpSample(x1)
        return self.ConvLayers(torch.cat([x1, x2], dim=1))


if __name__ == '__main__':
    net = UNetPP(in_channel=3, out_channel=3)
    x = torch.rand(1, 3, 224, 224)
    y = net(x)
    print(y.shape)
    print('Finish!')
