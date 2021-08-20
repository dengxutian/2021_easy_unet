# @Time : 2021/8/17 21:47
# @Author : Deng Xutian
# @Email : dengxutian@126.com
import numpy as np
import torch
import dataset
import network

import cv2 as cv


def main():
    net = network.UNet(colordim=3)
    net.load_state_dict(torch.load('unet.pth', map_location='cpu'))
    img = cv.imread('./dataset/training/00050.png') / 255
    img = cv.resize(img, (572, 572))
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor([img]).float()
    map = net(img)
    map = map.detach().numpy()[0]
    map = np.transpose(map, (1, 2, 0))
    cv.imshow('map', map)
    cv.waitKey(10000)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
