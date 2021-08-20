# @Time : 2021/8/13 16:48
# @Author : Deng Xutian
# @Email : dengxutian@126.com

import os
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader


class img_dataset(Dataset):

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def __len__(self):
        return int(len(os.listdir(self.dir_path)) / 2)

    def __getitem__(self, index):
        file_name = str(index + 1)
        while (len(file_name) < 5):
            file_name = '0' + file_name
        img_name = file_name + '.png'
        label_name = file_name + '_matte.png'
        img = cv.imread(self.dir_path + img_name) / 255
        img = cv.resize(img, (512, 512))
        img = np.transpose(img, (2, 0, 1))

        label = cv.imread(self.dir_path + label_name)
        label = cv.cvtColor(label, cv.COLOR_RGB2GRAY)
        label = cv.resize(label, (512, 512))
        # label = np.transpose(label, (2, 0, 1))
        label = np.where(label < 100, 0, 1)
        label = np.array([label])
        return img, label


if __name__ == '__main__':
    test_dataset = img_dataset(dir_path='./dataset/testing/')
    dataset_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    for data in dataset_loader:
        img, label = data
        print(label)
