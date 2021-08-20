# @Time : 2021/8/13 16:48
# @Author : Deng Xutian
# @Email : dengxutian@126.com

import torch
import dataset
import network
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class train():

    def __init__(self):
        self.batch_size = 1
        self.num_workers = 4
        self.shuffle = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = network.unet(in_channel=3, out_channel=1).to(self.device)

        # train_dataset_list = []
        # valid_dataset_list = []
        #
        # for i in range(60):
        #     if i % 5 == 0:
        #         valid_dataset_list.append(dataset.label_dataset(dataset_index=i))
        #     else:
        #         train_dataset_list.append(dataset.label_dataset(dataset_index=i))

        self.train_dataset_loader = DataLoader(dataset.img_dataset(dir_path='./dataset/training/'),
                                               shuffle=self.shuffle, batch_size=self.batch_size,
                                               num_workers=self.num_workers)
        self.valid_dataset_loader = DataLoader(dataset.img_dataset(dir_path='./dataset/testing/'),
                                               shuffle=self.shuffle, batch_size=self.batch_size,
                                               num_workers=self.num_workers)

        self.train_loss_list = []
        self.valid_loss_list = []
        pass

    def save(self):
        torch.save(self.net.state_dict(), './unet.pth')
        pass

    def load(self):
        self.net.load_state_dict(torch.load('./unet.pth', map_location='cpu'))
        self.net.to(self.device)
        pass

    def save_txt(self):
        train_loss_file = open('./train_loss.txt', 'a')
        valid_loss_file = open('./valid_loss.txt', 'a')

        try:
            train_loss_str = str(len(self.train_loss_list)) + ' ' + str(
                self.train_loss_list[len(self.train_loss_list) - 1]) + '\n'
        except:
            train_loss_str = '\n'

        try:
            valid_loss_str = str(len(self.valid_loss_list)) + ' ' + str(
                self.valid_loss_list[len(self.valid_loss_list) - 1]) + '\n'
        except:
            valid_loss_str = '\n'

        train_loss_file.write(train_loss_str)
        valid_loss_file.write(valid_loss_str)

        train_loss_file.close()
        valid_loss_file.close()

    def train(self):
        self.net.train()
        train_n, train_loss = 0, 0.0
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        loss_func = torch.nn.BCELoss()
        for data in self.train_dataset_loader:
            img, label = data
            img = img.to(self.device).float()
            label = label.to(self.device).float()
            fake_label = self.net(img)
            fake_label = torch.sigmoid(fake_label)
            loss = loss_func(fake_label, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_n += label.size(0)
            train_loss += loss.cpu().item()
            print('train n = ' + str(train_n))
            print('train loss = ' + str(loss.cpu().item()))

        train_loss = train_loss / train_n
        print('Train Loss = ' + str(train_loss))
        self.train_loss_list.append(train_loss)

    def valid(self):
        self.net.eval()
        valid_n, valid_loss = 0, 0.0
        loss_func = torch.nn.BCELoss()
        for data in self.train_dataset_loader:
            img, label = data
            img = img.to(self.device).float()
            label = label.to(self.device).float()
            fake_label = self.net(img)
            fake_label = torch.sigmoid(fake_label)
            loss = loss_func(fake_label, label)

            valid_n += label.size(0)
            valid_loss += loss.cpu().item()
            print('valid n = ' + str(valid_n))
            print('valid loss = ' + str(loss.cpu().item()))

        valid_loss = valid_loss / valid_n
        print('Valid Loss = ' + str(valid_loss))
        self.valid_loss_list.append(valid_loss)
        pass


if __name__ == '__main__':
    demo = train()
    for i in range(20):
        print('##########')
        print('Epoch = ' + str(i + 1))
        demo.train()
        demo.valid()
        demo.save_txt()
        if (i + 1) % 5 == 0:
            demo.save()
    print('Finish!')
