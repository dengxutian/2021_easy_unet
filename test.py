import torch
import network
import numpy as np

def main():
    #net = network.unet(in_channel=3, out_channel=1)
    #x = torch.rand(1,3,224,224).float()
    #y = net(x)
    #print(y)
    net = torch.nn.MaxPool2d(2)
    x = torch.rand(1,3,224,224)
    y = net(x)
    pass

if __name__ == '__main__':
    main()
    print('Finish!')
