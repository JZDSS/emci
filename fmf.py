import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2
from models import Ushape

#注意这个fmf输出是输入通道数的二倍
class fmf(nn.Module):
    def __init__(self):
        super(fmf, self).__init__()
        self.u =Ushape.UNet(64, in_channels=64 + 3, depth=5, merge_mode='concat')
        self.resize = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        #第一个featurefusion其中heatmap缩放用的maxpooling，128到32
    def forward(self, featuremap, heatmap):

        size = featuremap.shape[2]#[3]也行无所谓

        heatmapall = self.resize(heatmap)

        #print(heatmapall.shape)
        #print(featuremap.shape)
        x = torch.cat([heatmapall,featuremap],1)
        #print(x.shape)
        # x = np.c_[heatmapall, featuremap]
        out = self.u(x)
        out = torch.sigmoid(out)
        out = out * featuremap
        #print(out.shape)
        #out = np.c_[out, featuremap]
        out = torch.cat([out,featuremap],1)
        #print(x.shape)
        return out



if __name__ == '__main__':
    from torch.autograd import Variable
    # heatmap =np.zeros((128, 128, 3), np.uint8)
    heatmap = Variable(torch.FloatTensor(np.zeros((1, 3, 128, 128), np.uint8)))
    # print(heatmap)
    # print(heatmap.shape)
    featuremap = Variable(torch.FloatTensor(np.random.random((1, 64, 32, 32))))
    #假设featuremap是64通道的32*32
    FMF= fmf()
    z =FMF(featuremap,heatmap)
    #print(z.shape)
