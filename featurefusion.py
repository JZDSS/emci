from Ushape import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.alignment import Align
from torch.utils.data import DataLoader
from data.align_dataset_input_fusion import AlignDataset
import cv2

a = AlignDataset('/data/icme/data/picture',
                     '/data/icme/data/landmark',
                     '/data/icme/data/landmark',
                     '/data/icme/valid',
                     Align('../cache/mean_landmarks.pkl', (224, 224), (0.15, 0.05)))
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))

images, landmarks, heatmapall, imagesshape = next(batch_iterator)

#要把heatmapall从128*128*3变为featuremap的大小*3（文章里是32*32）


from models.resnet18_local import ResNet18
featuremap = ResNet18(images)
size = featuremap.shape[2:4]
#不知道featuresize是不是shape[2:4]


heatmapall =cv2.resize(heatmapall,size)
N = featuremap.shape[1]
#不知道通道数是不是shape1
model = UNet(N,in_channels=N+3, depth=5, merge_mode='concat')
x = np.c_[heatmapall,featuremap]
#现在x是32*32*（N+3）
#要把这个x变为 （N+3）*32*32
# x = Variable(torch.FloatTensor(np.random.random((1, N+13, 320, 320))))
out = model(x)
out = F.sigmoid(out)
print(out)
print(out.shape)
#实现了输入N+3输出N,文章里是输入N+13输出N那是因为文章的heatmap是13维的
loss = torch.sum(out)
print(loss)