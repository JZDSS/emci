import matplotlib.pyplot as plt
import matplotlib.image as im
from data import utils
from skimage import img_as_ubyte
from models import resnet18
import torch
import cv2
import numpy as np


def pic2tencolor(a, num):
    image = []
    for i in range(num):
        image.append(utils.random_color(a))
    return image


class DTA:
    """
    功能: data time augmentation
    输入: a为一张crop后的图片,类型为numpy数组,大小为(224,208,3),dtype为uint8
         n为生成的不同颜色的图片数量
         net为所需要调用的网络

    """
    def __init__(self, a, n):
        self.gencolor = pic2tencolor(a, n)
        self.net = net
        self.img = []
        self.image = []
        self.img2tensor = np.array
        self.tft = torch.from_numpy
        self.tmean = torch.mean
        self.out = torch.tensor
        self.output = torch.tensor
    def forward(self):

        self.image = self.gencolor
        for i in range(n):
            self.img.append(np.transpose(cv2.resize(self.image[i], (224, 224)), (2, 0, 1)).astype(np.float32))
        self.out = net((self.tft(self.img2tensor(self.img))))
        self.output = self.tmean(self.out, 0)

        return self.output
if __name__=='__main__':
    net = resnet18.ResNet18()
    a = im.imread("/home/zhang/data/crop/data/picture/AFW_134212_1_0.png")
    a = img_as_ubyte(a[:, :, 0:3])
    n = 5
    dta = DTA(a, n)
    output = dta.forward()
    '''image = pic2tencolor(a, n)
    img = []
    for i in range(n):
        img.append(np.transpose(cv2.resize(image[i], (224, 224)), (2, 0, 1)).astype(np.float32))
    
    img2tensor = np.array(img)
    out = net((torch.from_numpy(img2tensor)))
    output = torch.mean(out, 0)'''
    z=1



