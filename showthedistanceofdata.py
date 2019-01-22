from data.lb_dataset import LBDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import torch
from data.utils import draw_landmarks
a =LBDataset("D:\icmedata\correctdata","D:\icmedata\correctdata\\train")
b = iter(DataLoader(a, batch_size=1, shuffle=True, num_workers=0))
target=[]
for iteration in range(10):
    bbox, landmarks = next(b)
    landmarks = landmarks.numpy()
    landmarks_flat = landmarks.ravel()
    bbox = bbox.numpy()
    y = landmarks_flat[1::2]
    y_max = np.max(y)
    y_min = np.min(y)
    x = landmarks_flat[::2]
    x_max = np.max(x)
    x_min = np.min(x)
    chang = x_max - x_min
    kuan = y_max - y_min
    Slandmark = chang * kuan
    #print(bbox[:,0])
    Sbbox = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])
    Sbbox = Sbbox[0]
    target.append(Slandmark/Sbbox)
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()
draw_hist(target,'acreage title','Sb/Sl','amount',0,1,0,10)
#print(target)