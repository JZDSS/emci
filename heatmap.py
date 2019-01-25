import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline as SBS
import os
import cv2 as cv

from data import utils

def get_id(name):
    t = name.split('_')[0:2]
    return t[0] + t[1]

root_dir = 'D:\icmedata\sendaligned\\aligned\\'

lamdmark_dir = os.path.join(root_dir, 'landmark')
image_dir = os.path.join(root_dir, 'picture')
#bbox_dir = os.path.join(root_dir, 'bbox')
filenames = os.listdir(image_dir)

for filename in filenames:
    id = get_id(filename)
    landmark_path = os.path.join(lamdmark_dir, filename + '.txt')
    landmarks = utils.read_mat(landmark_path)
    x = landmarks[0:33,0]
    y = landmarks[0:33,1]
    img = np.zeros((128,128,3),np.uint8)
    # img = img +255
    # img.fill(255)
    color = (255, 255, 255)
    # print(landmarks[0,:])
    # print(landmarks[1,:])
    for j in range(105):
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j+1, 0], landmarks[j+1, 1]), color,thickness=2)#thickness
    #cv.imshow('image', img)
    pic2 = cv.GaussianBlur(img,(11,11),3)
    #两个sigma允许输入负数等其他不常用的输入。
    #cv.imshow('image',img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    # picstack = np.hstack([img,pic2])
    # cv.imshow('stack',picstack)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\" + id + ".png"
    cv.imwrite(Img_Name,pic2)