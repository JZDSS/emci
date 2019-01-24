import numpy as np
import cv2
from data import utils

imgdir = '/data/icme/data/picture/AFW_5452623_1_5.jpg'
landmarks = utils.read_mat('/data/icme/data/landmark/AFW_5452623_1_5.jpg.txt')
bbox = utils.read_bbox('/data/icme/bbox/AFW_5452623_1_5.jpg.rect')
img = cv2.imread(imgdir)
minx, miny, maxx, maxy = bbox
img = img[miny:maxy+1, minx:maxx+1, :]
landmarks = utils.norm_landmarks(landmarks, bbox)
img, landmarks = utils.random_flip(img, landmarks, 1)

img = np.transpose(img, (2, 0, 1))
img = utils.draw_landmarks(img, landmarks, (255, 255, 255))
img = np.transpose(img, (1, 2, 0))

cv2.imshow('', img)
cv2.waitKey(0)




