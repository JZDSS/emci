import os
import sys
curr = sys.path[0]
sys.path.append(curr)
import torch
from models import two_stage
import cv2
from utils.alignment import Align
from data import utils
import warnings
warnings.filterwarnings('ignore')


net = two_stage.TwoStage(Align(os.path.join(curr, 'weights/mean_landmarks.pkl'), (224, 224), (0.2, 0.1)),
                         os.path.join(curr, 'weights/1'),
                         os.path.join(curr, 'weights/2')).cuda()

net.eval()

img_path = sys.argv[1]
image = cv2.imread(img_path)
bbox = [0, 0, image.shape[1] - 1, image.shape[0] - 1]

with torch.no_grad():
    pr = net.forward(image, bbox)

utils.save_landmarks(pr, sys.argv[2])