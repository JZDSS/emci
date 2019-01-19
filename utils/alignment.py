import torch
import numpy as np
from sklearn.externals import joblib
import pdb
import cv2
import data.face_dataset as fd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import data.utils as ul
from PIL import Image

mean_landmarks = joblib.load('../cache/mean_landmarks.pkl')

def align(images, landmarks, bboxs):
    ones = np.ones(106)
    batch_size = int(images.shape[0])
    for i in range(batch_size):
        image = images[i].data.numpy()
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8).copy()
        plt.imshow(image)
        plt.show()

        landmark = landmarks[i].data.numpy()
        landmark = landmark.reshape((106, 2), order='C')
        #landmark = np.reshape(landmark, (106, 2))

        bbox = bboxs[2*i : 2*i+2]#(1, 4)
        bbox1 = bbox[0].data.numpy()
        bbox2 = bbox[1].data.numpy()
        bbox = np.r_[bbox1, bbox2]
        bbox = bbox.reshape((2, 2))
        bbox = np.c_[bbox, [1, 1]]

        x = np.c_[landmark, ones]
        T = pdb.procrustes(x, mean_landmarks)
        cols, rows, ch = image.shape
        image = cv2.warpAffine(image, np.transpose(T), (cols, rows))
        plt.imshow(image)
        plt.show()

        bbox = bbox@T

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        images[i] = torch.from_numpy(image)
        time.sleep(3)

    return images




if __name__ == "__main__":
    a = fd.FaceDataset("/home/orion/Desktop/ecmi/emci/data/icme", "/home/orion/Desktop/ecmi/emci/data/icme/train")
    b = iter(DataLoader(a, batch_size=2, shuffle=True, num_workers=0))
    while True:
        images, landmarks, bboxs = next(b)
        images = align(images, landmarks, bboxs)