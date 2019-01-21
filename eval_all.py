from data.face_dataset import FaceDataset
import torch
from models.saver import Saver
from models import dense201
from models import resnet18
import numpy as np
from utils.metrics import Metrics
from data import utils
import cv2
import os


net = resnet18.ResNet18().cuda()

#PATH = './ckpt'
a = FaceDataset("/data/icme", "/data/icme/valid", phase='eval')
#Saver.dir=PATH
saver = Saver('backup', 'model')
current = None
net.eval()

epoch_size = len(a)
metrics = Metrics().add_nme().add_auc()
model_name = 'resnet18-9200.pth'
saver.load(net, model_name)


all_pr = []
all_gt = []
save_dir = '/data/icme/data/pred_landmark'
for i in range(len(a)):
    img_path = a.images[i]
    name = img_path.split('/')[-1]
    bbox_path = a.bboxes[i]
    landmark_path = a.landmarks[i]
    bbox = utils.read_bbox(bbox_path)
    landmarks = utils.read_landmarks(landmark_path)
    landmarks = utils.norm_landmarks(landmarks, bbox)
    image = cv2.imread(img_path)
    minx, miny, maxx, maxy = bbox
    image = image[miny:maxy + 1, minx:maxx + 1, :]
    image = cv2.resize(image, a.shape)

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(image, 0)
    landmarks = np.expand_dims(landmarks, 0)

    image = torch.tensor(image).cuda()
    landmarks = torch.tensor(landmarks).cuda()
    with torch.no_grad():
        out = net.forward(image)

    pr = out.cpu().data.numpy()
    gt = landmarks.cpu().data.numpy()

    all_pr.append(pr)
    all_gt.append(gt)
    # save prediction
    pr = np.reshape(pr.copy(), (106, 2))
    pr = utils.inv_norm_landmark(pr, bbox)
    utils.save_landmarks(pr, os.path.join(save_dir, name + '.txt'))

all_gt = np.concatenate(all_gt, axis=0)
all_pr = np.concatenate(all_pr, axis=0)

nme = metrics.nme.update(np.reshape(all_gt, (-1, 106, 2)), np.reshape(all_pr, (-1, 106, 2)))
metrics.auc.update(nme)

print("NME: %f\nAUC: %f" % (metrics.nme.value, metrics.auc.value))
