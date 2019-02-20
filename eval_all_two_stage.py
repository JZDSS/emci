from data.original_dataset import OriginalDataset
import torch
from models.saver import Saver
from models import two_stage
import numpy as np
from utils.metrics import Metrics
from data import utils
import cv2
import os
from torch.utils.data import DataLoader
from utils.alignment import Align


net = two_stage.TwoStage(Align('cache/mean_landmarks.pkl', (224, 224), (0.2, 0.1))).cuda()

#PATH = './ckpt'
a = OriginalDataset('/data/icme/data/picture',
                 '/data/icme/data/landmark',
                 '/data/icme/bbox',
                 '/data/icme/valid',
                 phase='eval')
current = None
net.eval()

epoch_size = len(a)
metrics = Metrics().add_nme().add_auc()

all_pr = []
all_gt = []
save_dir = '/data/icme/data/pred_landmark_two_stage'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
batch_iterator = iter(DataLoader(a, batch_size=1, shuffle=False, num_workers=4))

for i in range(len(a)):
    image, gt, bbox, name = a.__getitem__(i)

    with torch.no_grad():
        pr = net.forward(image, bbox)

    all_pr.append(pr)
    all_gt.append(gt)
    # save prediction
    utils.save_landmarks(pr, os.path.join(save_dir, name[0] + '.txt'))

all_gt = np.stack(all_gt, axis=0)
all_pr = np.stack(all_pr, axis=0)

nme = metrics.nme.update(np.reshape(all_gt, (-1, 106, 2)), np.reshape(all_pr, (-1, 106, 2)))
metrics.auc.update(nme)

print("NME: %f\nAUC: %f" % (metrics.nme.value, metrics.auc.value))
