from data._align_dataset import AlignDataset
import torch
from models.saver import Saver
from models import dense201
from models import resnet18
import numpy as np
from utils.metrics import Metrics
from data import utils
import cv2
import os
from torch.utils.data import DataLoader
from utils.alignment import Align


net = dense201.Dense201(num_classes=30).cuda()
ids = list(range(51, 66))
#PATH = './ckpt'
a = AlignDataset('/data/icme/data/picture',
                 '/data/icme/data/landmark',
                 '/data/icme/data/pred_landmark',
                 '/data/icme/valid',
                 Align('cache/mean_landmarks.pkl', (224, 224), (0.15, 0.1),
                       idx=ids),
                 phase='eval',
                 ldmk_ids=ids
                 )
#Saver.dir=PATH
saver = Saver('backup', 'model')
current = None
net.eval()

epoch_size = len(a)
metrics = Metrics().add_nme().add_auc()
model_name = 'nose-model-8400.pth'
saver.load(net, model_name)


all_pr = []
all_gt = []
save_dir = '/data/icme/data/pred_landmark_align_nose'
base_dir = '/data/icme/data/pred_landmark_align'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
batch_iterator = iter(DataLoader(a, batch_size=1, shuffle=False, num_workers=4))

for i in range(len(a)):
    image, gt, t, name = next(batch_iterator)
    pr = utils.read_mat(os.path.join(base_dir, name[0] + '.txt'))
    image = image.cuda()
    with torch.no_grad():
        nose = net.forward(image)

    nose = nose.cpu().data.numpy()
    gt = gt.data.numpy()
    t = t.data.numpy()
    nose = np.reshape(nose, (-1, 2))
    nose[:, 0] *= a.shape[1]
    nose[:, 1] *= a.shape[0]
    nose = a.aligner.inverse(nose, t[0])
    pr[ids] = nose
    gt = np.reshape(gt, (-1, 2))
    all_pr.append(pr)
    all_gt.append(gt)
    # save prediction
    utils.save_landmarks(pr, os.path.join(save_dir, name[0] + '.txt'))

all_gt = np.stack(all_gt, axis=0)
all_pr = np.stack(all_pr, axis=0)

nme = metrics.nme.update(np.reshape(all_gt, (-1, 106, 2)), np.reshape(all_pr, (-1, 106, 2)))
metrics.auc.update(nme)

print("NME: %f\nAUC: %f" % (metrics.nme.value, metrics.auc.value))
