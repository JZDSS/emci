from data.face_dataset import FaceDataset
from data.pose_dataset import PoseDataset
import torch
from torch.utils.data import DataLoader
from layers.module.wing_loss import WingLoss
from models.saver import Saver
from models.resnet50 import ResNet50
from models.resnet18_rfb import ResNet18
from tensorboardX import SummaryWriter
import numpy as np
from utils.metrics import Metrics
import time
from data.utils import draw_landmarks


net = ResNet18().cuda()

criterion = WingLoss(10, 0.5)

#PATH = './ckpt'
a = FaceDataset("/data/icme", "//data/icme/valid", phase='eval')
batch_iterator = iter(DataLoader(a, batch_size=1, shuffle=True, num_workers=4))
#Saver.dir=PATH
saver = Saver('ckpt', 'model')
current = None
net.eval()

epoch_size = len(a)
metrics = Metrics().add_nme().add_auc()
model_name = 'model-???.pth'
saver.load(net, model_name)


all_pr = []
all_gt = []
while True:
    try:
        images, landmarks = next(batch_iterator)
    except:
        break
    images = images.cuda()
    landmarks = landmarks.cuda()
    with torch.no_grad():
        out = net.forward(images)

    pr = out.cpu().data.numpy()
    gt = landmarks.cpu().data.numpy()

    all_pr.append(pr)
    all_gt.append(gt)
all_gt = np.concatenate(all_gt, axis=0)
all_pr = np.concatenate(all_pr, axis=0)

nme = metrics.nme.update(np.reshape(all_gt, (-1, 106, 2)), np.reshape(all_pr, (-1, 106, 2)))
metrics.auc.update(nme)

print("NME: %f\nAUC: %f" % (metrics.nme.value, metrics.auc.value))
