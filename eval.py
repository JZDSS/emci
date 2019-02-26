import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
from torch.utils.data import DataLoader
from sparse import loss
from models.saver import Saver
from models.dense201 import Dense201
from tensorboardX import SummaryWriter
import numpy as np
from utils.metrics import Metrics
import time
from data.utils import draw_landmarks
from utils.alignment import Align
from data.bbox_dataset import BBoxDataset
from data.align_dataset import AlignDataset
import argparse
from proto import all_pb2
from google.protobuf import text_format

parser = argparse.ArgumentParser(
    description='Landmark Detection Training')
parser.add_argument('-c', '--config', type=str)
args = parser.parse_args()

cfg = all_pb2.Config()
with open(args.config, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, cfg)

net = Dense201(num_classes=212)
if cfg.device == all_pb2.GPU:
    net = net.cuda()

criterion = loss.get_criterion(cfg.loss)
# criterion = WingLoss(10, 2)
#PATH = './ckpt'
# a = BBoxDataset('/data/icme/crop/data/picture',
#                     '/data/icme/crop/data/landmark',
#                     '/data/icme/valid', phase='eval')
a = AlignDataset('/data/icme/crop/data/picture',
                 '/data/icme/crop/data/landmark',
                 '/data/icme/crop/data/pred_landmark',
                 '/data/icme/valid',
                 Align('./cache/mean_landmarks.pkl', (224, 224), (0.2, 0.1),
                       ), # idx=list(range(51, 66))),
                 phase='eval',
                 # ldmk_ids=list(range(51, 66))
                 )
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))
#Saver.dir=PATH
saver = Saver(os.path.join(cfg.root, 'ckpt'), 'model')
saver2 = Saver(os.path.join(cfg.root, 'snapshot'), 'model')
current = None
net.eval()
batch_size = 4
epoch_size = len(a) // batch_size
writer = SummaryWriter(os.path.join(cfg.root, 'logs/valid'))
metrics = Metrics().add_nme(0.9).add_auc(decay=0.9).add_loss(decay=0.9)
while True:
    if current == saver.last_ckpt():
        time.sleep(1)
    else:
        last = saver.last_ckpt()
        current_iter = 0 if last is None else int(last.split('.')[0].split('-')[-1])
        while True:
            try:
                saver.load(net, last)
                break
            except:
                continue
        batch_iterator = iter(DataLoader(a, batch_size,
                                         shuffle=True, num_workers=0))
        images = None
        landmarks = None
        gt = None
        pr = None
        for iteration in range(40):
            images, landmarks = next(batch_iterator)
            if cfg.device == all_pb2.GPU:
                images = images.cuda()
                landmarks = landmarks.cuda()
            with torch.no_grad():
                out = net.forward(images)

            loss = criterion(out, landmarks)
            pr = out.cpu().data.numpy()
            gt = landmarks.cpu().data.numpy()
            nme = metrics.nme.update(np.reshape(gt, (-1, gt.shape[1]//2, 2)), np.reshape(pr, (-1, gt.shape[1]//2, 2)))
            metrics.auc.update(nme)
            metrics.loss.update(loss.item())
        image = images.cpu().data.numpy()[0]
        # 绿色的真实landmark
        image = draw_landmarks(image, gt[0], (0, 255, 0))
        # 红色的预测landmark
        image = draw_landmarks(image, pr[0], (0, 0, 255))
        image = image[::-1, ...]
        writer.add_scalar("watch/NME", metrics.nme.value * 100, current_iter)
        writer.add_scalar("watch/AUC", metrics.auc.value * 100, current_iter)
        writer.add_scalar("watch/loss", metrics.loss.value, current_iter)
        writer.add_image("result", image, current_iter)

        metrics.clear()
        current = last
        if current_iter % 2000 == 0:
            state = net.state_dict()
            saver2.save(state, current_iter)

        if current_iter == cfg.max_iter:
            break



