import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import shutil
import torch.optim as optim
from torch.utils.data import DataLoader

from data.utils import draw_landmarks
from data.bbox_dataset import BBoxDataset
from data.align_dataset import AlignDataset
from sparse import loss, learning_rate

from models.saver import Saver
from models.dense201 import Dense201
from utils.metrics import Metrics
from utils.alignment import Align
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

if not os.path.exists(cfg.root):
    os.makedirs(cfg.root)
shutil.copy(args.config, cfg.root)
# config = {0: learning_rate.polynomial_decay(1e-8, 8000, 2e-5),
#           8001: learning_rate.exponential_decay(2e-5, 500, 0.993)}
# lr_gen = learning_rate.mix(config)
lr_gen = learning_rate.get_lr(cfg.lr)
# lr_gen = learning_rate.piecewise_constant([80000], [1e-5, 1e-6])
def adjust_learning_rate(optimizer):
    lr = lr_gen.get()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    metrics = Metrics().add_nme(0.5).add_auc(decay=0.5, step=0.001).add_loss(decay=0.5)

    writer = SummaryWriter(os.path.join(cfg.root, 'logs/train'))
    net = Dense201(num_classes=212)

    a = BBoxDataset('/data/icme/crop/data/picture',
                    '/data/icme/crop/data/landmark',
                    '/data/icme/train',
                    max_jitter=0)
    # a = AlignDataset('/data/icme/crop/data/picture',
    #                  '/data/icme/crop/data/landmark',
    #                  '/data/icme/crop/data/landmark',
    #                  '/data/icme/train',
    #                  Align('./cache/mean_landmarks.pkl', (224, 224), (0.2, 0.1),
    #                        ),# idx=list(range(51, 66))),
    #                  flip=True,
    #                  max_jitter=0,
    #                  max_radian=0
    #                  # ldmk_ids=list(range(51, 66))
    #                  )
    batch_iterator = iter(DataLoader(a, batch_size=cfg.batch_size, shuffle=True, num_workers=4))

    criterion = loss.get_criterion(cfg.loss)
    if cfg.device == all_pb2.GPU:
        net = net.cuda()
        criterion = criterion.cuda()
    # criterion = WingLoss(10, 2)
    optimizer = optim.Adam(net.parameters(), lr=0, weight_decay=cfg.weight_decay)
    # optimizer = optim.SGD(net.parameters(), lr=0, weight_decay=cfg.weight_decay, momentum=0.9)
    saver = Saver(os.path.join(cfg.root, 'ckpt'), ['model', 'opt', 'crit'], 10)
    start_iter = saver.s['model'].last_ckpt()

    if not start_iter is None:
        start_iter = int(start_iter.split('.')[0].split('-')[1])
        saver.load_last_ckpt({'model': net,
                              'crit': criterion,
                              'opt': optimizer})
        lr_gen.set_global_step(start_iter)
    else:
        start_iter = 0
    running_loss = 0.0
    batch_size = cfg.batch_size
    epoch_size = len(a) // batch_size
    epoch = start_iter // epoch_size + 1
    for iteration in range(start_iter, cfg.max_iter + 1):
        if iteration % epoch_size == 0:
            # create batch iterator
            a = BBoxDataset('/data/icme/crop/data/picture',
                            '/data/icme/crop/data/landmark',
                            '/data/icme/train',
                            max_jitter=0)
            batch_iterator = iter(DataLoader(a, batch_size,
                                                  shuffle=True, num_workers=4))
            epoch_size = len(a) // batch_size
            epoch += 1

        lr = adjust_learning_rate(optimizer)
        # load train data
        images, landmarks = next(batch_iterator)
        if cfg.device == all_pb2.GPU:
            images = images.cuda()
            landmarks = landmarks.cuda()

        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss = criterion(out, landmarks)
        # clip_grad_norm_(net.parameters(), 0.5)
        loss.backward()
        optimizer.step()
        # criterion.update(loss.cpu().data.numpy())
        if iteration % 200 == 0:
            net.eval()
            out = net(images)
            net.train()

            image = images.cpu().data.numpy()[0]
            gt = landmarks.cpu().data.numpy()
            pr = out.cpu().data.numpy()
            # 绿色的真实landmark
            image = draw_landmarks(image, gt[0], (0, 255, 0))
            # 红色的预测landmark
            image = draw_landmarks(image, pr[0], (0, 0, 255))
            image = image[::-1, ...]
            nme = metrics.nme.update(np.reshape(gt, (-1, gt.shape[1]//2, 2)), np.reshape(pr, (-1, gt.shape[1]//2, 2)))
            metrics.auc.update(nme)
            metrics.loss.update(loss)
            writer.add_scalar("watch/NME", metrics.nme.value * 100, iteration)
            writer.add_scalar("watch/AUC", metrics.auc.value * 100, iteration)
            writer.add_scalar("watch/loss", metrics.loss.value, iteration)
            writer.add_scalar("watch/learning_rate", lr, iteration)

            writer.add_image("result", image, iteration)
            writer.add_histogram("predictionx", out.cpu().data.numpy()[:, 0:212:2], iteration)
            model_state = net.state_dict()
            opt_state = optimizer.state_dict()
            crit_state = criterion.state_dict()
            saver.save({'model': model_state,
                        'crit': crit_state,
                        'opt': opt_state}, iteration)
            metrics.clear()

