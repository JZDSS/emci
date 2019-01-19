import sys
sys.path.append('/home/orion/Desktop/ecmi/emci')
from data.pose_dataset import PoseDataset
import torch
from torch.utils.data import DataLoader
from layers.module.wing_loss import WingLoss
from models.resnet50 import ResNet50
from models.resnet18 import ResNet18
import numpy as np
from utils.metrics import Metrics
import math
import matplotlib.pyplot as plt

#criterion = WingLoss(10, 0.5)
batch_size = 4
metrics = Metrics().add_nme(0.9).add_auc(decay=0.9).add_loss(decay=0.9)


if __name__ == "__main__":
    current = None
    net = ResNet18().cuda()
    net.load_state_dict(torch.load('./ckpt/model-9200.pth'))
    net.eval()
    NME = np.empty(11)
    AUC = np.empty(11)
    for pose in range(11):
        a = PoseDataset("/home/orion/Desktop/ecmi/emci/data/icme", "/home/orion/Desktop/ecmi/emci/data/icme/valid",
                        phase='eval', pose=pose)
        batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))
        iterations = math.ceil(len(a) / batch_size)
        images = None
        landmarks = None
        gt = None
        pr = None
        interation = 0
        nmes = np.empty(iterations)
        aucs = np.empty(iterations)
        while True:
            try:
                images, landmarks = next(batch_iterator)
                images = images.cuda()
                landmarks = landmarks.cuda()
                with torch.no_grad():
                    out = net.forward(images)
                #loss = criterion(out, landmarks)
                pr = out.cpu().data.numpy()
                gt = landmarks.cpu().data.numpy()
                nme = metrics.nme.update(np.reshape(gt, (-1, 106, 2)), np.reshape(pr, (-1, 106, 2)))
                auc = metrics.auc.update(nme)
                nmes[interation] = np.mean(nme)
                aucs[interation] = auc
                #print('nme', nme, 'auc', auc, nmes[interation], aucs[interation])
                #metrics.loss.update(loss.item())
                metrics.clear()
                interation += 1
            except StopIteration:
                break
        print('pose:', pose, 'len:', len(a))
        NME[pose] = np.mean(nmes)
        AUC[pose] = np.mean(aucs)
    x = np.linspace(1, 11, 11)
    plt.plot(x, NME, )
    plt.title('NME')
    plt.xlabel = 'pose'
    plt.ylabel = 'NME'
    plt.show()

    plt.plot(x, AUC, )
    plt.title('AUC')
    plt.xlabel = 'pose'
    plt.ylabel = 'AUC'
    plt.show()