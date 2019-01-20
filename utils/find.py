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
    NME = np.zeros(11)
    AUC = np.zeros(11)
    for pose in range(11):
        a = PoseDataset("/home/orion/Desktop/ecmi/emci/data/icme", "/home/orion/Desktop/ecmi/emci/data/icme/valid",
                        phase='eval', pose=pose)
        index = int(a.get_index())
        batch_iterator = iter(DataLoader(a, batch_size=1, shuffle=False, num_workers=0))
        iterations = math.ceil(len(a) / batch_size)
        images = None
        landmarks = None
        gt = None
        pr = None
        pr_list = []
        gt_list = []
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
                pr_list.append(pr)
                gt_list.append(gt)
            except StopIteration:
                break
        gt_array = np.reshape(np.array(gt_list), (-1, 106, 2))
        pr_array = np.reshape(np.array(pr_list), (-1, 106, 2))
        nme = metrics.nme.update(gt_array, pr_array)
        auc = metrics.auc.update(nme)
        NME[index - 1] = np.mean(nme)
        AUC[index - 1] = auc
        metrics.clear()
        print('pose:', pose + 1, 'len:', len(a), 'index:', index)
        # print('nme', nme, 'auc', auc, nmes[interation], aucs[interation])
        # metrics.loss.update(loss.item())

    x = np.linspace(1, 11, 11)
    plt.plot(x, NME, )
    plt.title('NME')
    plt.xlim((0, 12))
    plt.grid()
    plt.show()

    plt.plot(x, AUC, )
    plt.title('AUC')
    plt.xlim((0, 12))
    plt.grid()
    plt.show()