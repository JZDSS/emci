from data.pose_dataset import PoseDataset
from data.pose_aligned_dataset import PoseAlignedDataset
import torch
from torch.utils.data import DataLoader
from models.resnet18 import ResNet18
from models.dense201 import Dense201
import numpy as np
from utils.metrics import Metrics
import matplotlib.pyplot as plt

metrics = Metrics().add_nme(0.9).add_auc(decay=0.9).add_loss(decay=0.9)

if __name__ == "__main__":
    current = None
    net = Dense201().cuda()
    net.load_state_dict(torch.load('../backup/aligned_densenet-166400.pth'))
    net.eval()
    NME = np.zeros(11)
    AUC = np.zeros(11)
    for pose in range(1, 12):
        a = PoseAlignedDataset("/data/icme", "/data/icme/valid",
                        phase='eval', pose=pose)
        batch_iterator = iter(DataLoader(a, batch_size=1, shuffle=False, num_workers=1))
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
        NME[pose - 1] = np.mean(nme)
        AUC[pose - 1] = auc
        metrics.clear()
        print('pose:', pose, 'len:', len(a))
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