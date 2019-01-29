import sys
from data.align_dataset import AlignDataset
from utils.alignment import Align
import torch
from torch.utils.data import DataLoader
from layers.module.wing_loss import WingLoss
from models.resnet50 import ResNet50
from models.resnet18 import ResNet18
from models.dense201 import Dense201
import numpy as np
from utils.metrics import Metrics
import matplotlib.pyplot as plt
import math

#criterion = WingLoss(10, 0.5)
metrics = Metrics().add_nme(0.9).add_auc(decay=0.9).add_loss(decay=0.9)

if __name__ == "__main__":
    net = Dense201().cuda()
    net.load_state_dict(torch.load('../backup/align-jitter.pth'))
    net.eval()
    PR_list = []
    GT_list = []
    for pose in range(1, 12):
        a = AlignDataset('/data/icme/data/picture',
                         '/data/icme/data/landmark',
                         '/data/icme/data/pred_landmark',
                         '/data/icme/valid',
                         Align('../cache/mean_landmarks.pkl', (224, 224), (0.15, 0.1)),
                         bins=[pose], phase='eval')
        batch_iterator = iter(DataLoader(a, batch_size=1, shuffle=False, num_workers=0))
        images = None
        landmarks = None
        gt = None
        pr = None
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
                PR_list.append(pr)
                GT_list.append(gt)
            except StopIteration:
                break
        print('pose:', pose, 'len:', len(a))

    def calc_nme(G, P):
        GMax = np.max(G, axis=1)
        GMin = np.min(G, axis=1)
        WH = GMax - GMin
        W = WH[:, 0]
        H = WH[:, 1]
        S = np.sqrt(W * H)  # 所有图中ground truth 的面积构成的数组

        A = G - P
        B = A ** 2
        C = B.sum(axis=2)
        D = np.sqrt(C)
        D /= S
        return D

    nmes = []
    # for gt, pr in zip(GT_list, PR_list):
    for i in range(len(GT_list)):
        gt = GT_list[i]
        pr = PR_list[i]
        nme = calc_nme(np.reshape(gt, (1, 106, 2)), np.reshape(pr, (1, 106, 2)))
        nmes.append(nme)
    nmes = np.concatenate(nmes, axis=0)
    nmes_ = np.mean(nmes, axis=0)
    plt.subplot(1, 2, 1)
    x = np.linspace(1, 106, 106)
    plt.plot(x, nmes_)
    plt.title('NME')
    aucs = []
    for i in range(106):
        nme = nmes[:, i]
        aucs.append(metrics.auc.update(nme))
        metrics.clear()
    plt.subplot(1,2,2)
    plt.plot(x, aucs)
    plt.title('AUC')
    plt.show()
    a = 1

    # M =len(GT_list)
    # G = []
    # P = []
    # for i in range(M):
    #     G.append(np.reshape(GT_list[i], (106, 2), order='F'))
    #     P.append(np.reshape(PR_list[i], (106, 2), order='F'))
    # GT_array = np.array(G)
    # PR_array = np.array(P)
    #
    # N = GT_array.shape[1]#关键点数
    #
    # NME = np.zeros(N)
    # AUC = np.zeros(N)
    # for n in range(N):
    #     GT_list = []
    #     PR_list = []
    #     for m in range(M):
    #         GT_list.append(GT_array[m, n, :])
    #         PR_list.append(PR_array[m, n, :])
    #     GT = np.array(GT_list)
    #     PR = np.array(PR_list)
    #     GT = np.reshape(GT, (1, -1, 2))
    #     PR = np.reshape(PR, (1, -1, 2))
    #     nmes = metrics.nme.update(GT, PR)
    #     auc = metrics.auc.update(nmes)
    #     nme = np.mean(nmes)
    #     NME[n] = nme
    #     AUC[n] = auc
    #     metrics.clear()
    #
    # print("NME:", NME)
    # print("AUC:", AUC)
    # x = np.linspace(1, 106, 106)
    # plt.plot(x, NME, )
    # plt.title('NME')
    # plt.xlim((0, 107))
    # plt.grid()
    # plt.show()
    #
    # plt.plot(x, AUC, )
    # plt.title('AUC')
    # plt.xlim((0, 107))
    # plt.grid()
    # plt.show()