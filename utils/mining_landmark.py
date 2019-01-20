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
import matplotlib.pyplot as plt
import math

#criterion = WingLoss(10, 0.5)
metrics = Metrics().add_nme(0.9).add_auc(decay=0.9).add_loss(decay=0.9)

if __name__ == "__main__":
    net = ResNet18().cuda()
    net.load_state_dict(torch.load('./ckpt/model-9200.pth'))
    net.eval()
    PR_list = []
    GT_list = []
    for pose in range(11):
        a = PoseDataset("/home/orion/Desktop/ecmi/emci/data/icme", "/home/orion/Desktop/ecmi/emci/data/icme/valid",
                        phase='eval', pose=pose)
        index = int(a.get_index())
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
        print('pose:', pose + 1, 'len:', len(a), 'index:', index)

    M =len(GT_list)
    K = int(M / 106) * 106#舍弃最后几张的图片以便能被106整除
    G = []
    P = []
    for i in range(M):
        G.append(np.reshape(GT_list[i], (106, 2), order='F'))
        P.append(np.reshape(PR_list[i], (106, 2), order='F'))
    GT_array = np.array(G)
    PR_array = np.array(P)

    N = GT_array.shape[1]#关键点数
    # temp = math.ceil(float(M / 106))
    # num = temp * 106 - M


    NME = np.zeros(N)
    AUC = np.zeros(N)
    for n in range(N):
        GT_list = []
        PR_list = []
        for m in range(M):
            GT_list.append(GT_array[m, n, :])
            PR_list.append(PR_array[m, n, :])
        GT_list = GT_list[0: K]
        PR_list = PR_list[0: K]
        GT = np.array(GT_list)
        PR = np.array(PR_list)
        #mean = (GT.mean(axis=0) + PR.mean(axis=0)) / 2
        #mean_array = np.array([mean] * num)
        # GT = np.r_[GT, mean_array]
        # PR = np.r_[PR, mean_array]
        GT = np.reshape(GT, (-1, 106, 2))
        PR = np.reshape(PR, (-1, 106, 2))
        nmes = metrics.nme.update(GT, PR)
        auc = metrics.auc.update(nmes)
        nme = np.mean(nmes)
        NME[n] = nme
        AUC[n] = auc
        metrics.clear()

    x = np.linspace(1, 106, 106)
    plt.plot(x, NME, )
    plt.title('NME')
    plt.xlim((0, 107))
    plt.grid()
    plt.show()

    plt.plot(x, AUC, )
    plt.title('AUC')
    plt.xlim((0, 107))
    plt.grid()
    plt.show()