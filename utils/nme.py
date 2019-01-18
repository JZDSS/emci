import numpy as np


class NME(object):
    def __init__(self, decay=0.999):
        self.value = None
        self.decay = decay

    def update(self, G, P):
        """
        计算NME
        :param G:ground truth ,形状为(M, N, 2),M为图片数量,N为关键点数量
        :param P:预测值,同上
        :return:整个图集的NME,形状为M维数组,M为图片数量
        """
        M = G.shape[0]
        N = G.shape[1]
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
        D1 = D.sum(axis=1)
        D2 = D1 / S
        D3 = D2 / N  # N表示点的数目,D3是一个数组,其中的元素是每个图的NME
        E = np.sum(D3)
        F = E / M  # M表示图的数目,F是整个图集的NME

        if self.value is None:
            self.value = F
        else:
            # 指数滑动窗口
            self.value *= self.decay
            self.value += (1 - self.decay) * F
        return D3

    def clear(self):
        self.value = None
