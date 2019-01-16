import numpy as np


def nme_calc(G, P):
    """
    计算每一张图的NME,输出长度为M的数组,其中M为图片的数量
    :param G: ground truth 形状为(M, N, 2),其中M为图片数量,N为关键点的数量
    :param P: 预测值 同上
    :return: 整个图片集的NME
    """
    M = G.shape[0]
    N = G.shape[1]
    GMax = np.max(G, axis=1)
    GMin = np.min(G, axis=1)
    WH = GMax - GMin
    W = WH[:, 0]
    H = WH[:, 1]
    S = np.sqrt(W*H)   # 所有图中ground truth 的面积构成的数组

    A = G - P
    B = A**2
    C = B.sum(axis=2)
    D = np.sqrt(C)
    D1 = D.sum(axis=1)
    D2 = D1 / S
    D3 = D2 / N    # N表示点的数目,D3是一个数组,其中的元素是每个图的NME
    E = np.sum(D3)
    F = E / M      # M表示图的数目,F是整个图集的NME

    return D3


if __name__ == '__main__':
    G = np.random.rand(4, 106, 2)  # ground truth (三维数组,M代表图的数量,N代表关键点数量)
    P = np.random.rand(4, 106, 2)  # 预测值
    print(nme_calc(G, P))