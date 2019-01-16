import numpy as np


def nme_calc(G, P):
    """
    计算每一张图的NME
    :param G: ground truth 形状为(M, N, 2),其中M为图片数量,N为关键点的数量
    :param P: 预测值 同上
    :return: 长度为M的数组,每个元素对应一张图的NME,其中M为图片的数量
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





def auc_calc(nme):

    a = np.argsort(nme)
    b = nme[a]
    n = nme.size
    p = np.zeros(9)
    s = np.zeros(8)

    for i in range(n):
        if nme[i] <= 0.01:
            p[1] += 1
        if (nme[i] <= 0.02) and (nme[i] > 0.01):
            p[2] += 1
        if (nme[i] <= 0.03) and (nme[i] > 0.02):
                p[3] += 1
        if (nme[i] <= 0.04) and (nme[i] > 0.03):
            p[4] += 1
        if (nme[i] <= 0.05) and (nme[i] > 0.04):
            p[5] += 1
        if (nme[i] <= 0.06) and (nme[i] > 0.05):
            p[6] += 1
        if (nme[i] <= 0.07) and (nme[i] > 0.06):
            p[7] += 1
        if (nme[i] <= 0.08) and (nme[i] > 0.07):
            p[8] += 1

    q = np.cumsum(p)
    k = q / n

    c = 0
    while (c < 8):
        s[c] = 0.01 * (k[c] + k[c + 1]) / 2
        c = c + 1

    auc = np.sum(s) / 0.08

    return auc


if __name__ == '__main__':
    nme = 0.1 * np.random.random(size=100000)
    print(auc_calc(nme))
    G = np.random.rand(4, 106, 2)  # ground truth (三维数组,M代表图的数量,N代表关键点数量)
    P = np.random.rand(4, 106, 2)  # 预测值
    print(nme_calc(G, P))
