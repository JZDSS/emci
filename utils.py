import numpy as np

#srcs.shape = (n,68,2)
#preds.shape = (n,68,2)
#1张图
def nme(src,pred):
    bbox = np.max(src, axis= -2)-np.min(src,axis= -2)
    d = np.sqrt(bbox[0]*bbox[1])
    f_norm = np.linalg.norm(src-pred)
    error = f_norm/d
    return error


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
