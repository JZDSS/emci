import numpy as np
M = 4
N = 106
G = np.random.rand(M, N, 2)  # ground truth (三维数组,M代表图的数量,N代表关键点数量)
P = np.random.rand(M, N, 2)  # 预测值

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


print(D3)
print(F)