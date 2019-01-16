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

nme0 = 0
num = 10
srcs = np.random.rand(num,68,2)
preds = np.random.rand(num,68,2)
# [n,a,b] = srcs.shape
# for i in range(n):
#     src = srcs[i,:,:]
#     pred = preds[i,:,:]
#     nme0 += nme(src,pred)
#
# nme = nme0/n
# print(nme)

alpha = np.arange(0.9,1.0,0.01)

def props(srcs,preds,alpha):
    props = np.array([])
    for alpha1 in alpha:
        n = srcs.shape[0]
        count = 0
        sum_points = n*srcs.shape[1]
        for i in range(n): #先算1张图的prop
            src = srcs[i,:,:]
            pred = preds[i,:,:]
            bbox = np.max(src, axis=-2) - np.min(src, axis=-2)
            d = np.sqrt(bbox[0] * bbox[1])
            nn = src.shape[0]
            for j in range(nn):
                src_p = src[j,:]
                pred_p = pred[j,:]
                dist = np.linalg.norm(src_p-pred_p)
                if dist/d < alpha1:
                    count = count+1
                    # print(count)
            prop = count/sum_points
        props = np.append(props,prop)
    return props

def auc(alpha,props):
    area = np.trapz(alpha,props)
    return area

props = props(srcs,preds,alpha)
auc = auc(alpha,props)
print(auc)