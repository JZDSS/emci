import numpy as np


def getalignedbbox(path):
    """
    输入对齐后的landmark,得到对齐后的bounding box
    :param path: 对齐后的landmark路径
    :return: 输出对齐后的bounding box ,形状为[xmin,ymin,xmax,ymax]
    """
    landmarks = []
    with open(path) as f:
        n = int(f.readline())
        for i in range(n):
            pt = f.readline().strip('\n').split(' ')
            pt = [int(float(p)) for p in pt]
            landmarks.append(pt)
        a = np.zeros((106,2))
        a += np.array(landmarks, dtype=np.float32)
        b = np.min((a), axis=0)
        c = np.max((a), axis=0)
        d = np.hstack((b,c))

    return np.array(d, dtype=np.float32)


if __name__ == '__main__':

    m = getalignedbbox('/home/zhang/aligned/landmark/AFW_134212_1_0.jpg.txt')
    print(m)
