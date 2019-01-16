import numpy as np
import os
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from face_dataset import utils
np.random.seed(0)
os.system('mkdir cache')


def procrustes(x, y):
    """
    求xT=y的变换矩阵T
    :param x: 形状为(N, 3)，其中前两列为坐标，第三列全1
    :param y: 形状为(N, 2), N个点的坐标
    :return: T的最小二乘解
    """
    # 奇异分解
    u, sigma, vt = np.linalg.svd(x)
    s = np.zeros((106, 3), dtype=np.float32)
    for i in range(sigma.shape[0]):
        s[i, i] = 1/sigma[i]
    # M-P逆
    xp = vt.T @ s.T @ u.T
    # 返回T
    return xp@y


def main():
    def get_id(name):
        t = name.split('_')[0:2]
        return t[0] + t[1]

    # 我偷懒了，所以最好不要写成'/home/yqi/data/icme/'
    root_dir = '/home/yqi/data/icme'

    lamdmark_dir = os.path.join(root_dir, 'data/landmark')
    image_dir = os.path.join(root_dir, 'data/picture')
    bbox_dir = os.path.join(root_dir, 'bbox')

    try:
        filenames = joblib.load('cache/filenames.pkl')
        norm_landmarks = joblib.load('cache/norm_landmarks.pkl')
        mean_landmarks = joblib.load('cache/mean_landmarks.pkl')
        bboxes = joblib.load('cache/bboxes.pkl')
        split = joblib.load('cache/split.pkl')
    except:

        filenames = os.listdir(image_dir)
        norm_landmarks = []
        bboxes = []
        split = {}
        for filename in filenames:
            id = get_id(filename)
            if np.random.uniform(0, 1) < 0.8:
                split[id] = 'train'
            else:
                split[id] = 'valid'
            landmark_path = os.path.join(lamdmark_dir, filename + '.txt')
            bbox_path = os.path.join(bbox_dir, filename + '.rect')
            bbox = utils.read_bbox(bbox_path)
            landmarks = utils.read_landmarks(landmark_path)
            landmarks = utils.norm_landmarks(landmarks, bbox)
            norm_landmarks.append(landmarks)
            bboxes.append(bbox)
        norm_landmarks = np.stack(norm_landmarks, axis=0)
        mean_landmarks = np.mean(norm_landmarks, axis=0)
        joblib.dump(norm_landmarks, 'cache/norm_landmarks.pkl', compress=3)
        joblib.dump(mean_landmarks, 'cache/mean_landmarks.pkl', compress=3)
        joblib.dump(filenames, 'cache/filenames.pkl', compress=3)
        joblib.dump(bboxes, 'cache/bboxes.pkl', compress=3)
        joblib.dump(split, 'cache/split.pkl', compress=3)
    # for i in range(106):
    #     plt.scatter(mean_landmarks[i, 0], mean_landmarks[i, 1])
    # plt.show()
    try:
        transform_matrix = joblib.load('cache/transform_matrix.pkl')
        aligned = joblib.load('cache/aligned.pkl')
    except:
        transform_matrix = []
        aligned = []
        for i, filename in enumerate(filenames):
            curr = norm_landmarks[i, :]
            one = np.ones(shape=(106, 1))
            curr = np.concatenate((curr, one), axis=1)
            t = procrustes(curr, mean_landmarks)
            transform_matrix.append(t)
            aligned.append(np.reshape(curr@t, (-1)))
        joblib.dump(transform_matrix, 'cache/transform_matrix.pkl', compress=3)
        joblib.dump(aligned, 'cache/aligned.pkl', compress=3)
    temp = (aligned - np.mean(aligned, axis=0))
    covariance = 1.0 / len(aligned) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    pc = temp.dot(U[:, 0])

    plt.hist(pc,bins=11)
    plt.show()
    for i, filename in enumerate(filenames):
        img_path = os.path.join(image_dir, filename)
        if pc[i] > 0.793:
            n = '1'
        elif pc[i] > 0.615:
            n = '2'
        elif pc[i] > 0.44:
            n = '3'
        elif pc[i] > 0.26:
            n = '4'
        elif pc[i] > 0.087:
            n = '5'
        elif pc[i] > -0.0913:
            n = '6'
        elif pc[i] > -0.264:
            n = '7'
        elif pc[i] > -0.448:
            n = '8'
        elif pc[i] > -0.62:
            n = '9'
        elif pc[i] > -0.79:
            n = '10'
        else:
            n = '11'
        id = get_id(filename)
        cmd = 'ln -s %s %s/%s/%s/%s' % (img_path, root_dir, split[id], n, filename)
        os.system(cmd)


if __name__ == '__main__':
    main()
