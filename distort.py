import os
import matplotlib.pyplot as plt
from utils import pwa
from data import utils
from sklearn.externals import joblib
import numpy as np
from utils import alignment


reference = joblib.load('cache/mean_landmarks.pkl')
U = joblib.load('cache/u.pkl')
aligned = joblib.load('cache/aligned.pkl')
temp = (aligned - np.mean(aligned, axis=0))
covariance = 1.0 / len(aligned) * temp.T.dot(temp)
joblib.dump(U, 'cache/u.pkl', compress=3)
pc = temp.dot(U[:, 0])
filenames = np.array(joblib.load('cache/filenames.pkl'))
img_dir = '/data/icme/data/picture'
bbox_dir = '/data/icme/bbox'
landmark_dir = '/data/icme/data/landmark'


aligner = alignment.Align('cache/mean_landmarks.pkl')
idx = np.argsort(pc)

for i in range(5000, len(idx) - 5):
    id = list(idx[i: i + 5])
    names = filenames[id]
    for j in range(1, 5):
        src = names[0]
        dst = names[j]
        image = plt.imread(os.path.join(img_dir, src))
        bbox = utils.read_bbox(os.path.join(bbox_dir, src + '.rect'))
        src = utils.read_mat(os.path.join(landmark_dir, src + '.txt'))
        image, src, _ = aligner(image, src, bbox)
        src[:, 0] /= aligner.scale[0]
        src[:, 1] /= aligner.scale[1]
        image2 = plt.imread(os.path.join(img_dir, dst))
        bbox = utils.read_bbox(os.path.join(bbox_dir, dst + '.rect'))
        dst = utils.read_mat(os.path.join(landmark_dir, dst + '.txt'))
        image2, dst, _ = aligner(image2, dst, bbox)
        dst[:, 0] /= aligner.scale[0]
        dst[:, 1] /= aligner.scale[1]
        dst[0:34, :] = src[0:34, :]
        out = pwa.pwa(image, src, dst, (128, 128))


        plt.subplot(1, 3, 1)
        plt.imshow(image)

        plt.subplot(1, 3, 2)
        plt.imshow(image2)

        plt.subplot(1, 3, 3)
        plt.imshow(out)

        # plt.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
        plt.axis((0, 128, 128, 0))
        plt.show()
    a = 1

out_dir = '/data/icme-g'
root = '/data/icme/train'

for i in range(1, 12):
    pose = '%d' % i
    curr = os.path.join(root, pose)
    imgs = os.listdir(curr)
    for src in imgs:
        for dst in imgs:
            image = plt.imread(os.path.join(curr, src))
            bbox = utils.read_bbox(os.path.join(bbox_dir, src + '.rect'))
            src = utils.read_mat(os.path.join(landmark_dir, src + '.txt'))
            src = utils.norm_landmarks(src, bbox)
            image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]

            bbox = utils.read_bbox(os.path.join(bbox_dir, dst + '.rect'))
            dst = utils.read_mat(os.path.join(landmark_dir, dst + '.txt'))
            dst = utils.norm_landmarks(dst, bbox)

            out = pwa.pwa(image, src, dst, (128, 128))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)

            plt.imshow(out)
            # plt.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
            plt.axis((0, 128, 128, 0))
            plt.show()