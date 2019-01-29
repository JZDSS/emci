import sys
sys.path.append('/home/orion/Desktop/ecmi/emci')
import os
import matplotlib.pyplot as plt
from utils.pwa import pwa
from data import utils
from sklearn.externals import joblib
import numpy as np
from utils import alignment
from PIL import Image
import cv2
import time

#在/data下新建expand文件夹，再在expand文件夹下建立
#images，landmarks，picture_draws三个文件夹，在
#picture_draws文件夹下建立文件名为1～7的7个文件夹

reference = joblib.load('cache/mean_landmarks.pkl')
U = joblib.load('cache/u.pkl')
aligned = joblib.load('cache/aligned.pkl')
temp = (aligned - np.mean(aligned, axis=0))
joblib.dump(U, 'cache/u.pkl', compress=3)
pc = temp.dot(U[:, 0])
filenames = np.array(joblib.load('cache/filenames.pkl'))
aligner = alignment.Align('cache/mean_landmarks.pkl', margin=(0.2, 0.15))
idx = np.argsort(pc)


img_dir = '/data/icme/data/picture'
bbox_dir = '/data/icme/bbox'
landmark_dir = '/data/icme/data/landmark'
out_dir = '/data/expand/'


N = 7#每次选择的目标landmark数目，src_nums : dst_nums = 1 : N
stop = len(idx) - N

for i in range(0, stop):
    print('Nums:', stop - 1, 'Done:', i)
    id = list(idx[i: i + N + 1])#取N + 1张图片
    names = filenames[id]
    src_name = names[0]
    src_image = plt.imread(os.path.join(img_dir, src_name))
    src_landmark = utils.read_mat(os.path.join(landmark_dir, src_name + '.txt'))
    ali_src_image, ali_src_landmark, _ = aligner(src_image.copy(), src_landmark.copy())

    ali_src_landmark[:, 0] /= aligner.scale[0]
    ali_src_landmark[:, 1] /= aligner.scale[1]
    for j in range(1, N + 1):
        dst_name = names[j]
        dst_image = plt.imread(os.path.join(img_dir, dst_name))
        dst_landmark = utils.read_mat(os.path.join(landmark_dir, dst_name + '.txt'))
        ali_dst_image, ali_dst_landmark, _ = aligner(dst_image.copy(), dst_landmark.copy())

        ali_dst_landmark[:, 0] /= aligner.scale[0]
        ali_dst_landmark[:, 1] /= aligner.scale[1]

        ali_dst_landmark[0:33, :] = ali_src_landmark[0:33, :]
        out_landmark = ali_dst_landmark.copy()

        out = pwa(ali_src_image.copy(), ali_src_landmark.copy(), ali_dst_landmark.copy(), (128, 128))

        out_landmark[:, 0] *= aligner.scale[0]
        out_landmark[:, 1] *= aligner.scale[1]
        out_landmark = out_landmark.astype(int)
        out = (out * 255).astype(np.uint8)
        temp_out = out.copy()

        # plt.subplot(1, 3, 1)
        # plt.imshow(ali_src_image)
        # plt.subplot(1, 3, 2)
        # plt.imshow(ali_dst_image)
        # plt.subplot(1, 3, 3)
        # plt.imshow(out)
        # plt.axis((0, 128, 128, 0))
        # plt.show()

        for k in range(106):
            cv2.circle(temp_out, (out_landmark[k, 0], out_landmark[k, 1]), 0, (0, 255, 0))
        temp_out = cv2.resize(temp_out, (1024, 1024))
        # cv2.namedWindow('output', flags=cv2.WINDOW_NORMAL)
        # cv2.imshow('output', temp_out)
        # cv2.waitKey(0)
        # time.sleep(3)

        #创建新的名字
        temp_src_name = src_name[0: len(src_name) - 4]
        temp_dst_name = dst_name[0: len(dst_name) - 4]
        distort_image_name = temp_src_name + '_' + temp_dst_name + '.jpg'

        #保存扭曲图片
        imagefile = Image.fromarray(out).convert('RGB')
        imagefile.save(out_dir + 'images/' + distort_image_name)

        #保存picture_draws
        imagefile2 = Image.fromarray(temp_out).convert('RGB')
        imagefile2.save(out_dir + 'picture_draws/' + str(j) + '/' + distort_image_name)

        #保存landmarks
        utils.save_landmarks(out_landmark, out_dir + 'landmarks/' + distort_image_name + '.txt')