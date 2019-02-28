import matplotlib.pyplot as plt
import numpy as np
from data import utils
import pwa
from skimage import io

#anotest输出为本文件输入,所以要考虑到anotest输出为0的情况
def save_pic_and_lmk(kk):
    shuru = kk
    if shuru == 0:
        e = 1  #return 0 输入为0,说明只有一张图
    else:
        lmk_dir = '/home/zhang/correctdata/data/landmark/'       #注意lmk文件的后缀为.txt
        pic_dir = '/home/zhang/correctdata/data/picture/'
        bbox_dir ='/home/zhang/correctdata/bbox/'
        length = len(shuru)
        for i in range(length):
            if i % 2 == 0:
                filename1 = pic_dir+shuru[i]
                '''filename2 = pic_dir+shuru[i+1]'''
                lmk1 = utils.read_mat(lmk_dir+shuru[i+1]+'.txt')
                lmk2 = utils.read_mat(lmk_dir+shuru[i]+'.txt')
                lmk = 0.5*lmk1 + 0.5*lmk2
                image = plt.imread(filename1)

                #图一的landmark归一化
                bbox1 = utils.read_bbox(bbox_dir+shuru[i+1]+'.rect')
                lmk1 = utils.norm_landmarks(lmk1, bbox1)
                image = image[bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]]

                #图一图二的插值图的landmark归一化
                bbox2 = utils.read_bbox(bbox_dir+shuru[i]+'.rect')
                lmk = utils.norm_landmarks(lmk, bbox2)
                out = pwa.pwa(image.copy(), lmk1.copy(), lmk.copy(), (256, 256))
                # 图片与lmk保存

                utils.save_landmarks(lmk*255, '/home/zhang/ming/save_for_lmk2/' + shuru[i][:-5] + "%d" % (200+i) + '.jpg' +'.txt')
                io.imsave('/home/zhang/ming/save_for_pic2/' + shuru[i][:-5] + "%d" % (200+i) + '.jpg', out)

if __name__ == '__main__':

    a = ['HELEN_1218567979_3_0.jpg', 'HELEN_1218567979_3_1.jpg', 'HELEN_1218567979_3_1.jpg', 'HELEN_1218567979_3_2.jpg', 'HELEN_1218567979_3_2.jpg', 'HELEN_1218567979_3_3.jpg']
    save_pic_and_lmk(a)




