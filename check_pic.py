import matplotlib.pyplot as plt
import numpy as np
from data import utils
import os

lmk_input = '/home/zhang/ming/save_for_lmk/'
pic_input = '/home/zhang/ming/save_for_pic/'
pic_filename = os.listdir(pic_input)
lmk_filename = os.listdir(lmk_input)

for img_name in pic_filename:
    img = plt.imread(pic_input+img_name)
    out = (np.transpose(img, (2, 0, 1))).astype(np.uint8)
    lmk = utils.read_mat(lmk_input+img_name+'.txt')
    output = utils.draw_landmarks(out, lmk/255, (255, 0, 0))
    output = np.transpose(output, (1, 2, 0))
    plt.imshow(output)
    plt.show()