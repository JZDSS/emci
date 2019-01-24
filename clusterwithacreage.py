import numpy as np
import os
import matplotlib.pyplot as plt
from data import utils

def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()


def main():
    def get_id(name):
        t = name.split('_')[0:2]
        return t[0] + t[1]


    root_dir = 'D:\icmedata\correctdata\\'

    lamdmark_dir = os.path.join(root_dir, 'data/landmark')
    image_dir = os.path.join(root_dir, 'data/picture')
    bbox_dir = os.path.join(root_dir, 'bbox')
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
        landmarks = utils.read_mat(landmark_path)
        landmarks = utils.norm_landmarks(landmarks, bbox)
        norm_landmarks.append(landmarks)
        bboxes.append(bbox)
    norm_landmarks = np.stack(norm_landmarks, axis=0)
    mean_landmarks = np.mean(norm_landmarks, axis=0)
    # for i in range(106):
    #     plt.scatter(mean_landmarks[i, 0], mean_landmarks[i, 1])
    # plt.show()
    target=[]
    for i, filename in enumerate(filenames):
        curr = norm_landmarks[i, :]
        y = curr[1::2]
        y_max = np.max(y)
        y_min = np.min(y)
        x = curr[::2]
        x_max = np.max(x)
        x_min = np.min(x)
        chang = x_max - x_min
        kuan = y_max - y_min
        Slandmark = chang * kuan
        #print((Slandmark))
        #print(bboxes[i])
        # bbox_tempt = np.array(bboxes)
        # Sbbox = (bbox_tempt[:, 2] - bbox_tempt[:, 0]) * (bbox_tempt[:, 3] - bbox_tempt[:, 1])
        # print(Sbbox[i])
        #landmark就是基于bbox做的归一化在untils。norm——landmark所以就不用求Sbbox
        target.append(Slandmark)

    draw_hist(target, 'acreage title', 'SL/SB', 'amount', 0, 1, 0, 3000)

    for i, filename in enumerate(filenames):
        img_path = os.path.join(image_dir, filename)
        if target[i] > 0.8:
            n = 's1'
        elif target[i] > 0.75:
            n = 's2'
        elif target[i] > 0.7:
            n = 's3'
        elif target[i] > 0.64:
            n = 's4'
        elif target[i] > 0.6:
            n = 's5'
        elif target[i] > 0.54:
            n = 's6'
        elif target[i] > 0.5:
            n = 's7'
        else:
            n = 's8'
        id = get_id(filename)
        cmd = 'ln -s %s %s/%s/%s/%s' % (img_path, root_dir, split[id], n, filename)
        os.system(cmd)



if __name__ == '__main__':
    main()
