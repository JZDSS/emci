import os
import cv2
import numpy as np
from data.face_dataset import FaceDataset
from data import utils

class AlignDataset(FaceDataset):

    def __init__(self,
                 img_dir,
                 gt_ldmk_dir,
                 al_ldmk_dir,
                 bin_dir,
                 aligner,
                 bins=[1,2,3,4,5,6,7,8,9,10,11],
                 phase='train',
                 shape=(224, 224)):
        super(AlignDataset, self).__init__(img_dir, gt_ldmk_dir, bin_dir, bins, phase, shape)
        self.aligner = aligner
        self.algin_ldmk = [os.path.join(al_ldmk_dir, f + '.txt') for f in self.file_list]

    def __getitem__(self, item):
        image, landmarks = super(AlignDataset, self).__getitem__(item)
        al_ldmk = utils.read_mat(self.algin_ldmk[item])
        image, _, t = self.aligner(image, al_ldmk)
        landmarks = landmarks @ t[0:2, :] + t[2, :]
        landmarks[:, 0] /= self.aligner.scale[1]
        landmarks[:, 1] /= self.aligner.scale[0]
        if self.phase == 'train':
            image, landmarks = utils.random_flip(image, landmarks, 0.5)
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)
        color = (255, 255, 255)
        # 脸部外轮廓
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(32):
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic1 = cv2.GaussianBlur(img, (11, 11), 3)
        # 鼻梁
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(3):
            j = j + 51
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic2 = cv2.GaussianBlur(img, (11, 11), 3)
        # 鼻子下轮廓
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(10):
            j = j + 55
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic3 = cv2.GaussianBlur(img, (11, 11), 3)
        # 左眉
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(8):
            j = j + 33
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic4 = cv2.GaussianBlur(img, (11, 11), 3)
        # 右眉
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(8):
            j = j + 42
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic5 = cv2.GaussianBlur(img, (11, 11), 3)
        # 左眼
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(7):
            j = j + 66
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic6 = cv2.GaussianBlur(img, (11, 11), 3)
        # 右眼
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(7):
            j = j + 75
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic7 = cv2.GaussianBlur(img, (11, 11), 3)
        # 嘴唇
        img = np.zeros((128, 128, 3), np.uint8)
        for j in range(19):
            j = j + 84
            cv2.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                    thickness=2)  # thickness
        pic8 = cv2.GaussianBlur(img, (11, 11), 3)

        image = cv2.resize(image, self.shape)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        heatmapfusion1 = pic1 * image
        heatmapfusion2 = pic2 * image
        heatmapfusion3 = pic3 * image
        heatmapfusion4 = pic4 * image
        heatmapfusion5 = pic5 * image
        heatmapfusion6 = pic6 * image
        heatmapfusion7 = pic7 * image
        heatmapfusion8 = pic8 * image
        image = np.append(heatmapfusion1,heatmapfusion2,heatmapfusion3,heatmapfusion4,heatmapfusion5,heatmapfusion6,heatmapfusion7,heatmapfusion8)


        image = cv2.resize(image, self.shape)
        return image, np.reshape(landmarks, (-1))


if __name__ == '__main__':
    from utils.alignment import Align
    from torch.utils.data import DataLoader

    a = AlignDataset('/data/icme/data/picture',
                     '/data/icme/data/landmark',
                     '/data/icme/data/landmark',
                     '/data/icme/valid',
                     Align('../cache/mean_landmarks.pkl', (224, 224), (0.15, 0.05)))
    batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))
    while True:
        images, landmarks = next(batch_iterator)
        print(images.shape)



