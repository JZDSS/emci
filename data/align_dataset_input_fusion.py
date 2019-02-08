import os
import cv2
import numpy as np
from data.face_dataset import FaceDataset
from data import utils

class AlignFusionDataset(FaceDataset):

    def __init__(self,
                 img_dir,
                 gt_ldmk_dir,
                 al_ldmk_dir,
                 bin_dir,
                 aligner,
                 bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 phase='train',
                 shape=(224, 224),
                 flip=True,
                 max_jitter=4,
                 kernel_size=15, sigma=5):
        super(AlignFusionDataset, self).__init__(img_dir, gt_ldmk_dir, bin_dir, bins, phase, shape)
        self.aligner = aligner
        self.algin_ldmk = [os.path.join(al_ldmk_dir, f + '.txt') for f in self.file_list]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.idxs = [list(range(32)),
                     [51, 52, 53, 60],
                     list(range(55, 66)),
                     list(range(33, 38)),
                     [33, 41, 40, 39, 38, 37],
                     list(range(42, 47)),
                     [46, 47, 48, 49, 50, 42],
                     list(range(66, 71)),
                     [66, 73, 72, 71, 70],
                     list(range(75, 80)),
                     [75, 82, 81, 80, 79],
                     [84, 85, 86, 87, 88, 89, 90],
                     [90, 100, 99, 98, 97, 96, 84],
                     [84, 103, 102, 101, 90],
                     [90, 91, 92, 93, 94, 95, 84]]
        self.flip = flip
        self.max_jitter = max_jitter

    def __getitem__(self, item):
        image, gt_landmarks = super(AlignFusionDataset, self).__getitem__(item)
        pr_landmarks = utils.read_mat(self.algin_ldmk[item])
        image, pr_landmarks, t = self.aligner(image, pr_landmarks,
                                              noise=np.random.uniform(-self.max_jitter, self.max_jitter, 2))
        gt_landmarks = gt_landmarks @ t[0:2, :] + t[2, :]

        # start_y = np.random.randint(0, self.aligner.scale[0] - self.shape[0] + 1)
        # start_x = np.random.randint(0, self.aligner.scale[1] - self.shape[1] + 1)
        # gt_landmarks[:, 0] -= start_x
        # gt_landmarks[:, 1] -= start_y
        # pr_landmarks[:, 0] -= start_x
        # pr_landmarks[:, 1] -= start_y

        gt_landmarks[:, 0] /= self.shape[1]
        gt_landmarks[:, 1] /= self.shape[0]

        # image = image[start_y:start_y + self.shape[0], start_x:start_x + self.shape[1]]
        if self.phase == 'train':
            if self.flip:
                a = np.random.uniform(0, 1, 1)
                if a < 0.5:
                    image = cv2.flip(image, 1)
                    gt_landmarks = utils.landmark_flip(gt_landmarks)
                    pr_landmarks = utils.landmark_flip(pr_landmarks, max_x=self.shape[1])
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)

        kernel_size = self.kernel_size
        sigma = self.sigma
        heatmaps = []

        for idx in self.idxs:
            img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
            draw_curve(img, pr_landmarks, idx)
            heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        inputs = []
        for i in range(len(heatmaps)):
            heatmaps[i] = heatmaps[i].astype(np.float32) / np.max(heatmaps[i])
            inputs.append(image * np.expand_dims(heatmaps[i], -1))

        inputs.append(image)
        cv2.imshow(",", image)
        cv2.waitKey(1)
        for i in inputs:
            cv2.imshow("", i.astype(np.uint8))
            cv2.waitKey(0)
        inputs = np.concatenate(inputs, axis=-1)
        inputs = np.transpose(inputs, [2, 0, 1])
        heatmaps = np.stack(heatmaps, axis=2)
        heatmaps = np.transpose(heatmaps, [2, 0, 1])

        return inputs, np.reshape(gt_landmarks, (-1)), heatmaps


def draw_curve(img, landmarks, idx):
    for j in range(len(idx) - 1):
        cv2.line(img, (landmarks[idx[j], 0], landmarks[idx[j], 1]),
                 (landmarks[idx[j + 1], 0], landmarks[idx[j + 1], 1]), (255, 255, 255),
                     thickness=4)

