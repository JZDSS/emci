import os
import cv2
import numpy as np
from data.face_dataset import FaceDataset
from data import utils

class BBoxDataset(FaceDataset):

    def __init__(self,
                 img_dir,
                 ldmk_dir,
                 bin_dir,
                 bins=[1,2,3,4,5,6,7,8,9,10,11],
                 phase='train',
                 shape=(224, 224),
                 max_jitter=0,
                 max_angle=0,
                 img_format=None):
        super(BBoxDataset, self).__init__(img_dir, ldmk_dir, bin_dir, bins, phase, shape, img_format)
        # self.bboxes = [os.path.join(bbox_dir, f + '.rect') for f in self.file_list]
        self.max_jitter = max_jitter
        self.max_rand = max_angle / 180 * np.pi

    def __getitem__(self, item):
        image, landmarks = super(BBoxDataset, self).__getitem__(item)

        H, W, _ = image.shape
        # landmarks /= [W, H]
        resize = \
            np.array([
                [self.shape[0] / W, 0, 0],
                [0, self.shape[1] / H, 0],
                [0, 0, 1]],
                dtype=np.float32)
        if self.phase == 'train':
            theta = np.random.uniform(-self.max_rand, self.max_rand)
            dx = np.random.uniform(-self.max_jitter, self.max_jitter)
            dy = np.random.uniform(-self.max_jitter, self.max_jitter)
            dice = np.random.uniform(0, 1)
            if dice > 0.5:
                flip = \
                    np.array([
                        [1, 0, self.shape[1]],
                        [0, 1, 0],
                        [0, 0, 1]],
                        dtype=np.float32
                    ) @ \
                    np.array([
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]],
                        dtype=np.float32)
            else:
                flip = np.identity(3, np.float32)
            center = \
                np.array([
                    [1, 0, -self.shape[1] / 2],
                    [0, 1, -self.shape[1] / 2],
                    [0, 0, 1]],
                    dtype=np.float32
                )

            c = np.cos(theta)
            s = np.sin(theta)
            rotation = \
                np.array([
                    [c, s, 0],
                    [-s, c, 0],
                    [0, 0, 1]],
                    dtype=np.float32
                )

            decenter = \
                np.array([
                    [1, 0, self.shape[1] / 2],
                    [0, 1, self.shape[1] / 2],
                    [0, 0, 1]],
                    dtype=np.float32
                )
            rotation = decenter @ rotation @ center

            translate = \
                np.array([
                    [1, 0, dx],
                    [0, 1, dy],
                    [0, 0, 1]],
                    dtype=np.float32
                )
            T = translate @ rotation @ flip @ resize
        else:
            T = resize


        image = cv2.warpAffine(image, T[:2, :], self.shape)
        ones = np.ones(len(landmarks), dtype=np.float32)
        x = np.c_[landmarks, ones]
        landmarks = x @ np.transpose(T[:2, :])
        if self.phase == 'train':
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        landmarks[:, 0] /= self.shape[1]
        landmarks[:, 1] /= self.shape[0]
        return image, np.reshape(landmarks, (-1))


if __name__ == '__main__':
    a = BBoxDataset('/data/icme/crop/data/picture',
                    '/data/icme/crop/data/landmark',
                    '/data/icme/train',
                    img_format='png')
    import matplotlib.pyplot as plt
    for i in range(100):
        image, landmark = a.__getitem__(i)
        image = np.transpose(image, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
        landmark = np.reshape(landmark, [-1, 2])
        plt.imshow(image)
        plt.scatter(landmark[:, 0], landmark[:, 1])
        plt.show()