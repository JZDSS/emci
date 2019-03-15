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
                 max_jitter=5,
                 img_format=None):
        super(BBoxDataset, self).__init__(img_dir, ldmk_dir, bin_dir, bins, phase, shape, img_format)
        # self.bboxes = [os.path.join(bbox_dir, f + '.rect') for f in self.file_list]
        self.max_jitter = max_jitter

    def __getitem__(self, item):
        image, landmarks = super(BBoxDataset, self).__getitem__(item)
        max_x = image.shape[1] - 1
        zero = np.zeros_like(image, np.float32)[:, :, 0]
        image = cv2.resize(image, self.shape)
        if self.phase == 'train':
            image, landmarks = utils.random_flip(image, landmarks, 0.5, max_x)
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)
        gts = []
        for i in range(106):
            x, y = landmarks[i]
            z = zero.copy()
            x, y = int(x) - 1, int(y) - 1
            z[y, x] = 255
            z = cv2.resize(z, self.shape, interpolation=cv2.INTER_AREA)
            z /= np.sum(z)
            gts.append(z)
        gts = np.stack(gts, axis=0)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, gts


if __name__ == '__main__':
    a = BBoxDataset('/data/icme/data/picture',
                    '/data/icme/data/landmark',
                    '/data/icme/train')
    for i in range(10000):
        im, l = a.__getitem__(i)
        print(i)