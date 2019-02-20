import os
import cv2
import numpy as np
from data.face_dataset import FaceDataset
from data import utils

class BBoxDataset(FaceDataset):

    def __init__(self,
                 img_dir,
                 ldmk_dir,
                 bbox_dir,
                 bin_dir,
                 bins=[1,2,3,4,5,6,7,8,9,10,11],
                 phase='train',
                 shape=(224, 224),
                 max_jitter=5):
        super(BBoxDataset, self).__init__(img_dir, ldmk_dir, bin_dir, bins, phase, shape)
        self.bboxes = [os.path.join(bbox_dir, f + '.rect') for f in self.file_list]
        self.max_jitter = max_jitter

    def __getitem__(self, item):
        image, landmarks = super(BBoxDataset, self).__getitem__(item)
        bbox_path = self.bboxes[item]
        try:
            bbox = utils.read_bbox(bbox_path)
        except:
            bbox = [0, 0, image.shape[1] - 1, image.shape[0] - 1]
        minx, miny, maxx, maxy = bbox
        if self.phase == 'train':
            left = min(minx, self.max_jitter)
            right = min(image.shape[1] - maxx - 1, self.max_jitter)
            up = min(miny, self.max_jitter)
            down = min(image.shape[0] - maxy - 1, self.max_jitter)
            dh = np.random.randint(-up, down+1, 1)
            dv = np.random.randint(-left, right+1, 1)
            bbox[0::2] += dv
            bbox[1::2] += dh
            minx, miny, maxx, maxy = bbox
        landmarks = utils.norm_landmarks(landmarks, bbox)
        image = image[miny:maxy + 1, minx:maxx + 1, :]
        image = cv2.resize(image, self.shape)
        if self.phase == 'train':
            image, landmarks = utils.random_flip(image, landmarks, 0.5)
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, np.reshape(landmarks, (-1))


if __name__ == '__main__':
    a = BBoxDataset('/data/icme/data/picture',
                    '/data/icme/data/landmark',
                    '/data/icme/bbox',
                    '/data/icme/train')
    a.__getitem__(1)