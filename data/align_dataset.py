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
                 shape=(224, 224),
                 flip=True,
                 ldmk_ids=[i for i in range(106)],
                 max_jitter=3,
                 max_radian=0,
                 img_format='png'):
        super(AlignDataset, self).__init__(img_dir, gt_ldmk_dir, bin_dir, bins, phase, shape, img_format)
        self.aligner = aligner
        self.algin_ldmk = [os.path.join(al_ldmk_dir, f + '.txt') for f in self.file_list]
        self.ldmk_ids = ldmk_ids
        if phase == 'train':
            self.flip = flip
            self.max_jitter = max_jitter
            self.max_radian = max_radian
        else:
            self.flip = False
            self.max_jitter = 0
            self.max_radian = 0

    def __getitem__(self, item):
        image, landmarks = super(AlignDataset, self).__getitem__(item)
        al_ldmk = utils.read_mat(self.algin_ldmk[item])
        image, _, t = self.aligner(image,
                                   al_ldmk,
                                   noise=np.random.uniform(-self.max_jitter, self.max_jitter, 2),
                                   radian=np.random.uniform(-self.max_radian, self.max_radian))
        landmarks = landmarks @ t[0:2, :] + t[2, :]
        # start_y = np.random.randint(0, self.aligner.scale[0] - self.shape[0] + 1)
        # start_x = np.random.randint(0, self.aligner.scale[1] - self.shape[1] + 1)
        # landmarks[:, 0] -= start_x
        # landmarks[:, 1] -= start_y
        landmarks[:, 0] /= self.shape[1]
        landmarks[:, 1] /= self.shape[0]
        # image = image[start_y:start_y + self.shape[0], start_x :start_x + self.shape[1]]
        if self.phase == 'train':
            if self.flip:
                image, landmarks = utils.random_flip(image, landmarks, 0.5)
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)

        image = cv2.resize(image, self.shape)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        landmarks = landmarks[self.ldmk_ids, :]
        return image, np.reshape(landmarks, (-1))


