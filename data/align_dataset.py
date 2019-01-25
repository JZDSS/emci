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

        image = cv2.resize(image, self.shape)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, np.reshape(landmarks, (-1))


