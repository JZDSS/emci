from data import align_dataset
from sklearn.externals import joblib
from pdb import get_id

class AlignLabelDataset(align_dataset.AlignDataset):

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
                 label_file='cache/labels.pkl'):
        super(AlignLabelDataset, self).__init__(img_dir, gt_ldmk_dir, al_ldmk_dir, bin_dir, aligner, bins,
                                                phase, shape, flip, ldmk_ids, max_jitter, max_radian)
        self.labels = joblib.load(label_file)

    def __getitem__(self, item):
        image, landmark = super(AlignLabelDataset, self).__getitem__(item)
        img_path = self.images[item]
        img_name = img_path.split('/')[-1]
        id = get_id(img_name)
        label = self.labels[id]
        return image, landmark, label