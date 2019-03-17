from proto import all_pb2
from data.bbox_dataset import BBoxDataset
from data.align_dataset import AlignDataset
from utils.alignment import Align

def get_dataset(phase):
    if phase == 'train':
        return BBoxDataset('/data/icme/crop/data/picture',
                           '/data/icme/crop/data/landmark',
                           '/data/icme/all', phase='train',
                           max_jitter=30, max_angle=30)
    else:
        return BBoxDataset('/data/icme/crop/data/picture',
                           '/data/icme/crop/data/landmark',
                           '/data/icme/valid', phase='eval')
    # if phase == 'train':
    #     return AlignDataset('/data/icme/crop/data/picture',
    #                         '/data/icme/crop/data/landmark',
    #                         '/data/icme/crop/data/landmark',
    #                         '/data/icme/all',
    #                         Align('./cache/mean_landmarks.pkl', (224, 224), (0.2, 0.1),
    #                               ),
    #                         flip=True,
    #                         max_jitter=10,
    #                         max_angle=10
    #                         )
    # else:
    #     return AlignDataset('/data/icme/crop/data/picture',
    #                         '/data/icme/crop/data/landmark',
    #                         '/data/icme/crop/data/pred_landmark',
    #                         '/data/icme/valid',
    #                         Align('./cache/mean_landmarks.pkl', (224, 224), (0.2, 0.1),
    #                               ),
    #                         phase='eval',
    #                         )