import os
from utils.alignment import Align
from data import utils
import cv2


if __name__ == '__main__':

    root_dir = '/data/icme'
    bin_dir = '/data/icme/valid'
    pose = 1
    aligner = Align('cache/mean_landmarks.pkl', scale=(128, 128), margin=(0.15, 0.1))
    bins = os.listdir(bin_dir)
    img_dir = os.path.join(root_dir, 'data/picture')
    landmark_dir = os.path.join(root_dir, 'data/pred_landmark')
    bbox_dir = os.path.join(root_dir, 'bbox')
    file_list = []
    for b in bins:
        curr = os.path.join(bin_dir, b)
        files = os.listdir(curr)
        file_list.extend(files)

    images = [os.path.join(img_dir, f) for f in file_list]
    landmarks = [os.path.join(landmark_dir, f + '.txt') for f in file_list]
    bboxes = [os.path.join(bbox_dir, f + '.rect') for f in file_list]

    img_out_dir = '/data/icme/align_by_ldmk/picture'
    landmarks_out_dir = '/data/icme/align_by_ldmk/landmark'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.exists(landmarks_out_dir):
        os.makedirs(landmarks_out_dir)

    for img_path, bbox_path, landmark_path, f in zip(images, bboxes, landmarks, file_list):

        bbox = utils.read_bbox(bbox_path)
        landmark = utils.read_landmarks(landmark_path)
        image = cv2.imread(img_path)

        image, landmark = aligner(image, landmark, bbox)

        cv2.imwrite(os.path.join(img_out_dir, f), image)
        utils.save_landmarks(landmark, os.path.join(landmarks_out_dir, f + '.txt'))
