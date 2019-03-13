import os
import matplotlib.pyplot as plt
from data import utils


root = '/data/icme/'
img_dir = os.path.join(root, 'data/picture')
ldmk_dir = os.path.join(root, 'data/landmark')
bbox_dir = os.path.join(root, 'bbox')

out_dir = os.path.join(root, 'crop')
if not os.path.exists(out_dir):
    os.makedirs(os.path.join(out_dir, 'data/picture'))
    os.makedirs(os.path.join(out_dir, 'data/landmark'))

for name in os.listdir(img_dir):
    img_name = name
    ldmk_name = name + '.txt'
    bbox_name = name + '.rect'
    img_path = os.path.join(img_dir, img_name)
    ldmk_path = os.path.join(ldmk_dir, ldmk_name)
    bbox_path = os.path.join(bbox_dir, bbox_name)

    img_out_path = img_path.replace(root, root + 'crop/')
    ldmk_out_path = ldmk_path.replace(root, root + 'crop/')

    img = plt.imread(img_path)
    bbox = utils.read_bbox(bbox_path)
    ldmk = utils.read_mat(ldmk_path)

    minx, miny, maxx, maxy = bbox
    ldmk -= [minx, miny]

    img = img[miny:maxy, minx:maxx, :]

    plt.imsave(img_out_path, img)
    utils.save_landmarks(ldmk, ldmk_out_path)
