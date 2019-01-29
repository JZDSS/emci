import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
from data import utils
import cv2

def pwa(image, src, dst, shape):
    """
    :param image: aligned image or cropped image
    :param src: normalized src landmarks
    :param dst: normalized dst landmarks
    :param shape: output shape, [rows, cols]
    :return: piece-wise affined image
    """
    image = cv2.resize(image, shape)
    src[:, 0] *= shape[0]
    src[:, 1] *= shape[1]
    dst[:, 0] *= shape[0]
    dst[:, 1] *= shape[1]
    N = 10
    z = np.zeros((N, 1))
    l = np.reshape(np.linspace(0, shape[1], N), (N, 1))
    top = np.concatenate([l, z], axis=1)
    bottom = np.concatenate([l, np.ones((N, 1)) * shape[0]], axis=1)

    l = np.reshape(np.linspace(0, shape[0], N), (N, 1))
    left = np.concatenate([z, l], axis=1)
    right = np.concatenate([np.ones((N, 1)) * shape[1], l], axis=1)

    add = np.concatenate([top, bottom, left, right], axis=0)
    src = np.concatenate([src, add], axis=0)
    dst = np.concatenate([dst, add], axis=0)
    tform = PiecewiseAffineTransform()
    tform.estimate(dst, src)
    # out_rows ,out_cols = shape
    out_rows = image.shape[0]
    out_cols = image.shape[1]
    out = warp(image, tform, output_shape=(out_rows, out_cols))
    return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = plt.imread('/data/icme/data/picture/AFW_134212_1_0.jpg')
    bbox = utils.read_bbox('/data/icme/bbox/AFW_134212_1_0.jpg.rect')
    src = utils.read_mat('/data/icme/data/landmark/AFW_134212_1_0.jpg.txt')

    src = utils.norm_landmarks(src, bbox)
    image = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    bbox = utils.read_bbox('/data/icme/bbox/AFW_134212_1_3.jpg.rect')
    dst = utils.read_mat('/data/icme/data/landmark/AFW_134212_1_3.jpg.txt')
    dst = utils.norm_landmarks(dst, bbox)

    out = pwa(image, src, dst, (128, 128))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)

    plt.imshow(out)
    # plt.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    plt.axis((0, 128, 128, 0))
    plt.show()