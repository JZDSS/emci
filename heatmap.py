
import numpy as np

import os
import cv2 as cv

from data import utils

#每张aligin后图片生成8张heatmap对应五官位置
def get_id(name):
    t = name.split('_')[0:2]
    return t[0] + t[1]

root_dir = 'D:\icmedata\sendaligned\\aligned\\'

lamdmark_dir = os.path.join(root_dir, 'landmark')
image_dir = os.path.join(root_dir, 'picture')
#bbox_dir = os.path.join(root_dir, 'bbox')
filenames = os.listdir(image_dir)

for filename in filenames:
    id = get_id(filename)
    landmark_path = os.path.join(lamdmark_dir, filename + '.txt')
    landmarks = utils.read_mat(landmark_path)
    x = landmarks[0:33,0]
    y = landmarks[0:33,1]

    # img = img +255
    # img.fill(255)
    color = (255, 255, 255)
    # print(landmarks[0,:])
    # print(landmarks[1,:])


    #脸部外轮廓
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(32):
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j+1, 0], landmarks[j+1, 1]), color,thickness=2)#thickness
    pic1 = cv.GaussianBlur(img,(11,11),3)
    # #两个sigma允许输入负数等其他不常用的输入。
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\faceoutline\\" + id + ".png"
    cv.imwrite(Img_Name,pic1)


    #鼻梁
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(3):
        j =j + 51
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j+1, 0], landmarks[j+1, 1]), color,thickness=2)#thickness
    pic2 = cv.GaussianBlur(img, (11, 11), 3)
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\nosebridge\\" + id + ".png"
    cv.imwrite(Img_Name,pic2)

    #鼻子下轮廓
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(10):
        j = j + 55
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                thickness=2)  # thickness
    pic3 = cv.GaussianBlur(img, (11, 11), 3)
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\noseboundary\\" + id + ".png"
    cv.imwrite(Img_Name, pic3)


    #左眉
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(8):
        j = j + 33
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                thickness=2)  # thickness
    pic4 = cv.GaussianBlur(img, (11, 11), 3)
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\lefteyebrow\\" + id + ".png"
    cv.imwrite(Img_Name, pic4)

    #右眉
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(8):
        j = j + 42
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                thickness=2)  # thickness
    pic5 = cv.GaussianBlur(img, (11, 11), 3)
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\righteyebrow\\" + id + ".png"
    cv.imwrite(Img_Name, pic5)

    #左眼
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(7):
        j = j + 66
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                thickness=2)  # thickness
    pic6 = cv.GaussianBlur(img, (11, 11), 3)
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\lefteye\\" + id + ".png"
    cv.imwrite(Img_Name, pic6)

    #右眼
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(7):
        j = j + 75
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                thickness=2)  # thickness
    pic7 = cv.GaussianBlur(img, (11, 11), 3)
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\righteye\\" + id + ".png"
    cv.imwrite(Img_Name, pic7)

    #嘴唇
    img = np.zeros((128, 128, 3), np.uint8)
    for j in range(19):
        j = j + 84
        cv.line(img, (landmarks[j, 0], landmarks[j, 1]), (landmarks[j + 1, 0], landmarks[j + 1, 1]), color,
                thickness=2)  # thickness
    pic8 = cv.GaussianBlur(img, (11, 11), 3)
    Img_Name = "D:\icmedata\sendaligned\\aligned\heatmap\\mouth\\" + id + ".png"
    cv.imwrite(Img_Name, pic8)




    # picstack = np.hstack([img,pic2])
    # cv.imshow('stack',picstack)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
