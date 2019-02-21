import sys
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()
window = dlib.image_window()
img = io.imread("/home/zhzhong/Desktop/test.jpg")

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    width = d.right() - d.left()
    hight = d.bottom() - d.top()
    center_r = (d.right() + d.left())/2
    center_c = (d.top() + d.bottom())/2
    left = int(center_r - width*1.25/2)
    right = int(center_r + width*1.25/2)
    top = int(center_c - hight*1.25/2)
    bottom = int(center_c + hight*1.25/2)
    dets = dlib.rectangle(left,top,right,bottom)
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, left, top, right, bottom))
window.clear_overlay()
window.set_image(img)
window.add_overlay(dets)
dlib.hit_enter_to_continue()