import numpy as np
import cv2 as cv
from data import utils

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
# color = np.random.randint(0,255,(106,3))
# Take first frame and find corners in it
old_frame = cv.imread('/data/icme/data/picture/AFW_70037463_1_2.jpg')
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = utils.read_mat('/data/icme/data/landmark/AFW_70037463_1_2.jpg.txt')
p0 = np.expand_dims(p0, 1)
for new in p0:
    a, b = new.ravel()
    # c, d = old.ravel()
    # mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    frame = cv.circle(old_frame, (a, b), 2, (0, 0, 255), 1)
cv.imshow("frame", frame)
cv.waitKey(0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
for i in [3, 4]:
    name = '/data/icme/data/picture/AFW_70037463_1_%d.jpg' % i
    frame = cv.imread(name)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    # good_old = p0[st==1]
    good_old = utils.read_mat('/data/icme/data/landmark/AFW_70037463_1_%d.jpg.txt' % i)
    good_old = np.expand_dims(good_old, 1)
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),2,(0, 0, 255), 1)
        frame = cv.circle(frame, (c, d), 2, (0, 255, 0), 1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(0) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv.destroyAllWindows()
