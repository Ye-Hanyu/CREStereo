import numpy as np
import cv2


ref = np.load('ref.npy')
disp = np.load('disp.npy')

# limg = cv2.imread('img/left.png')
# rimg = cv2.imread('img/right.png')
# cv2.imwrite('img/left_crop.png', limg[298:438,578:788,:])
# cv2.imwrite('img/right_crop.png', rimg[298:438,578:788,:])


dif = ref[298:438,578:788]-disp*16
a=0
