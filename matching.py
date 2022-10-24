import numpy as np
import cv2
import sys
from time import time

import kcftracker

file_ref = sys.argv[1]
file_tar = sys.argv[2]
img_ref = cv2.imread(file_ref)
img_tar = cv2.imread(file_tar)
height = img_ref.shape[0]
width = img_ref.shape[1]

tracker = kcftracker.KCFTracker(True, False, True)  # hog, fixed_window, multiscale
bbox = cv2.selectROI("Tracking",img_ref, fromCenter = False, showCrosshair = False)
cv2.destroyAllWindows()
cx_ref = bbox[0] + bbox[2] / 2.
cy_ref = bbox[1] + bbox[3] / 2.
tracker.init([bbox[0],bbox[1],bbox[2],bbox[3]],img_ref)
t0 = time()
cx_tar,cy_tar, value = tracker.matching(img_tar)
t1 = time()
print("x方向の平行移動量:  {}\ny方向の平行移動量:  {}\nピーク値: {}\n処理時間: {}".format(cx_tar-cx_ref,cy_tar-cy_ref,value,t1-t0))