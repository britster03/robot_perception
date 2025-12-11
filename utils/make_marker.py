# save as make_marker.py, then:  python make_marker.py
import cv2
from cv2 import aruco

dict4x4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = aruco.generateImageMarker(dict4x4, 0, 1200)  # ID=0, 1200×1200 px
cv2.imwrite("aruco_id0.png", img)
print("Saved aruco_id0.png in current folder.")

