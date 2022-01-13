import cv2
import numpy as np
from realsense_depth import *

file = 'foto/enkel_1.PNG'
global color
dc = DepthCamera()



def track(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    H_low = cv2.getTrackbarPos('low H', 'controls')
    H_high = cv2.getTrackbarPos('high H', 'controls')
    S_low = cv2.getTrackbarPos('low S', 'controls')
    S_high = cv2.getTrackbarPos('high S', 'controls')
    V_low = cv2.getTrackbarPos('low V', 'controls')
    V_high = cv2.getTrackbarPos('high V', 'controls')

cv2.namedWindow('controls')
cv2.resizeWindow(('controls'), 700, 512)

H_low = 0
H_high = 179
S_low = 0
S_high = 255
V_low = 0
V_high = 255

cv2.createTrackbar('low H', 'controls', 0, 179, track)
cv2.createTrackbar('high H', 'controls', 179, 179, track)
cv2.createTrackbar('low S', 'controls', 0, 255, track)
cv2.createTrackbar('high S', 'controls', 255, 255, track)
cv2.createTrackbar('low V', 'controls', 0, 255, track)
cv2.createTrackbar('high V', 'controls', 255, 255, track)

while (1):
    ret, depth_frame, color_frame = dc.get_frame()
    color = color_frame


    img = color
    imgCropped = img[300:650, 250:600]
    HSV = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2HSV)
    HSV_low = np.array([H_low, S_low, V_low], np.uint8)
    HSV_high = np.array([H_high, S_high, V_high], np.uint8)
    mask = cv2.inRange(HSV, HSV_low, HSV_high)
    res = cv2.bitwise_and(imgCropped, imgCropped, mask=mask)
    # cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()