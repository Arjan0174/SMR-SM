import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        cv2.imshow("image", img)

def cutframe(to_cut_image, start_x,  width, start_y, hight):
    image_cut = to_cut_image[start_y:start_y+hight,start_x:start_x+width]
    return image_cut

img = cv2.imread("Color frame.png")
img = cutframe(img, 80,500,120,400)
cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()