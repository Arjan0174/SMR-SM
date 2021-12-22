#####################################################
##               Read bag from file                ##
#####################################################

import time

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
from statistics import mean


def cutframe(to_cut_image, start_x,  width, start_y, hight):
    image_cut = to_cut_image[start_y:start_y+hight,start_x:start_x+width]
    return image_cut

def gem_cirkelcoordinaten(arrmetcrdnt):
    lijst = arrmetcrdnt.tolist()
    li = 0
    lijstx = []
    lijsty= []
    gemx = 0
    gemy = 0
    for li in lijst:
        lijstx.append(li[0])
        lijsty.append(li[1])
    if li != 0:
        gemx = int(round(mean(lijstx)))
        gemy = int(round(mean(lijsty)))

    # print(lijsty)
    # print(lijstx)
    return(gemy,gemx)

def get_diepte():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames) # Align the depth frame to color frame
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        return
    diepteplaatje = np.asanyarray(aligned_depth_frame.get_data())
    kleurenplaatje = np.asanyarray(color_frame.get_data())

    diepteplaatje = cutframe(diepteplaatje, 250, 800, 300, 800)
    kleurenplaatje = cutframe(kleurenplaatje, 250, 800, 300, 800)
    kleurenplaatje = cv2.cvtColor(kleurenplaatje, cv2.COLOR_BGR2RGB)
    return(diepteplaatje, kleurenplaatje)

def dieptebepaling(depth_image_3d, color_image):
    grey_color = 0

    for x in range(650, 835, 5):
        x = x / 1000 / depth_scale
        removed = np.where((depth_image_3d > x) | (depth_image_3d <= 0), grey_color, color_image)
        # zonderachtergrond = Image.fromarray(removed)
        # ret, mask = cv2.threshold(removed, 1, 255, cv2.THRESH_BINARY)
        print("x =", x)
        # lijst = (np.argwhere(mask != 0))
        # cord = gem_cirkelcoordinaten(lijst)
        # print('cord bij x', cord)
        # print('lengte lijst: ', len(lijst))
        # mask2 = cv2.circle(mask, cord, 10, (255, 255, 0), 3)
        nogreen = remove_green(color_image)
        kartonnetjes = cv2.bitwise_and(nogreen, removed)
        cv2.namedWindow('Exampe', cv2.WINDOW_NORMAL)
        cv2.imshow('Exampe', kartonnetjes)

        cv2.waitKey(1)
        time.sleep(0.2)


def remove_green(image_green):
    global result_no_green
    lbh = 26
    lbs = 26  # 0
    lbv = 0  # 0
    ubh = 255  # 255
    ubs = 255
    ubv = 223
    img = image_green
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # Threshold of blue in HSV space
    lower_blue = np.array([lbh, lbs, lbv])
    upper_blue = np.array([ubh, ubs, ubv])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_not(mask, mask)
    # The black region in the mask has the value of 0,
    result_no_green = cv2.bitwise_and(img, img, mask=mask)
    return (result_no_green)

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, r'D:\python\pycharm\rbag\20211221_160744.bag')

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    # Start streaming from file
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth Stram", cv2.WINDOW_AUTOSIZE)
    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    while True:
        # Get frameset of depth

        depth_image, color_image = get_diepte()

        cv2.imshow("Depth Stram", color_image)
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)
        # depth_colormaps = colorizer.colorize(depth_image)
        dieptebepaling(depth_image_3d,color_image)


        #cv2.imshow("Depth Stream", depth_colormap)




        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass