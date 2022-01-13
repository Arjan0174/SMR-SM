import numpy as np
import math
import socket
import serial
import subprocess
from tkinter import *
from PIL import Image, ImageTk
import cv2
import tkinter.font as font
import time
import pyrealsense2 as rs
from statistics import mean

THREAD = []
message_thread = []
thread = []
FORMAT = 'ascii'                                                                # Format
host = "192.168.11.5"                                                           # Server IP
port = 8000                                                                     # socket server port number
POST_MSG = "PSM"
DISAPPR_MSG_PC = 'MDP'
APPROVED_MSG_PC = 'MAP'

format = 'utf-8'
arduino = serial.Serial(port='COM7', baudrate=115200, timeout=.1)
num = 1

def client_program():
    global client_socket
    client_socket = socket.socket()                                             # instantiate
    client_socket.connect((host, port))

def init_arduino():
    data = datasplit()
    type_split = type(data)
    while type_split != list:
        data = datasplit()
        type_split = type(data)

def write_read():
    data = arduino.readline()
    if data == b'':
        return
    else: data = data.decode(format)
    return data

def datasplit():
    print_statement = write_read()
    if print_statement == None:
        pass
    else:
        part_one = ""
        part_two = ""
        changepart = 0
        for char in print_statement:
            if char != ',':
                if changepart == 0:
                    part_one = part_one + char
                if changepart == 1:
                    part_two = part_two + char
            else:
                changepart = + 1
        splitted_data = [part_one, part_two]
        # print (splitted_data)
        # print (splitted_data[-1])
        return splitted_data

def Start_stream():
    global depth_scale, pipeline, align
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)


def calibration():
    global PXTMTM
    features_mm_to_pixels_dict = {(316, 185): (179, 31), (566, 183): (359, 30), (565, -66): (360, 211), (313, -65): (179, 211)}
    A = np.zeros((2 * len(features_mm_to_pixels_dict), 6), dtype=float)
    b = np.zeros((2 * len(features_mm_to_pixels_dict), 1), dtype=float)
    index = 0
    for XY, xy in features_mm_to_pixels_dict.items():
        X = XY[0]
        Y = XY[1]
        x = xy[0]
        y = xy[1]
        A[2 * index, 0] = x
        A[2 * index, 1] = y
        A[2 * index, 2] = 1
        A[2 * index + 1, 3] = x
        A[2 * index + 1, 4] = y
        A[2 * index + 1, 5] = 1
        b[2 * index, 0] = X
        b[2 * index + 1, 0] = Y
        index += 1
    # A @ x = b
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    pixels_to_mm_transformation_mtx = np.array([[x[0, 0], x[1, 0], x[2, 0]], [x[3, 0], x[4, 0], x[5, 0]], [0, 0, 1]])
    return (pixels_to_mm_transformation_mtx)


def OmrekeningCord(pixelx, pixely, pixel_matrix):
    test_xy_1 = (pixelx, pixely, 1)
    test_XY_1 = pixel_matrix @ test_xy_1
    return (test_XY_1)

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
    global kleurenplaatje
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames) # Align the depth frame to color frame
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        return
    diepteplaatje = np.asanyarray(aligned_depth_frame.get_data())
    kleurenplaatje = np.asanyarray(color_frame.get_data())

    diepteplaatje = cutframe(diepteplaatje, 80,500,120,400)
    kleurenplaatje = cutframe(kleurenplaatje, 80,500,120,400)
    # kleurenplaatje = cv2.cvtColor(kleurenplaatje, cv2.COLOR_BGR2RGB)
    return(diepteplaatje, kleurenplaatje)

def dieptebepaling(depth_image_3d, color_image):
    grey_color = 0

    for x in range(650, 835, 5):
        x = x / 1000 / depth_scale
        removed = np.where((depth_image_3d > x) | (depth_image_3d <= 0), grey_color, color_image)
        # ret, mask = cv2.threshold(removed, 1, 255, cv2.THRESH_BINARY)
        # print("x =", x)

        # cord = gem_cirkelcoordinaten(lijst)
        # print('cord bij x', cord)
        # print('lengte lijst: ', len(lijst))

        # mask2 = cv2.circle(mask, cord, 10, (255, 255, 0), 3)
        nogreen = remove_green(color_image)
        kartonnetjes = cv2.bitwise_and(nogreen, removed)
        lijst = (np.argwhere(kartonnetjes != 0))
        # print('lengte lijst', len(lijst))
        if len(lijst) > 13000:
            karton, check = get_cartboard_angle(kartonnetjes)
        else:
            check = False
            # print("lijst is te kort")
        if check == True:
            return (check, kartonnetjes)

        #cv2.namedWindow('Exampe', cv2.WINDOW_NORMAL)
        #cv2.imshow('Exampe', kartonnetjes)

        cv2.waitKey(1)
    return(check, kartonnetjes)

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


def get_cartboard_angle(cartboard_image):
    global cardboardimage
    global x, y, w, h
    global rotation_angle_deg, rotation_angle_rad
    global MidpointA, MidpointB, MidpointC
    global lefty, righty, rows, cols
    # print ("in cartboard_angle")
    kernel1 = [5,5]
    kernel = np.ones((8, 8), np.uint8)
    # locate contour en get data where the centre is
    image = cartboard_image
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale,kernel1,4)
    edges = cv2.Canny(blur, 30, 100)
    dialation = cv2.dilate(edges, kernel, iterations=1)

    _, contours, hierarchy = cv2.findContours(dialation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(cv2.contourArea(contours[len(contours)-1]))

    try:
        cnt = contours[len(contours)-1]
        area = cv2.contourArea(cnt)
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     print (area)
    #     # print(area)
        if area > 4000: #8000
            rows, cols = image.shape[:2]
            [vx, vy, xf, yf] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            # print (vx,vy, xf, yf)
            lefty = int((-xf * vy / vx) + yf)
            righty = int(((cols - xf) * vy / vx) + yf)
            cv2.line(image, (cols - 1, righty), (0, lefty), (20, 20, 255), 2)
            # print ("rightly" ,lefty, righty)

            O = abs((righty-lefty))
            A = cols-1
            rotation_angle_rad = round(math.atan((O/A)),2)
            rotation_angle_deg = round(math.degrees(rotation_angle_rad),2)
            # print ("rotation angle", rotation_angle_rad, "Radian" )
            # print ("rotation angle", rotation_angle_deg, "Degrees")

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            BoxCords1 = box[0]
            BoxCords2 = box[1]
            BoxCords3 = box[2]
            BoxCords4 = box[3]
            MidpointA = midpoint(BoxCords1,BoxCords2)
            MidpointB = midpoint(BoxCords3,BoxCords4)
            MidpointC = midpoint(MidpointA,MidpointB)
            # print("!!!!!!!!!!!!!", MidpointC)
            cv2.circle(image,MidpointC,10,(255,255,0),3)


            cv2.drawContours(dialation, [box], 0, (0, 191, 255), 2)
            # draw_angled_rec(x0, y0, width, height, angle, img)

            cv2.drawContours(image, cnt, -1, (0, 100, 100), 2, 0)
            x, y, w, h = cv2.boundingRect(cnt)
            # print (x,y,w,h)
            cv2.rectangle(image, (x, y), (x + w , y + h), (90, 30, 0), 2)
            cardboardimage = image
            check = True
            return cardboardimage, check#, x, y, w, h
        else:
            # print("AREA TO SMALL")
            check = False
            nepcardboard = 0
            return nepcardboard, check
    except:
        print("NONE DETECTED")
        check = False
        nepcardboard = 0
        return nepcardboard, check

def midpoint(p1, p2):
    Point = [round((p1[0]+p2[0])/2), round((p1[1]+p2[1])/2)]
    return Point

def getservermessage():
    try:
        client_socket.settimeout(0.05)
        msg = client_socket.recv(3)
        msg = msg.decode(FORMAT)
        client_socket.settimeout(1000)
        return msg
    except:
        msg = ""
        return msg

def formatmessage(coords):
    data = datasplit()
    pickupdata = str(round(coords[0],2)) +","+str(round(coords[1],2))+","+str(distance)+",0,0,"+str(round(coords[2],2))

    try:
        message = pickupdata +","+data[1]
        message_thread.append(message)
        if len(message_thread)>1:
            del message_thread[0]
        print("MESSAGE THREAD:   ", message_thread[:])
    except:
        pass

def sendtrue(server_msg):
    if len(message_thread) != 0:
        client_socket.send(message_thread[-1].encode(FORMAT))
    if len(message_thread)> 1:
        del message_thread[0]

def printcoordinaten(PXTMTM):
    World_Cord = OmrekeningCord(MidpointC[0], MidpointC[1], PXTMTM)

    print('World_Cord:', World_Cord)
    print(rotation_angle_rad, '....................................................................')
    if World_Cord[0] <= 410:
        xcord = World_Cord[0] + 600
        if righty >= lefty:                                                                                 # /
            xcord = round((xcord) - (300 - math.cos(rotation_angle_rad + 0.000001) * 300))
            ycord = round((World_Cord[1]) - math.sin(rotation_angle_rad + 0.000001) * 300)
            Cord = [xcord, ycord, -90 - rotation_angle_deg + 180]
        else:                                                                                               # \
            xcord = round((xcord) - (300 - math.cos(rotation_angle_rad + 0.000001) * 300))
            ycord = round((World_Cord[1]) + math.sin(rotation_angle_rad + 0.000001) * 300)
            Cord = [xcord, ycord, -90 + rotation_angle_deg + 180]

    else:
        if righty >= lefty:                                                                                 # /
            xcord = round((World_Cord[0]) + 300 - math.cos(rotation_angle_rad + 0.000001) * 300)
            ycord = round((World_Cord[1]) + math.sin(rotation_angle_rad + 0.000001) * 300)
            Cord = [xcord, ycord, -90 - rotation_angle_deg]
        else:                                                                                               # \
            xcord = round((World_Cord[0]) + 300 - math.cos(rotation_angle_rad + 0.000001) * 300)
            ycord = round((World_Cord[1]) - math.sin(rotation_angle_rad + 0.000001) * 300)
            Cord = [xcord, ycord, -90 + rotation_angle_deg]
    return Cord

def runserver():
    subprocess.Popen('python server.py', shell=True)
def HMI_bloem_lengte():
    global bloemlengte
    bloemlengte = entry.get()
    print(bloemlengte)

def HMI_pauze():
    global pauze, num
    num = num+1
    while num %2 == 0:
        print("PAUZE")
        root.after(100, root.update())

def HMI_start():
    start()

def HMI():
    global root, L1, entry
    root = Tk()
    root.title("Flower length")
    root.geometry('1520x800+0+0')

    buttonFont = font.Font(family='Helvetica', size=20, weight='bold')
    buttonFont1 = font.Font(family='Helvetica', size=14, weight='bold')
    Achtergrond = PhotoImage(file='Images/rozen1.png')

    Label(root, image=Achtergrond).pack()
    Label(root, height=32, width=39, bg='thistle').place(x=325, y=150)
    Label(root, text='Flower length?', height=2, width=14, font=buttonFont1).place(x=380, y=425)
    Label(root, text='mm', height=2, width=5, font=buttonFont1).place(x=490, y=480)

    entry = Entry(root, font=buttonFont1).place(x=380, y=480, width=100, height=50)

    Button(root, command=HMI_start,text='Start', height=2, width=10, font=buttonFont).place(x=375, y=175)
    Button(root, command=HMI_pauze, text='Pauze', height=2, width=10, font=buttonFont).place(x=375, y=275)
    Button(root, command=HMI_bloem_lengte, text='submit', height=2, width=14, font=buttonFont1).place(x=380, y=540)

    f1 = LabelFrame(root, bg='red', height=10, width=8)
    f1.place(x=625, y=150)
    L1 = Label(f1, bg='red')
    L1.pack()

    cap = cv2.VideoCapture(1)
    while True:
        img = cap.read()[1]
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img1))
        L1['image'] = img
        root.update()
    cap.release()
    root.mainloop()


def start():
    global distance
    runserver()
    time.sleep(2)
    client_program()
    init_arduino()
    count = 4
    Start_stream()
    # Create opencv window to render image in
    #cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("Depth Stram", cv2.WINDOW_AUTOSIZE)
    # Create colorizer object
    colorizer = rs.colorizer()
    checkt = False
    PXTMTM = calibration()
    time.sleep(1)
    # Streaming loop
    while True:
        depth_image, color_image = get_diepte()
        color_frame = color_image
        #cv2.imshow("Depth Stram", color_frame)

        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)
        # depth_colormaps = colorizer.colorize(depth_image)
        checkt, boundingkarton = dieptebepaling(depth_image_3d,color_image)

        if checkt == True:
            World_Cord = printcoordinaten(PXTMTM)
            print('wereldcor', World_Cord)

            distance = depth_image[MidpointC[1], MidpointC[0]]
            distance = 400 -distance-60
            if distance <= -455:
                distance = -455
            print(distance)

        #cv2.imshow("Depth Stream", boundingkarton)
        # time.sleep(1)

        message_from_server = getservermessage()
        print(' ------------------------------------------------------', message_from_server )


        if ((message_from_server == POST_MSG) and (count > 1)):
            print ("MESAAGE ___________________________________________________")
            count = 0

        formatmessage(World_Cord)

        count = count + 1
        if count == 1:
            # CORD1 = coordinaat_y
            # CORD2 = coordinaat_x
            # cv2.circle(result_image, (World_Cord[0], World_Cord[1]), (8), (0, 255, 0), (3))
            sendtrue(message_from_server)
            try:
                del message_thread[:]
            except:
                pass

        try:
            cv2.circle(color_frame, MidpointC, 10, (255, 255, 0), 3)
        except:
            pass

        img1 = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img1))



        L1['image'] = img

        root.update()
        key = cv2.waitKey(2)
        if key == 27:
            break

HMI()