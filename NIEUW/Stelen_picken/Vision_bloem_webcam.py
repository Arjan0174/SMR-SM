from realsense_depth import *
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

THREAD = []
message_thread = []
thread = []
FORMAT = 'ascii'                                                                # Format
host = "192.168.11.5"                                                           # Server IP
port = 8000                                                                     # socket server port number
POST_MSG = "PSM"
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
        print (splitted_data)
        print (splitted_data[-1])
        return splitted_data

def cutframe(to_cut_image, start_x,  width, start_y, hight):
    global imgCropped
    imgCropped = to_cut_image[start_y:start_y+hight,start_x:start_x+width]
    return imgCropped

def verwijder_zwart():
    global result_no_zwart
    HSV = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2HSV)
    lower_zwart = np.array([0,70,0])
    upper_zwart = np.array([179,255,255])
    mask = cv2.inRange(HSV, lower_zwart, upper_zwart)
    result_no_zwart = cv2.bitwise_and(imgCropped, imgCropped, mask=mask)

def verwijder_yellow():
    global result_no_yellow
    HSV = cv2.cvtColor(result_no_zwart, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([25,0,0])
    upper_yellow = np.array([179,255,255])
    mask = cv2.inRange(HSV, lower_yellow, upper_yellow)
    result_no_yellow = cv2.bitwise_and(result_no_zwart, result_no_zwart, mask=mask)

def verwijder_blauw():
    global result_image
    HSV = cv2.cvtColor(result_no_zwart, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0,0,0])
    upper_blue = np.array([70,255,255])
    mask = cv2.inRange(HSV, lower_blue, upper_blue)
    result_no_blue = cv2.bitwise_and(result_no_zwart, result_no_zwart, mask=mask)
    result_image = result_no_blue

def filter_image():
    global erode
    kernel1 = [5,5]
    kernel = np.ones((3, 6), np.uint8)
    grayscale = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, kernel1, 4)
    erode = cv2.erode(blur, kernel, iterations=3)

def contour():
    global cnt, kontoeren, area, cx, cy
    _, contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 350:
            cv2.drawContours(result_image, cnt, -1, (0, 0, 255), 2, 0)
            #cv2.drawContours(color_frame, cnt, -1, (0, 0, 255), 2, 0, offset=(80, 120))
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            kontoeren = cnt

def center():
    global cx, cy, M
    M = cv2.moments(kontoeren)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

def cirkel():
    cv2.circle(result_image, (cx, cy), 20, (255, 255, 255), 1)

def lijn():
    global rows, cols, lefty, righty, result_image
    rows, cols = result_image.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(kontoeren, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(result_image, (cols - 1, righty), (0, lefty), (255, 255, 255), 1)

def hoek():
    global rotation_angle_deg, rotation_angle_rad
    O = abs((righty - lefty))
    A = cols - 1
    if lefty>=righty:
        rotation_angle_rad = round(math.atan((O / A)), 2)
        rotation_angle_deg = -1* round(math.degrees(rotation_angle_rad), 2)
    else:
        rotation_angle_rad = round(math.atan((O / A)), 2)
        rotation_angle_deg = round(math.degrees(rotation_angle_rad), 2)

def fix_line():
    global cirkel_zwart, coordinaten, zwart, nulpunt, colsaccent
    if righty >= lefty:
        nulpunt= 800
        left = nulpunt-lefty
    else:
        nulpunt = 0
        left = lefty
    if rotation_angle_rad != 0:
        colsaccent = round(left/math.tan(rotation_angle_rad))
    else:
        colsaccent = round(left / math.tan(rotation_angle_rad+0.0001))

def coordinates():
    global coordinaten
    radius = 20
    zwart_vlak1 = np.zeros((800, 800), dtype="uint8")
    lijn_zwart = cv2.line(zwart_vlak1, (colsaccent, nulpunt), (0, lefty), (255, 255, 255), 1)
    zwart_vlak2 = np.zeros((800, 800), dtype="uint8")
    cirkel_zwart = cv2.ellipse(zwart_vlak2, center=(cx,cy), axes=(radius, radius),angle=0, startAngle=-50, endAngle=50, color=(255, 255, 255), thickness=1)

    zwart = cv2.bitwise_and(lijn_zwart, cirkel_zwart)
    if ((rotation_angle_deg > 10) or (rotation_angle_deg < -10)):
        coordinaten = np.array([[cy,cx]])
    else:
        coordinaten = np.argwhere(zwart == 255)

def coordinaten_formateren(coordinaten):
    global coordinaat_x, coordinaat_y
    lijst = coordinaten.tolist()
    li = 0
    coordinaat_x = 0
    coordinaat_y = 0
    try:
        for li in lijst:
            coordinaat_x =(li[0])
            coordinaat_y =(li[1])
            cv2.circle(result_image, (coordinaat_y, coordinaat_x), (2), (255, 0, 0), (1))
            if ((rotation_angle_deg > 10) or (rotation_angle_deg < -10)):
                cv2.circle(color_frame, (coordinaat_y+80, coordinaat_x+80), (5), (255, 0, 0), (6))
            else:
                cv2.circle(color_frame, (coordinaat_y + 80, coordinaat_x +80), (5), (255, 0, 0), (6))
        return (coordinaat_x, coordinaat_y)
    except:
        coordinaat_x = 0
        coordinaat_y = 0
        print("geen cordinaten")
        return (coordinaat_x, coordinaat_y)

def OmrekeningCord(pixelx, pixely, pixel_matrix):
    test_xy_1 = (pixelx+80, pixely+80, 1)
    test_XY_1 = pixel_matrix @ test_xy_1
    return (test_XY_1)

def calibration():
    features_mm_to_pixels_dict ={(515, 203): (252, 144), (783, 201): (445, 144), (782, -66): (446, 338), (512, -66): (252, 337)}
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
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    pixels_to_mm_transformation_mtx = np.array([[x[0, 0], x[1, 0], x[2, 0]], [x[3, 0], x[4, 0], x[5, 0]], [0, 0, 1]])
    return (pixels_to_mm_transformation_mtx)

def getservermessage():
    try:
        client_socket.settimeout(0.05)
        msg = client_socket.recv(1024)
        msg = msg.decode(FORMAT)
        client_socket.settimeout(1000)
        return msg
    except:
        msg = ""
        return msg

def formatmessage(coords):
    data = datasplit()
    pickupdata = str(round(coords[0],2)) +","+str(round(coords[1],2))+",-450,0,0,"+str(round(coords[2],2))

    try:
        message = pickupdata +","+data[1]
        message_thread.append(message)
        if len(message_thread)>1:
            del message_thread[0]
        print("MESSAGE THREAD:   ", message_thread[:])
    except:
        pass

def sendtrue(server_msg):
    client_socket.send(message_thread[-1].encode(FORMAT))
    if len(message_thread)> 1:
        del message_thread[0]

def printcoordinaten():
    ycord, xcord = coordinaten_formateren(coordinaten)
    PXTMTM = calibration()
    if lefty >= righty:
        xcord = round(xcord + math.tan(rotation_angle_rad+0.000001) * 75)
    else:
        xcord = round(xcord + math.tan(rotation_angle_rad + 0.000001) * 75)
    World_Cord = OmrekeningCord(xcord, ycord, PXTMTM)
    if len(coordinaten) == 0:
        CORDS = []
    else:
        CORDS = [World_Cord[0], World_Cord[1], -86 - rotation_angle_deg]
    print('Pick Coordinaten:', CORDS)
    return CORDS

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

def start():
    global color, color_frame
    dc = DepthCamera()
    runserver()
    time.sleep(2)
    client_program()
    init_arduino()
    count = 4

    while True:
        pauze = True
        ret, depth_frame, color_frame = dc.get_frame()
        color = color_frame
        try:
            cutframe(color_frame, 80, 200, 120, 1000)
            verwijder_zwart()
            verwijder_blauw()
            filter_image()
            contour()
            center()
            cirkel()
            lijn()
            hoek()
            fix_line()
            coordinates()
            coordinaten_formateren(coordinaten)
            coordinaten_CORDS = printcoordinaten()
            message_from_server = getservermessage()

            if ((message_from_server == POST_MSG) and (count > 4)):
                count = 0

            formatmessage(coordinaten_CORDS)

            count = count+1
            if count == 4:
                CORD1 = coordinaat_y
                CORD2 = coordinaat_x
                cv2.circle(result_image, (CORD1, CORD2), (8), (0, 255, 0), (3))
                sendtrue(message_from_server)
                try: del message_thread[:]
                except: pass

            img1 = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img1))
            L1['image'] = img

            root.update()
            key = cv2.waitKey(2)
            if key == 27:
                break
        except:
            print ("Error encountered")

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

HMI()
