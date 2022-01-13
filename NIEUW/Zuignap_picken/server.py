import socket
import time

PORT = 8000
SERVER = "192.168.11.5"
ADDR = (SERVER, PORT)
FORMAT = "ascii"

ip_robot = '192.168.11.31'
ip_self = "192.168.11.5"

thread = [] #empty posting thread

DISAPPR_MSG_PC = 'MDP'
APPROVED_MSG_PC = 'MAP'
ROBOT_MSG_REQUEST = 'REQ'
ROBOT_MSG_RECEIVED = 'REC'
EMPTY_MSG_ROBOT = 'EMP'
IN_MOTION_MSG = 'INM'
LOOPBAND_MSG = 'LPB'
POST_MSG = "PSM"

MSG_RECEIVED = True
EMPTY = False
LOOPBAND = False
INMOTION = False

def decode(data_to_decode):                             #decode the data
    decoded_data = data_to_decode.decode(FORMAT)
    if type(decoded_data) == str:
        return decoded_data
    else:
        print("Error data not allowed for processing")
        pass

def length_string_check(splitted_coords, length):  #check the length of the splitted string output
    for i in splitted_coords:
        if i == '':
            return False
    if len(splitted_coords) == length:
        return True
    else:
        return False

def split_string(string_to_split):          #splits the string
    x_part = ""
    y_part = ""
    z_part = ""
    w_part = ""
    p_part = ""
    r_part = ""
    v_part = ""
    changepart = 0

    for part in string_to_split:
        if part != ',':
            if changepart == 0:
                x_part = x_part + part
            if changepart == 1:
                y_part = y_part + part
            if changepart == 2:
                z_part = z_part + part
            if changepart == 3:
                w_part = w_part + part
            if changepart == 4:
                p_part = p_part + part
            if changepart == 5:
                r_part = r_part + part
            if changepart == 6:
                v_part = v_part + part
        else:
            changepart = changepart + 1

    splittedString = [x_part,y_part,z_part,w_part,p_part,r_part,v_part]


    return splittedString


def convert_str_to_float(list_to_float):         #converts string to float
    # all coordinates float
    floated_list = []
    for i in list_to_float:
        i = float(i)
        floated_list.append(i)
    for x in floated_list:
        if type(x) != float:
            print("Type not float")
            return False
    return floated_list

class DCS:                              #Virtual DCS coordinates, which coords are valid to send?
    def x():
        x = [float(440),float(900)]
        return x
    def y():
        y = [float(-450), float(350)]
        return y
    def z():
        z = [float(-455),float(-250)]
        return z
    def w():
        w = [float(-10),float(10)]
        return w
    def p():
        p = [float(-10),float(10)]
        return p
    def r():
        r = [float(-180),float(180)]
        return r

def between(value, lower, higher, included):           #determines if a val is between 2 (not)included values
    # checks if value is between values
    if included == True:
        if ((lower<= value) and (value <= higher) == True):
            return True
        else:
            return False
    if included == False:
        if ((lower < value) and (value < higher) == True):
            return True
        else:
            return False


def DCS_SafetyZone(coords):                                     #checks if coord is between the min and max DCS val
    # are coordinates in conflict with the DCS_Safetyzone
    DCSx = between(coords[0],DCS.x()[0],DCS.x()[1],True)
    DCSy = between(coords[1],DCS.y()[0],DCS.y()[1],True)
    DCSz = between(coords[2], DCS.z()[0], DCS.z()[1],True)
    DCSw = between(coords[3], DCS.w()[0], DCS.w()[1],True)
    DCSp = between(coords[4], DCS.p()[0], DCS.p()[1],True)
    DCSr = between(coords[5], DCS.r()[0], DCS.r()[1],True)

    DCS_zone = [DCSx, DCSy, DCSz, DCSw, DCSp, DCSr]
    for i in DCS_zone:
        if i == False:
            return False
    return True


def conditioning(data_input):                       #conditioning is the collect func for all conditioning funcs
    data = decode(data_input)
    print (data)
    splitted_string = split_string(data)
    if length_string_check(splitted_string,7) == False:
        print ("length string not OK")
        return False
    print(splitted_string)
    converted_string = convert_str_to_float(splitted_string)
    if converted_string == False:
        print ("STR convert not OK")
        return False
    if DCS_SafetyZone(converted_string) == False:
        print("DCS not OK")
        return False
    return True

def put_in_thread(data_to_thread):          # put conditioned data in thead
    thread.append(data_to_thread)
    if len(thread) > 1:
        del thread[0]

def get_from_thread(thread_list):           #retreive from thread
    if len(thread_list) == 0:
        return False
    else:
        return thread_list[-1]

def is_send():                          #if confirmation received, delete from thread
    # try: del thread[0]
    try: del thread[:]
    except: pass



def start():
    global conn1, conn2, addr1, addr2
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((SERVER, PORT))
        print("Bind done")
        s.listen()
        print("am listening")

        conn, addr = s.accept()             # connecting the clients
        if addr[0] == ip_self:
            conn1 = conn
            addr1 = addr
            print('Connected by PC', addr1, ',waiting for ROBOT.....')
        if addr[0] == ip_robot:
            conn2 = conn
            addr2 = addr
            print('Connected by ROBOT', addr2, ',waiting for PC.....')
        conn, addr = s.accept()
        if addr[0] == ip_self:
            conn1 = conn
            addr1 = addr
            print('Connected by PC', addr1, ',waiting for ROBOT.....')
        if addr[0] == ip_robot:
            conn2 = conn
            addr2 = addr
            print('Connected by ROBOT', addr2, ',waiting for PC.....')

def conn_receive():
    try:
        conn1.settimeout(0.05)
        datacon1 = conn1.recv(1024)
        return ['1', datacon1]
    except:
        pass
    try:
        conn2.settimeout(0.05)
        datacon2 = conn2.recv(3)
        return ['2', datacon2]
    except:
        return ['0','0']


if __name__ == '__main__':
    start()
    while True:
        mess_conn1_received = False
        mess_conn2_received = False
        if ((INMOTION == True) and (LOOPBAND == True)):
            time.sleep(0.1)
            conn1.sendall(POST_MSG.encode(FORMAT))
            INMOTION = False
            LOOPBAND = False
        if (EMPTY == True):
            time.sleep(0.2)
            conn1.sendall(POST_MSG.encode(FORMAT))
            EMPTY = False
            LOOPBAND = False

        com_msg = conn_receive() #receive messages

        # determine which msg is from which client
        if com_msg[0] == '1':
            mess_conn1 = com_msg[1]
            mess_conn1_received = True
        if com_msg[0] == '2':
            mess_conn2 = com_msg[1]
            mess_conn2_received = True
        else:
            pass


        if mess_conn1_received == True: # what to do if conn1 has send a msg
            print ('Server: following msg received from PC ::: ', mess_conn1)
            mess1_valid = conditioning(mess_conn1)
            print("Server: CONDITION = :", mess1_valid)
            if mess1_valid == True:
                put_in_thread(mess_conn1)
                conn1.sendall(APPROVED_MSG_PC.encode(FORMAT))
            else:
                conn1.sendall(DISAPPR_MSG_PC.encode(FORMAT))  # send error back to PC client


        elif mess_conn2_received == True: # what to do  if conn2 has send a msg
            mess2 = mess_conn2.decode(FORMAT)
            mess2 = str(mess2)

            if (mess2 == ROBOT_MSG_REQUEST): #request
                print ("Robot: 'MSG REQUEST'")
                data_to_send = get_from_thread(thread)
                if data_to_send == False:
                    print("Server: 'EMPTY THREAD'")
                    conn2.sendall(EMPTY_MSG_ROBOT.encode(FORMAT))
                    EMPTY = True

                else:
                    conn2.sendall(data_to_send)
                    MSG_REVEIVED = False
            elif mess2 == ROBOT_MSG_RECEIVED: # Reveived
                print ("Robot: 'MSG RECEIVED'")
                MSG_RECEIVED == True
                is_send()
            elif mess2 == IN_MOTION_MSG:
                print ("Robot:  '!!! IN MOTION !!!' ")
                INMOTION = True
                # conn1.sendall(IN_MOTION_MSG.encode(FORMAT))
            elif mess2 == LOOPBAND_MSG :
                print ("Robot:  'LOOPBAND_MSG' ")
                LOOPBAND = True
                # conn1.sendall(LOOPBAND_MSG.encode(FORMAT))
            else:
                print ("Server: 'DATA ERROR'")
                MSG_ERROR = 'ERR'
                # conn1.sendall(MSG_ERROR.encode(FORMAT))
        elif MSG_RECEIVED == False:
            try:
                time.sleep(0.5)
                conn2.sendall(data_to_send)
            except: pass

        else: pass



#560,400,-184,0,0,86

#moge beun upload gedaan mah











