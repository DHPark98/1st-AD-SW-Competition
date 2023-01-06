'''
키 조작

w : 속도 +1
s : 속도 -1
a : 회전 -1     (- : 좌회전, + : 우회전)
d : 회전 +1

r : 일시정지
f : 종료
'''


# from socket import *
import string
import keyboard
import time
import serial
import cv2
import time
import uuid
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from bird_eye_utils import *


# Variables
ser = serial.Serial()
ser.port = '/dev/ttyUSB0'      ### 아두이노 우노 (디버그용)
# ser.port = '/dev/ttyACM0'       ### 아두이노 메가
ser.baudrate = 9600

message = '0 0 '
message_prev = '0 0 '
direction = 0
speed = 0

keys = ['w', 'a', 'd', 's', 'r', 'o', 'f', 'c']  # control keys ('o' : meaningless key)
time_prev_read = time.time()
time_prev_write = time.time()

FPS = 15        ### FPS for Read
FWPS = 0.1      ### FPS for Write

image_width = 640
image_height = 480

# Functions
def dir_and_speed(input_key, direction, speed):
    brk = 0     # loop break
    if input_key == 'f':
        direction = 0
        speed = 0
        print('program finish')
        brk = 1
    elif input_key == 'w':#직진
        print('you pressed w')
        speed += 50
    elif input_key == 'a':#좌진
        print('you pressed a')
        direction -= 1
    elif input_key == 'd':#우진
        print('you pressed d')
        direction += 1 
    elif input_key == 's':#후진
        print('you pressed s')
        speed -= 50
    elif input_key == 'r':#stop
        print('you pressed r')
        direction = 0
        speed = 0
    elif input_key == 's':
        direction = 0   #정지
        speed = 0
    elif input_key == 'o':
        direction += 0   #정지
        speed += 0

    if (speed>250) :
        speed = 250
    elif (speed < -250):
        speed = -250
    if(direction < -7):
        direction = -7
    elif (direction > 7):
        direction = 7

    return (direction, speed, brk)

def receive_from_Ard():     # Argument : ser ?
    ser.flushInput()
    ser.flushOutput()
    if ser.readable():
        res = ser.readline()
        res = res.decode()[:len(res)-1]     # "angle : 0 straight :0 Read/Map [A0]/[b]: 575 / 31"
        direction_cur = res[-3:]        # read b (= 31)
        print('mapped_angle: ', direction_cur)
    print('------------------')

    return direction_cur


if __name__ == '__main__':
    cap_f = cv2.VideoCapture(0)     ###
    cap_f.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)      # 864
    cap_f.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)     # 480

    cap_b = cv2.VideoCapture(2)     ###
    cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)      # 864
    cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)     # 480

    print(cv2.__version__) 
    print(cap_f.isOpened())
    print(cap_b.isOpened())

    # FPS 확인
    fps_f = cap_f.get(cv2.CAP_PROP_FPS)     
    print('fps front', fps_f)
    fps_b = cap_b.get(cv2.CAP_PROP_FPS)     
    print('fps back', fps_b)

    ser.open()
    time.sleep(2)

    while True:
        retval_h, img_f = cap_f.read()  # left cam
        retval_b, img_b = cap_b.read()  # right cam
        time_pass_read = time.time() - time_prev_read
        time_pass_write = time.time() - time_prev_write
        input_key = 'o'

        # check whether one of the control keys is pressed
        for k in keys:
            if keyboard.is_pressed(k):
                input_key = k

        # Communicate with Arduino
        if (time_pass_read > 1./ FPS) :
            direction, speed, brk = dir_and_speed(input_key, direction, speed)
            if (brk == 1):
                break
            message = str(direction) +  ' ' + str(speed) + ' '
            ser.write(message.encode())      
            print(message)
            time_prev_read  = time.time()

            # mapped_besistance from Arduino (rename to direction_cur)
            direction_cur = receive_from_Ard()

            img_f = cv2.resize(img_f, (image_width, image_height))
            img_b = cv2.resize(img_b, (image_width, image_height))

            img_p_f = total_function(img_f, 'front')     # image processed front
            img_p_b = total_function(img_f, 'back')     # image processed back
            
            cv2.imshow('VideoCombined', img_p_f)
            cv2.imshow('VideoCombined', img_p_b)

            print(img_p_f.shape)

        # Write(Store) Image
        if ( (time_pass_write > 1./FWPS or input_key == 'c' ) and direction_cur != -100):
            path_cur = os.path.dirname(os.path.abspath(__file__))
            path = path_cur + '/data_img/'
            img_title = str(message) + str(uuid.uuid1())
            cv2.imwrite(path+img_title+".png", img_p_f)
            time_prev_write = time.time()

        # Break Loop
        if cv2.waitKey(25) == ord('f') :
            break

    ser.close()
    if cap_f.isOpened():
        cap_f.release()
    if cap_b.isOpened():
        cap_b.release()
    cv2.destroyAllWindows()