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

    if(direction < -4):
        direction = -4
    elif (direction > 4):
        direction = 4

    return (direction, speed, brk)

def receive_from_Ard():     # Argument : ser ?
    ser.flushInput()
    ser.flushOutput()
    if ser.readable():
        res = ser.readline()
        res = res.decode()[:len(res)-1]     # "angle : 0 straight :0 Read/Map [A0]/[b]: 575 / 31"
        direction_cur = res[-3:]        # read b (= 31)
        #print('mapped_angle: ', direction_cur)
    #print('------------------')

    return direction_cur


if __name__ == '__main__':
    cap_f = cv2.VideoCapture(4)    ###
    cap_f.set(cv2.CAP_PROP_BUFFERSIZE, 1)           
    cap_f.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)      # 864
    cap_f.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)     # 480
   
    cap_b = cv2.VideoCapture(2)     ###
    cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        retval_f, img_f = cap_f.read()  # left cam
        retval_b, img_b = cap_b.read()  # right cam
        #time_pass_read = time.time() - time_prev_read
        time_pass_read = time.time() - time_prev_read
        time_pass_write = time.time() - time_prev_write
        input_key = 'o'

        # check whether one of the control keys is pressed
        for k in keys:
            if keyboard.is_pressed(k):
                input_key = k

        # Communicate with Arduino
        if (retval_f is True):
        # if (time_pass_read > 1./ FPS) :

            direction, speed, brk = dir_and_speed(input_key, direction, speed)
            if (brk == 1):
                break
            message = str(direction) +  ' ' + str(speed) + ' '
            ser.write(message.encode())      
            print(message)
            #time_prev_read  = time.time()


            # mapped_besistance from Arduino (rename to direction_cur)
            direction_cur = receive_from_Ard()

            img_p_f = total_function(img_f, 'front')     # image processed front
            img_p_b = total_function(img_b, 'back')     # image processed back
            
            cv2.imshow('VideoCombined_f', img_f)
            cv2.imshow('VideoCombined_b', img_b)

            # print(img_p_f.shape)

        # Write(Store) Image
        
        
        if ( (time_pass_write > 1./FWPS or input_key == 'c' ) and direction_cur != -100):
            path = '/home/skkcar/Desktop/contest/data_img'
            time_stamp = str(time.time())
            uuid_cur = str(uuid.uuid1())
            
            img_f_title = path + 'f_' + str(message) + time_stamp + uuid_cur
            cv2.imwrite(path+img_f_title+".png", img_f)
            
            img_b_title = path + 'b_' + str(message) + time_stamp + uuid_cur
            # cv2.imwrite(path+img_b_title+".png", img_b)
            
            time_prev_write = time.time()
        
            img_p_f = total_function(img_f, 'front')     # image processed front
            img_p_b = total_function(img_f, 'back')     # image processed back
            
            cv2.imshow('VideoCombined', img_f)
            cv2.imshow('VideoCombined', img_b)
            
            #img_p_f_title = path + 'f_' + str(message) + time_stamp + uuid_cur
            #cv2.imwrite(path + img_p_f_title + ".png", img_p_f)
            #img_p_b_title = path + 'f_' + str(message) + time_stamp + uuid_cur
            #cv2.imwrite(path + img_p_b_title + ".png", img_p_b)
            

            print(img_p_f.shape)



        # Break Loop
        if cv2.waitKey(25) == ord('f') :
            break

    ser.close()
    if cap_f.isOpened():
        cap_f.release()
    if cap_b.isOpened():
        cap_b.release()

    cv2.destroyAllWindows()

