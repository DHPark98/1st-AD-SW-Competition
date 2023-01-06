import cv2
import uuid
import time
import os

from bird_eye_utils import *

image_width = 640   # 640   864   
image_height = 480  # 360   480

# cap = cv2.VideoCapture('/dev/video2')
# cap = cv2.VideoCapture(2, cv2.CAP_V4L2)   # CAP_DSHOW : Microsoft, CAP_V4L2 : Linux
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
print(cv2.__version__) 
print(cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)     # FPS 확인
print('fps', fps)

img_idx = 0
while img_idx < 30:
    ret, frame = cap.read()
    if img_idx == 0:
        print(frame.shape)
        img_idx = 1
    if (ret is True):
        # print(frame.shape)
        cv2.imshow('frame', frame)
        frame_p = total_function(frame, 'front')
        # cv2.imshow('processed', frame_p)

        if cv2.waitKey(25) == ord('f') :
            break
