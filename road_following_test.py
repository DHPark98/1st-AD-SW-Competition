import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 15)

model.load_state_dict(torch.load('best_steering_model_0110.pth', "cuda:0"))

device = "cuda"
model = model.to(device)
model = model.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(imagefromarray((image*255).astype(np.uint8)).convert('RGB')
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# from socket import *
import time
import serial
import cv2
import time
import uuid
import os
import sys
from bird_eye_utils import *

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# Variables
ser = serial.Serial()
# ser.port = '/dev/ttyUSB0'      ### 아두이노 우노 (디버그용)
ser.port = '/dev/ttyUSB0'       ### 아두이노 메가
ser.baudrate = 9600

direction = 0
speed = 30
FPS = 15        ### FPS for Read
FWPS = 0.1 

if __name__ == '__main__':
    cap_f = cv2.VideoCapture(4)    ###
    cap_f.set(cv2.CAP_PROP_BUFFERSIZE, 1)           
    cap_f.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)      # 864
    cap_f.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)     # 480
   
    # cap_b = cv2.VideoCapture(2)     ###
    # cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)      # 864
    # cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)     # 480

    print(cv2.__version__) 
    print(cap_f.isOpened())
    # print(cap_b.isOpened())

    # FPS 확인
    fps_f = cap_f.get(cv2.CAP_PROP_FPS)     
    print('fps front', fps_f)
    # fps_b = cap_b.get(cv2.CAP_PROP_FPS)     
    # print('fps back', fps_b)

    ser.open()
    time.sleep(2)

    while True:
        retval_f, img_f = cap_f.read()  # front cam
        # retval_b, img_b = cap_b.read()  # behind cam
        # time_pass_read = time.time() - time_prev_read

        # Communicate with Arduino
        if (retval_f == True) :
            # time_prev_read  = time.time()

            # img_f = cv2.resize(img_f, (image_width, image_height))
            # img_b = cv2.resize(img_b, (image_width, image_height))
            
            img_p_f = total_function(img_f, 'front')
            
            direction = torch.argmax(model(preprocess(img_p_f))) - 7
            
            message = 'a' + str(direction) +  's' + str(speed)
            ser.write(message.encode())
            
            print("Current Direction is {}".format(direction))
            
            
            cv2.imshow('VideoCombined', img_f)
            # cv2.imshow('VideoCombined', img_b)

        # Break Loop
        if cv2.waitKey(25) == ord('f') :
            break

    ser.close()
    if cap_f.isOpened():
        cap_f.release()
    if cap_b.isOpened():
        cap_b.release()
    cv2.destroyAllWindows()

