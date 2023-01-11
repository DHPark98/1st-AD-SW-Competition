import Devices.Camera
from Algorithm.BirdEyeConverter import *
from Networks import processing
from Networks import model
import serial
import time
import torch

class DoWork:
    def __init__(self, play_name, cam_name, weight_file):
        self.camera_module = None
        self.play_type = play_name
        self.cam_num = {"FRONT" : 4, "REAR" : 2}
        self.cam_name = cam_name
        self.weight_file = weight_file
        self.serial = serial.Serial()
        self.serial.port = '/dev/ttyUSB0'       ### 아두이노 메가
        self.serial.baudrate = 9600
        self.speed = 30
        self.direction = 0
        self.network = model.ResNet18(weight_file = self.weight_file)
    def serial_start(self):
        try:
            self.serial.open()
            print("Serial open")
            time.sleep(1)
            return True
        
        except Exception as _:
            return False
    
    def camera_start(self):
        try:
            self.camera_module = Devices.Camera.CameraModule(width=640, height=480)
            self.camera_module.open_cam(self.cam_num[self.cam_name])
            print("FRONT Camera open")
            return True
        
        except Exception as _:
            return False
        
        
    def Dowork(self):
        while True:
            try:
                if self.camera_module == None:
                    break
                    pass
                else:
                    cam_img = self.camera_module.read()
                    bird_img = bird_convert(cam_img, self.cam_name)
                    self.direction = torch.argmax(self.network.run(processing.preprocess(bird_img))).item() - 7
                    
                    message = 'a' + str(self.direction) +  's' + str(self.speed)
                    self.serial.write(message.encode())
                    print("Current Direction is {}".format(self.direction))
                    cv2.imshow('VideoCombined', cam_img)
                    
                    pass
            except Exception as e:
                if self.camera_module:
                    self.camera_module.close()
                break
                pass
            except KeyboardInterrupt:
                if self.camera_module:
                    self.camera_module.close()
                break
                pass
            
            
            time.sleep((0.00001))
            
                
                
        
        
        
        