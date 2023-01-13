import Devices.Camera
from Algorithm.BirdEyeConverter import *
from Networks import processing
from Networks import model
import serial
import time
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

class DoWork:
    def __init__(self, play_name, cam_name, rf_weight_file, detect_weight_file = None):
        self.camera_module = None
        self.play_type = play_name
        self.cam_num = {"FRONT" : 4, "REAR" : 2}
        self.cam_name = cam_name
        self.rf_weight_file = rf_weight_file
        self.detect_weight_file = detect_weight_file
        
        self.serial = serial.Serial()
        self.serial.port = '/dev/ttyUSB0'       ### 아두이노 메가
        self.serial.baudrate = 9600
        self.speed = 30
        self.direction = 0
        self.rf_network = model.ResNet18(weight_file = self.rf_weight_file)
        self.detect_network = DetectMultiBackend(weights = detect_weight_file, device = "cuda")
        
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
                    
                    if self.detect_weight_file != None:
                        pred = self.detect_network(cam_img, )
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    
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
            
            if cv2.waitKey(25) == ord('f') :
                end_message = "a0s0"
                self.serial.write(end_message.encode())
                self.serial.close()
                cv2.destroyAllWindows()
                break
            
            time.sleep((0.00001))
            
                
                
        
        
        
        