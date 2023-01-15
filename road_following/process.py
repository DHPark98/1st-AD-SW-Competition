import Devices.Camera
from Algorithm.BirdEyeConverter import *
from Networks import model
import serial
import time
import torch
import torchvision.transforms as transform
import sys
sys.path.append(os.path.join(os.path.abspath(__file__), "yolov5"))
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from utility import roi_cutting, preprocess, show_bounding_box, object_detection


class DoWork:
    def __init__(self, play_name, cam_name, rf_weight_file, detect_weight_file = None):
        self.camera_module = None
        self.play_type = play_name
        self.cam_num = {"FRONT" : 2, "REAR" : 4}
        self.cam_name = cam_name
        self.rf_weight_file = rf_weight_file
        self.detect_weight_file = detect_weight_file
        
        self.serial = serial.Serial()
        self.serial.port = '/dev/ttyUSB0'       ### 아두이노 메가
        self.serial.baudrate = 9600
        self.speed = 30
        self.direction = 0
        self.rf_network = model.ResNet18(weight_file = self.rf_weight_file)
        if self.detect_weight_file != None:
            self.detect_network = DetectMultiBackend(weights = detect_weight_file)
        
        
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
                    # roi_img = roi_cutting(bird_img)
                    
                    
                    draw_img = cam_img.copy()
                    
                    order_flag = 1
                    
                    if self.detect_weight_file != None:
                        image = transform.functional.to_tensor(cam_img)
                        image = image[None, ...]
                        pred = self.detect_network(cam_img)
                        pred = non_max_suppression(pred)[0]
                        
                        draw_img = show_bounding_box(draw_img)
                        
                        # order_flag = object_detection(pred)
                        
                    
                    self.direction = torch.argmax(self.rf_network.run(preprocess(bird_img, mode = "test"))).item() - 7 # bird_eye_view
                    # self.direction = torch.argmax(self.rf_network.run(preprocess(roi_img, mode = "test"))).item() - 7 # roi_view
                    
                    message = 'a' + str(self.direction) +  's' + str(self.speed)
                    self.serial.write(message.encode())
                    print(message)
                    cv2.imshow('VideoCombined', draw_img)
                    
                    
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
            
                
                
        
        
        
        
