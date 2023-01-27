import sys
import os
from pathlib import Path
PATH = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
sys.path.append(PATH)
from utility import object_detection, box_area, box_center, center_inside
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

class avoidance():
    def __init__(self, serial, camera_module, detect_weight_file, speed):
        self.serial = serial
        self.front_camera_module = camera_module
        self.detect_weight_file = detect_weight_file
        self.detect_network = DetectMultiBackend(weights = self.detect_weight_file)
        self.speed = speed
        pass
    def action(self, outside_flag):
        try:
            box_threshold = 21000
            while True:
                cam_img = self.front_camera_module.read()
                pred = self.detect_network(cam_img)
                pred = non_max_suppression(pred)[0]
                
                detect, _ = object_detection(pred)
                car_bbox = detect[3]
                
                bbox_center = box_center(car_bbox)
                bbox_area = box_area(car_bbox)
                if outside_flag == True:
                    
                    if center_inside(bbox_center) == True and bbox_area > box_threshold:
                        direction = -7
                    else:
                        break
                else:
                    if center_inside(bbox_center) == True and bbox_area > box_threshold:
                        direction = 7
                    else:
                        break
                        
                message = "a" + str(direction) + "s" + str(self.speed) + "o0"
                self.serial.write(message.encode())
                

            
            return True
            pass
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("avoidance error = {}, error line = {}".format(e, tb.tb_lineno))
            return False
            pass
