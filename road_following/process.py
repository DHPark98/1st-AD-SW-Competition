import Devices.Camera
import Devices.rplidar
from Algorithm.BirdEyeConverter import *
from Networks import model
import serial
import time
import torch
import torchvision.transforms as transform
import sys
rf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(rf_dir, "yolov5"))
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from utility import (roi_cutting, preprocess, show_bounding_box, 
                    object_detection, dominant_gradient, cvt_binary, 
                    return_road_direction, is_outside, box_area)
from Algorithm.Control import total_control, smooth_direction
from Algorithm.img_preprocess import total_function
from Algorithm.object_avoidance import avoidance
from Algorithm.ideal_parking import idealparking
class DoWork:
    def __init__(self, play_name, front_cam_name, rear_cam_name, rf_weight_file = None, detect_weight_file = None, parking_log = None):
        self.play_type = play_name
        
        # Camera
        self.front_camera_module = None
        self.rear_camera_module = None
        self.cam_num = {"FRONT" : 2, "REAR" : 4}
        self.front_cam_name = front_cam_name
        self.rear_cam_name = rear_cam_name
        
        # Model
        self.rf_weight_file = rf_weight_file
        self.detect_weight_file = detect_weight_file
    
        self.rf_network = model.ResNet18(weight_file = self.rf_weight_file) if play_name == "Driving" else None
        self.detect_network = DetectMultiBackend(weights = detect_weight_file)
        self.labels_to_names = {0 : "Crosswalk", 1 : "Green", 2 : "Red", 3 : "Car"}
        
        # Arduino Serial
        self.serial = serial.Serial()
        self.serial.port = '/dev/ttyUSB0'       ### 아두이노 메가
        self.serial.baudrate = 9600
        
        # Control
        self.speed = 0
        self.parking_speed = 0
        self.direction = 0
        
        # Lidar
        self.lidar_port = '/dev/ttyUSB1'
        self.lidar_module = None

    def serial_start(self):
        try:
            self.serial.open()
            print("Serial open")
            time.sleep(1)
            return True
        
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("serial start error = {}, error line = {}".format(e, tb.tb_lineno))
            return False
    
    def front_camera_start(self):
        try:
            self.front_camera_module = Devices.Camera.CameraModule(width=640, height=480)
            self.front_camera_module.open_cam(self.cam_num[self.front_cam_name])
            print("FRONT Camera open")
            return True
        
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("front camera start error = {}, error line = {}".format(e, tb.tb_lineno))
            return False
        
    def rear_camera_start(self):
        try:
            self.rear_camera_module = Devices.Camera.CameraModule(width=640, height=360)
            self.rear_camera_module.open_cam(self.cam_num[self.rear_cam_name])
            print("REAR Camera open")
            return True
        
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("rear camera start error = {}, error line = {}".format(e, tb.tb_lineno))
            return False
        
    def lidar_start(self):
        try:
            self.lidar_module = Devices.rplidar.RPLidar(self.lidar_port)
            print("Lidar open")
            return True
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("lidar start error = {}, error line = {}".format(e, tb.tb_lineno))
            return False
    
    def lidar_finish(self):
        self.lidar_module.stop()
        self.lidar_module.stop_motor()
        self.lidar_module.disconnect()

    def Driving(self):
        bef_1d, bef_2d, bef_3d = 0,0,0
        while True:
            try:
                if self.front_camera_module == None:
                    print("Please Check Camera module")
                    break
                    pass
                else:
                    cam_img = self.front_camera_module.read()
                    bird_img = bird_convert(cam_img, self.front_cam_name)
                    
                    preprocess_img = total_function(bird_img)
                    binary_img = cvt_binary(bird_img)
                    roi_img = roi_cutting(binary_img)
                    
                    draw_img = cam_img.copy()
                    
                    outside = int(is_outside(preprocess_img))

                    
                    order_flag = 1
                    
                    if self.detect_weight_file != None: # Detection 했을 경우
                        image = preprocess(cam_img, "test", device = "cpu")
                        pred = self.detect_network(image)
                        pred = non_max_suppression(pred)[0]
                        
                        draw_img = show_bounding_box(draw_img, pred)

                        _, order_flag = object_detection(pred)

                    road_gradient, bottom_value = dominant_gradient(roi_img, preprocess_img)
                        

                    if (road_gradient == None and bottom_value == None): # Gradient가 없을 경우 예외처리(Exception Image)
                        self.direction = 0
                        message = 'a' + str(bef_1d) +  's' + str(self.speed) +'o' + str(outside)
                        self.serial.write(message.encode())
                        print(message)
                        continue    


                    print('grad: ',road_gradient)
                    print('bottom: ', bottom_value)
                    road_direction = return_road_direction(road_gradient)
                    model_direction = torch.argmax(self.rf_network.run(preprocess(roi_img, mode = "test"))).item() - 7
                    final_direction = total_control(road_direction, model_direction, bottom_value)

                    if order_flag == 0: # stop
                        self.direction = 0
                        self.speed = 0
                        pass
                    elif order_flag == 1: # go
                        # self.direction = final_direction
                        self.direction = smooth_direction(bef_1d, bef_2d, bef_3d, final_direction)
                        pass
                    
                    elif order_flag == 2: # road change
                        print("road change!")
                        avoidance_processor = avoidance(self.serial, self.front_camera_module, 
                                                        self.detect_weight_file, self.speed)
                        avoidance_processor(is_outside(preprocess_img))

                    message = 'a' + str(self.direction) +  's' + str(self.speed) + 'o' + str(outside)
                    self.serial.write(message.encode())
                    print(message)
                    
                    cv2.imshow('VideoCombined_detect', draw_img)
                    cv2.imshow('VideoCombined_rf', roi_img)
                    cv2.imshow('VideoCombined_rf2', preprocess_img)
                    
                    
                    bef_1d, bef_2d, bef_3d = self.direction, bef_1d, bef_2d
                    pass
                
            except Exception as e:
                if self.front_camera_module:
                    _, _, tb = sys.exc_info()
                    print("process error = {}, error line = {}".format(e, tb.tb_lineno))
                    self.front_camera_module.close_cam()
                    end_message = "a0s0o0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                break
                pass
            except KeyboardInterrupt:
                if self.front_camera_module:
                    print("Keyboard Interrupt occur")
                    self.front_camera_module.close_cam()
                    end_message = "a0s0o0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                break
                pass
            
            if cv2.waitKey(25) == ord('f'):
                end_message = "a0s0o0"
                self.serial.write(end_message.encode())
                self.serial.close()
                if self.front_camera_module:
                    self.front_camera_module.close_cam()
                    cv2.destroyAllWindows()
                
                print("Program Finish")
                
                break
            
            time.sleep((0.0001))
    
    def Parking(self):
        """
        1. Search Parking location
        => 잠깐 정지 후에 Parking Position 연산 => 대표값으로 연산
        2. Ideal Parking Position
        3. Action
        """
        

        near_detect_condition = ((-90 < scan[:,0]) & (scan[:,0] < 90)) & (scan[:,1] < 500)
        car_search_condition = (((-90 < scan[:,0]) & (scan[:,0] < -80)) | ((80 < scan[:,0]) & (scan[:,0] < 90))) & (scan[:,1] < 2000)
        car_left_condition = ((80 < scan[:,0]) & (scan[:,0] < 90)) & (scan[:,1] < 2000)
        car_right_condition = ((-90 < scan[:,0]) & (scan[:,0] < -80)) & (scan[:,1] < 2000)
        car_detect_queue = 0
        near_detect_queue = 0
        parking_stage = -1
        new_car_cnt = 0
        obj = False
        parking_direction = 0
        while True:
            try:
                if self.front_camera_module == None or self.rear_camera_module == None:
                    print("Please Check Camera module")
                    break
                    pass
                if self.lidar_module == None:
                    print("Please Check Lidar module")
                    break
                    pass
                front_cam_img, rear_cam_img = self.front_camera_module.read(), self.rear_camera_module.read()
                
                front_bev, rear_bev = bird_convert(front_cam_img, self.front_cam_name), bird_convert(rear_cam_img, self.rear_cam_name)
                
                front_prep_img, rear_prep_img = total_function(front_bev), total_function(rear_bev)
                
                front_binary_img, rear_binary_img = cvt_binary(front_prep_img), cvt_binary(rear_prep_img)
                
                scan = np.array(self.lidar_module.iter_scans())
                car_scan = scan[np.where(car_search_condition)]
                near_scan = scan[np.where(near_detect_condition)]
                
                if parking_stage == -1:
                    print(near_scan)
                    pass
                
                
                if parking_stage == 0:
                    if len(np.where(car_search_condition)[0]):
                        car_detect_queue = (car_detect_queue * 2 + 1) % 64
                    else:
                        car_detect_queue = (car_detect_queue * 2) % 64
                    
                    print(scan)
                    if car_detect_queue == 0:
                        print('car not detected')
                        obj = False
                    else:
                        print('car detected')
                        if obj == False:
                            new_car_cnt +=1
                        obj = True
                    
                    if new_car_cnt == 2:
                        parking_stage = 1
                        obj = False
                        if len(np.where(car_left_condition)[0]):
                            parking_direction = -1
                        elif len(np.where(car_right_condition)[0]):
                            parking_direction = 1
                        else:
                            parking_direction = 0
                        pass

                    self.direction = 0
                    
                    
                elif parking_stage == 1:
                    self.parking_speed *= -1
                    self.direction = parking_direction * 7
                    
                    if len(np.where(near_detect_condition)[0]):
                        self.parking_speed = 0
                        parking_stage = 2
                    pass
                
                elif parking_stage == 2:
                    self.direction = 0
                    """
                    직진하며 앞라인 검출 시작
                    """
                
                
                    
                
                
                
                message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                self.serial.write(message.encode())
                
                
                    
                
                cv2.imshow("video_original_f", front_cam_img)
                cv2.imshow("video_original_r", rear_cam_img)
                cv2.imshow("video_binary_f", front_prep_img)
                cv2.imshow("video_binary_r", rear_prep_img)
                
                
                
                
            except Exception as e:
                if self.front_camera_module and self.rear_camera_module:
                    print("Exception occur")
                    self.front_camera_module.close_cam()
                    self.rear_camera_module.close_cam()
                    cv2.destroyAllWindows()
                if self.lidar_module:
                    self.lidar_finish()
                    end_message = "a0s0o0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                    self.lidar_finish()
                break
                pass
            except KeyboardInterrupt:
                if self.front_camera_module and self.rear_camera_module:
                    print("Keyboard Interrupt occur")
                    self.front_camera_module.close_cam()
                    self.rear_camera_module.close_cam()
                    cv2.destroyAllWindows()
                if self.lidar_module:
                    self.lidar_finish()
                end_message = "a0s0o0"
                self.serial.write(end_message.encode())
                self.serial.close()
                
                break
                pass
            
            if cv2.waitKey(25) == ord('f') :
                if self.front_camera_module and self.rear_camera_module:
                    self.rear_camera_module.close_cam()
                    self.front_camera_module.close_cam()
                    cv2.destroyAllWindows()
                if self.lidar_module:
                    self.lidar_finish()
                end_message = "a0s0o0"
                self.serial.write(end_message.encode())
                self.serial.close()
                self.lidar_finish()
                print("Program Finish")
                
                break
            
            time.sleep((0.0001))
        pass
                
                
        
        
