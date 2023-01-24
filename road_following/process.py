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
        self.front_camera_module = None
        self.play_type = play_name
        self.cam_num = {"FRONT" : 2, "REAR" : 4}
        self.front_cam_name = front_cam_name
        self.rear_cam_name = rear_cam_name
        
        self.rf_weight_file = rf_weight_file
        self.detect_weight_file = detect_weight_file
        
        self.serial = serial.Serial()
        self.serial.port = '/dev/ttyUSB0'       ### 아두이노 메가
        self.serial.baudrate = 9600
        self.speed = 200
        self.direction = 0
        if play_name == "Driving":
            self.rf_network = model.ResNet18(weight_file = self.rf_weight_file)
        self.detect_network = DetectMultiBackend(weights = detect_weight_file)
        self.labels_to_names = {0 : "Crosswalk", 1 : "Green", 2 : "Red", 3 : "Car"}
        
        self.lidar_port = '/dev/ttyUSB1'

        self.avoidance_processor = avoidance(self.serial, left_log='left_move.txt',right_log='right_move.txt')
        self.parking_log = parking_log
        self.parking_processor = idealparking(self.serial, self.parking_log)
        self.parking_stage = 1
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
            self.rear_camera_module = Devices.Camera.CameraModule(width=640, height=480)
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
            self.lidar_info = self.lidar_module.get_info()
            print(self.lidar_info)

            self.lidar_health = self.lidar_module.get_health()
            print(self.lidar_health)
            return True
        except:
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
                    cv2.imshow("bird_raw",bird_img)
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

                        detect, order_flag = object_detection(pred)
                    
                    # do not cut roi when turn right case
                    
                    # if(0):
                    #     road_gradient, bottom_value = dominant_gradient(preprocess_img, preprocess_img)
                    #     if road_gradient < 0:
                    #         road_gradient, bottom_value = dominant_gradient(roi_img, preprocess_img)

                    road_gradient, bottom_value = dominant_gradient(roi_img, preprocess_img)
                        

                    if (road_gradient == None and bottom_value == None): # Gradient가 없을 경우 예외처리(Exception Image)
                        self.direction = 0
                        message = 'a' + str(self.direction) +  's' + str(self.speed) +'o' + str(outside)
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
                        if is_outside(preprocess_img) == True:
                            self.direction = -7
                        else:
                            self.direction = 7

                        # self.avoidance_processor.action(preprocess_img)
                        
                        pass

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
            
            if cv2.waitKey(25) == ord('f') :
                end_message = "a0s0o0"
                self.serial.write(end_message.encode())
                self.serial.close()
                if self.front_camera_module:
                    self.front_camera_module.close_cam()
                    cv2.destroyAllWindows()
                
                print("Program Finish")
                
                break
            
            time.sleep((0.0003))
    
    def Parking(self):
        """
        1. Search Parking location
        => 잠깐 정지 후에 Parking Position 연산 => 대표값으로 연산
        2. Ideal Parking Position
        3. Action
        """
        detect_queue = 0
        self.lidar_start()
        for i, scan in enumerate(self.lidar_module.iter_scans()):
            #print('%d: Got %d measurments' % (i, len(scan)))

            scan = np.array(scan, dtype=np.int16)
            #scan_right = scan[np.where(s)]
            # print(scan)
            lidar_detect_condition = (scan[:,1]<305) & (scan[:,1]>275)
            #print(scan[np.where(lidar_detect_condition)])
            if len(np.where(lidar_detect_condition)[0]) > 0:
                detect_queue *= 2
                detect_queue += 1
                detect_queue %= 32
            else:
                detect_queue *= 2
                detect_queue %= 32
            #print("detect queue: ", detect_queue)
            if detect_queue == 0:
                print("object not detected")
            else:
                print("object detected")

            if i > 500:
                break

        self.lidar_finish()


        while True:
            try:
                if self.front_camera_module == None or self.rear_camera_module == None:
                    print("Please Check Camera module")
                    break
                    pass
                else:
                    
                    if  self.parking_stage == 1:
                        """
                        직진하며 주차 공간 탐색
                        라이다로 어느 위치에 주차할지 탐색
                        """
                        if True:
                            self.parking_stage = 2
                            pass
                        pass
                    elif    self.parking_stage == 2:
                        """
                        Ideal Parking Location으로 이동
                        """
                        if True:
                            """"Ideal Location 이동 성공"""
                            self.parking_stage = 3
                            pass
                        pass
                    elif    self.parking_stage == 3:
                        """
                        parking action
                        """
                        self.parking_processor.action(parking_location = 3)
                        if True:
                            """paking finish"""
                            if self.front_camera_module and self.rear_camera_module:
                                self.rear_camera_module.close_cam()
                                self.front_camera_module.close_cam()
                                cv2.destroyAllWindows()
                                end_message = "a0s0o0"
                                self.serial.write(end_message.encode())
                                self.serial.close()
                                print("Parking Finish")
                    
                                break
                        pass
                    pass
            except Exception as e:
                if self.front_camera_module and self.rear_camera_module:
                    print("Exception occur")
                    self.front_camera_module.close_cam()
                    self.rear_camera_module.close_cam()
                    end_message = "a0s0o0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                break
                pass
            except KeyboardInterrupt:
                if self.front_camera_module and self.rear_camera_module:
                    print("Keyboard Interrupt occur")
                    self.front_camera_module.close_cam()
                    self.rear_camera_module.close_cam()
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
                end_message = "a0s0o0"
                self.serial.write(end_message.encode())
                self.serial.close()
                print("Program Finish")
                
                break
            
            time.sleep((0.0001))
        pass
                
                
        
        
