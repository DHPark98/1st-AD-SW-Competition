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
                    return_road_direction, is_outside, box_area, total_process,
                    distinguish_traffic_light)
from Algorithm.parking import (detect_parking_car, parking_steering_angle, return_parking_direction, good_parking)
from Algorithm.Control import total_control, smooth_direction, strengthen_control
from Algorithm.img_preprocess import total_function
from Algorithm.object_avoidance import avoidance
from Algorithm.ideal_parking import idealparking
from Devices.Lidar import LidarModule
class DoWork:
    def __init__(self, play_name, front_cam_name, rear_cam_name, rf_weight_file = None, detect_weight_file = None):
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
        self.speed = 250
        self.speed_value = self.speed

        self.parking_speed = 50
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
            self.lidar_module = LidarModule()
            print("Lidar open")
            return True
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("lidar start error = {}, error line = {}".format(e, tb.tb_lineno))
            return False    

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
                    cv2.imshow("bird", bird_img)
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
                        
                        pred = distinguish_traffic_light(draw_img, pred)
                        draw_img = show_bounding_box(draw_img, pred)

                        _, order_flag = object_detection(pred)
                        
                    road_gradient, bottom_value = dominant_gradient(roi_img, preprocess_img)
                    

                    if (road_gradient == None and bottom_value == None): # Gradient가 없을 경우 예외처리(Exception Image)
                        self.direction = 0
                        message = 'a' + str(bef_1d) +  's' + str(self.speed) +'o' + str(outside)
                        self.serial.write(message.encode())
                        print(message)
                        continue    

                    
                    # print('grad: ',road_gradient)
                    # print('bottom: ', bottom_value)
                    road_direction = return_road_direction(road_gradient)
                    # model_direction = torch.argmax(self.rf_network.run(preprocess(roi_img, mode = "test"))).item() - 7
                    # final_direction = total_control(road_direction, model_direction, bottom_value)
                    final_direction = strengthen_control(road_direction, bottom_value)
                    if order_flag == 0: # stop
                        self.direction = 0
                        self.speed = 0
                        pass
                    elif order_flag == 1: # go
                        # self.direction = final_direction
                        self.speed = self.speed_value
                        self.direction = smooth_direction(bef_1d, bef_2d, bef_3d, final_direction)
                        pass
                    
                    elif order_flag == 2: # road change
                        print("road change!")
                        avoidance_processor = avoidance(self.serial, self.front_camera_module, 
                                                        self.detect_weight_file, 50)
                        avoidance_processor.action(is_outside(preprocess_img))

                    message = 'a' + str(self.direction) +  's' + str(self.speed) + 'o' + str(outside)
                    self.serial.write(message.encode())
                    print()
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
                print(end_message)
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
        

        
        car_detect_queue = 0
        near_detect_queue = 0
        parking_stage = 0
        new_car_cnt = 0
        obj = False
        parking_direction = 0
        parking_speed = 40
        self.parking_speed = parking_speed
        detect_cnt = 0
        queue_key = 0
        parking_steering_weight = 5
        left_cnt = 0
        right_cnt = 0
        total_array = np.array([[-1, -1, -1]])
        distance_threshold = 250
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
                print("Parking stage : {}".format(parking_stage))
                scan = np.array(self.lidar_module.iter_scans())
                    
                car_search_condition = (((-100 < scan[:,0]) & (scan[:,0] < -90)) | ((90 < scan[:,0]) & (scan[:,0] < 100)))  & (scan[:,1] < 2000)
                near_detect_condition = ((-100 < scan[:,0]) & (scan[:,0] < 100)) & (scan[:,1] < 300)
                car_left_condition = ((90 < scan[:,0]) & (scan[:,0] < 100)) & (scan[:,1] < 2000)
                car_right_condition = ((-100 < scan[:,0]) & (scan[:,0] < -90)) & (scan[:,1] < 2000)
                detect_condition = ((-100 < scan[:,0]) & (scan[:,0] < 100)) & (scan[:,1] < 2000)
                straight_condition = ((-100 < scan[:,0]) & (scan[:,0] < 100)) & (scan[:,1] < 1200)

                left_condition = (((-100 < scan[:,0]) & (scan[:,0] < -80)))  & (scan[:,1] < 1000)
                right_condition = (((80 < scan[:,0]) & (scan[:,0] < 100)))  & (scan[:,1] < 1000)
                

                stop_condition = (((-90 < scan[:,0]) & (scan[:,0] < -80)) | ((80 < scan[:,0]) & (scan[:,0] < 90)))  & (scan[:,1] < 1000)

                if parking_stage == -1: # Lidar Test
                    # 
                    self.parking_speed = 0

                    scan = scan[np.where(detect_condition)]
                    print(parking_steering_angle(scan, queue_key))

                    queue_key = (queue_key + 1) % 10

                    pass
                
                
                if parking_stage == 0: # Search parking start position
                    # detect_parking_car(self.lidar_module, detect_cnt, new_car_cnt)
                    if len(np.where(car_search_condition)[0]):
                        car_detect_queue = (car_detect_queue * 2 + 1) % 32
                    else:
                        car_detect_queue = (car_detect_queue * 2) % 32

                    if car_detect_queue == 0:
                        
                        if detect_cnt == 0:
                            if obj == True:
                                new_car_cnt += 1
                            obj = False
                            # print('car not detected')
                            pass
                        else:
                            detect_cnt -= 1
                    else:
                        if detect_cnt == 5:
                            # print('car detected')
                            # if obj == False:
                            #     new_car_cnt += 1
                            obj = True
                        else:
                            detect_cnt += 1
                        
                    if new_car_cnt == 2:
                        print("Detect two car!")
                        # rest
                        self.parking_speed = 0
                        self.direction = 0
                        message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                        self.serial.write(message.encode())
                        time.sleep(0.3)
                        parking_stage = 1
                        obj = False
                        for i in range(5):
                            left_cnt += len(scan[np.where(car_left_condition)])
                            right_cnt += len(scan[np.where(car_right_condition)])

                        if left_cnt > right_cnt:
                            parking_direction = -1
                        else:
                            parking_direction = 1


                        # if len(np.where(car_left_condition)[0]):
                            
                        #     parking_direction = -1
                        # elif len(np.where(car_right_condition)[0]):
                        #     parking_direction = 1
                        # else:
                        #     parking_direction = 0
                        # pass

                    self.direction = 0 # 선 따라가도록 바꿀 예정
                    
                    
                elif parking_stage == 1:
                    self.parking_speed = -1 * parking_speed
                    self.direction = parking_direction * 7
                    
                    if len(np.where(near_detect_condition)[0]):
                        
                        # rest
                        self.parking_speed = 0
                        self.direction = 0
                        message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                        self.serial.write(message.encode())
                        time.sleep(0.3)
                        
                        parking_stage = 2
                    pass
                
                elif parking_stage == 2:
                    self.direction = 0
                    self.parking_speed = parking_speed
                    
                    if len(np.where(straight_condition)[0]) == 0:
                        self.parking_speed = 0
                        self.direction = 0
                        message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                        self.serial.write(message.encode())
                        time.sleep(0.3)
                        parking_stage = 3
                
                elif parking_stage == 3:
                    
                    detect_scan = scan[np.where(detect_condition)]
                    steering_angle = parking_steering_angle(detect_scan, queue_key, total_array)
                    # print(steering_angle)
                    self.direction = return_parking_direction(-1 * steering_angle)
                    self.parking_speed = parking_speed * -1
                    queue_key = (queue_key + 1) % 7
                    
                    if len(np.where(near_detect_condition)[0]):
                        
                        # rest
                        self.parking_speed = 0
                        self.direction = 0
                        message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                        self.serial.write(message.encode())
                        time.sleep(0.3)
                        
                        parking_stage = 2

                    if len(np.where(left_condition)[0]) and len(np.where(right_condition)[0]):

                        # rest
                        self.parking_speed = 0
                        self.direction = 0
                        message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                        self.serial.write(message.encode())
                        time.sleep(0.3)

                        # calculate distance
                        total_left_scan = []
                        total_right_scan = []
                        for i in range(5):
                            scan = np.array(self.lidar_module.iter_scans())

                            left_condition = (((-100 < scan[:,0]) & (scan[:,0] < -80)))  & (scan[:,1] < 1000)
                            right_condition = (((80 < scan[:,0]) & (scan[:,0] < 100)))  & (scan[:,1] < 1000)

                            left_scan = scan[np.where(left_condition)]
                            right_scan = scan[np.where(right_condition)]
                            
                            if i == 0:
                                total_left_scan = left_scan
                                total_right_scan = right_scan
                            else:
                                total_left_scan = np.concatenate((total_left_scan, left_scan), axis = 0)
                                total_right_scan = np.concatenate((total_right_scan, right_scan), axis = 0)
                            
                        left_distance = np.min(total_left_scan[:,1])
                        right_distance = np.min(total_right_scan[:,1])

                        if(abs(left_distance - right_distance) > distance_threshold):
                            parking_stage = 4
                        else:
                            parking_stage = 5
                            queue_key = 0
                            total_array = np.array([[-1, -1, -1]])

                elif parking_stage == 4:
                    self.direction = 0
                    self.parking_speed = parking_speed
                    
                    if len(np.where(straight_condition)[0]) == 0:
                        self.parking_speed = 0
                        self.direction = 0
                        message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                        self.serial.write(message.encode())
                        time.sleep(0.3)
                        parking_stage = 3     

                elif parking_stage == 5:
                    scan = np.array(self.lidar_module.iter_scans())
                    left_condition = (((-100 < scan[:,0]) & (scan[:,0] < -80)))  & (scan[:,1] < 1000)
                    right_condition = (((80 < scan[:,0]) & (scan[:,0] < 100)))  & (scan[:,1] < 1000)
                    left_scan = scan[np.where(left_condition)]
                    right_scan = scan[np.where(right_condition)]
                    print("left_scan : ", left_scan)
                    print("right_scan : ", right_scan)
                    left_angle = good_parking(left_scan, queue_key, total_array)
                    right_angle = good_parking(right_scan, queue_key, total_array)

                    steering_angle = (left_angle + right_angle) / 2

                    self.parking_speed = -1 * parking_speed

                    self.direction = return_parking_direction(steering_angle)

                    queue_key = (queue_key + 1) % 5

                    if len(np.where(stop_condition)[0]) == 0:
                        parking_stage = 10


                
                
                elif parking_stage == 10: # finish
                    self.direction = 0
                    self.parking_speed = 0
                    
                    
                    """
                    직진하며 앞라인 검출 시작
                    """
                
                
                    
                
                
                
                message = 'a' + str(self.direction) +  's' + str(self.parking_speed) +'o0'
                # print(message)
                self.serial.write(message.encode())
                
                
                    
                
                cv2.imshow("video_original_f", front_cam_img)
                cv2.imshow("video_original_r", rear_cam_img)
                # cv2.imshow("video_binary_f", front_prep_img)
                # cv2.imshow("video_binary_r", rear_prep_img)
                
                
                
                
            except Exception as e:
                if self.front_camera_module and self.rear_camera_module:
                    print("Exception occur")
                    self.front_camera_module.close_cam()
                    self.rear_camera_module.close_cam()
                    cv2.destroyAllWindows()
                if self.lidar_module:
                    self.lidar_module.lidar_finish()
                print("Exception error : ", e)
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
                    cv2.destroyAllWindows()
                if self.lidar_module:
                    self.lidar_module.lidar_finish()
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
                    self.lidar_module.lidar_finish()
                end_message = "a0s0o0"
                self.serial.write(end_message.encode())
                self.serial.close()
                self.lidar_module.lidar_finish()
                print("Program Finish")
                
                break
            
            time.sleep((0.00003))
        pass
                
                
        
        
