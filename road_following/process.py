from Algorithm.img_preprocess import total_function
from Algorithm.Control import total_control, smooth_direction
from utility import roi_cutting, preprocess, show_bounding_box, object_detection, dominant_gradient, cvt_binary, return_road_direction

import Devices.Camera
from Algorithm.BirdEyeConverter import *
from Networks import model
import serial
import time
import torch
import torchvision.transforms as transform
import sys
dir, file = os.path.split(os.path.join(os.path.abspath(__file__)))
sys.path.append(os.path.join(dir, "yolov5"))
from yolov5.utils.general import non_max_suppression
from yolov5.models.common import DetectMultiBackend
import traceback

class DoWork:
    def __init__(self, play_name, front_cam_name, rear_cam_name, rf_weight_file, detect_weight_file=None):
        self.front_camera_module = None
        self.play_type = play_name
        self.cam_num = {"FRONT": 2, "REAR": 4}

        self.front_cam_name = front_cam_name
        self.rear_cam_name = rear_cam_name

        self.rf_weight_file = rf_weight_file
        self.detect_weight_file = detect_weight_file

        self.serial = serial.Serial()
        self.serial.port = '/dev/ttyUSB0'  # 아두이노 메가
        self.serial.baudrate = 9600
        self.speed = 150
        self.direction = 0
        self.rf_network = model.ResNet18(weight_file=self.rf_weight_file)
        self.detect_network = DetectMultiBackend(weights=detect_weight_file)
        self.labels_to_names = {0: "Crosswalk", 1: "Green", 2: "Red", 3: "Car"}

    def serial_start(self):
        try:
            self.serial.open()
            print("Serial open")
            time.sleep(1)
            return True

        except Exception as _:
            print("Serial Fail")
            return False

    def front_camera_start(self):
        try:
            self.front_camera_module = Devices.Camera.CameraModule(
                width=640, height=480)
            self.front_camera_module.open_cam(
                self.cam_num[self.front_cam_name])
            print("FRONT Camera open")
            return True

        except Exception as _:
            print("FRONT Camera Fail")
            return False

    def rear_camera_start(self):
        try:
            self.rear_camera_module = Devices.Camera.CameraModule(
                width=640, height=480)
            self.rear_camera_module.open_cam(self.cam_num[self.rear_cam_name])
            print("REAR Camera open")
            return True

        except Exception as _:
            print("REAR Camera Fail")
            return False

    def Driving(self):
        bef_1d, bef_2d, bef_3d = 0, 0, 0
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

                    order_flag = 1
                    
                    if self.detect_weight_file != None:  # Detection 했을 경우
                        image = preprocess(cam_img, "test")
                        image = image.cpu()
                        pred = self.detect_network(image)
                        
                        pred = non_max_suppression(pred)[0]
                        draw_img = show_bounding_box(draw_img, pred)

                        order_flag = object_detection(pred)

                    road_gradient, bottom_value = dominant_gradient(roi_img)
                    
                    # Gradient가 없을 경우 예외처리(Exception Image)
                    if (road_gradient, bottom_value) == (None, None):
                        self.direction = 0
                        message = 'a' + str(self.direction) + \
                            's' + str(self.speed)#
                        self.serial.write(message.encode())
                        print(message)
                        continue
                    
                    road_direction = return_road_direction(road_gradient)
                    model_direction = torch.argmax(self.rf_network.run(
                        preprocess(roi_img, mode="test"))).item() - 7
                    final_direction = total_control(
                        road_direction, model_direction, bottom_value)

                    if order_flag == 0:
                        print("Stop")
                        self.direction = 0
                        self.speed = 0
                        pass
                    elif order_flag == 1:
                        self.direction = final_direction
                        # self.direction = smooth_direction(
                            # bef_1d, bef_2d, bef_3d, final_direction)
                        pass

                    elif order_flag == 2:
                        print("Road change")

                        pass

                    message = 'a' + str(self.direction) + 's' + str(self.speed)
                    self.serial.write(message.encode())
                    print(message)
                    cv2.imshow('VideoCombined_detect', draw_img)
                    cv2.imshow('VideoCombined_rf', roi_img)
                    cv2.imshow('VideoCombined_rf2', preprocess_img)

                    bef_1d, bef_2d, bef_3d = self.direction, bef_1d, bef_2d
                    pass
            except Exception as e:
                if self.front_camera_module:
                    print(e)
                    print(traceback.format_exc)
                    print("Exception occur")
                    self.front_camera_module.close_cam()
                    end_message = "a0s0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                break
                pass
            except KeyboardInterrupt:
                if self.front_camera_module:
                    print("Keyboard Interrupt occur")
                    self.front_camera_module.close_cam()
                    end_message = "a0s0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                break
                pass

            if cv2.waitKey(25) == ord('f'):
                if self.front_camera_module:
                    self.front_camera_module.close_cam()
                    cv2.destroyAllWindows()
                end_message = "a0s0"
                self.serial.write(end_message.encode())
                self.serial.close()
                print("Program Finish")

                break

            time.sleep((0.0001))

    def Parking(self):
        """
        1. Search Parking location
        2. Ideal Parking Position
        3. Action
        """

        while True:
            try:
                if self.front_camera_module == None or self.rear_camera_module == None:
                    print("Please Check Camera module")
                    break
                    pass
                else:
                    state = 1

                    if state == 1:
                        """
                        직진하며 주차 공간 탐색
                        """
                        if True:
                            """주차 공간 탐색 성공"""
                            parking_state = 2  # 1, 2, 3, 4 중에 하나
                            state = 2
                            pass
                        pass
                    elif state == 2:
                        """
                        Ideal Parking Location으로 이동
                        """
                        if True:
                            """"Ideal Location 이동 성공"""
                            state = 3
                            pass
                        pass
                    elif state == 3:
                        """
                        parking action
                        """
                        if True:
                            """paking finish"""
                            if self.front_camera_module and self.rear_camera_module:
                                self.rear_camera_module.close_cam()
                                self.front_camera_module.close_cam()
                                cv2.destroyAllWindows()
                                end_message = "a0s0"
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
                    end_message = "a0s0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                break
                pass
            except KeyboardInterrupt:
                if self.front_camera_module and self.rear_camera_module:
                    print("Keyboard Interrupt occur")
                    self.front_camera_module.close_cam()
                    self.rear_camera_module.close_cam()
                    end_message = "a0s0"
                    self.serial.write(end_message.encode())
                    self.serial.close()
                break
                pass

            if cv2.waitKey(25) == ord('f'):
                if self.front_camera_module and self.rear_camera_module:
                    self.rear_camera_module.close_cam()
                    self.front_camera_module.close_cam()
                    cv2.destroyAllWindows()
                end_message = "a0s0"
                self.serial.write(end_message.encode())
                self.serial.close()
                print("Program Finish")

                break

            time.sleep((0.0001))
        pass
