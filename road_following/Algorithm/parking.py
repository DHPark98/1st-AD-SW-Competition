import numpy as np
import time
import sys

def lidar_condition(min_angle, max_angle, search_distance, scan):
    condition = (((min_angle < scan[:,0]) & (scan[:,0] < max_angle)) &
                 (scan[:,1] < search_distance))
    return condition
def lidar_condition2(min_angle1, max_angle1, min_angle2, max_angle2, search_distance, scan):
    condition = ((((min_angle1 < scan[:,0]) & (scan[:,0] < max_angle1)) | ((min_angle2 < scan[:,0]) & (scan[:,0] < max_angle2))) &
                 (scan[:,1] < search_distance))
    return condition

def detect_parking_car(lidar_module, detect_cnt, new_car_cnt, car_detect_queue, obj):
    try:
        scan = np.array(lidar_module.iter_scans())
        # print(lidar_condition(-100, -90, 2000, scan))
        # print('\n')
        # print((((-100 < scan[:,0]) & (scan[:,0] < -90)) | ((90 < scan[:,0]) & (scan[:,0] < 100)))  & (scan[:,1] < 2000))
        car_search_condition = lidar_condition2(-100, -90, 90, 100, 2000, scan)
        
        if len(np.where(car_search_condition)[0]):
            car_detect_queue = (car_detect_queue * 2 + 1) % 32
        else:
            car_detect_queue = (car_detect_queue * 2) % 32

        if car_detect_queue == 0:
            
            if detect_cnt == 0:
                # if obj == True:
                #     new_car_cnt += 1
                obj = False
                # print('car not detected')
                pass
            else:
                detect_cnt -= 1
        else:
            if detect_cnt == 5:
                # print('car detected')
                if obj == False:
                    new_car_cnt += 1
                obj = True
            else:
                detect_cnt += 1
                
        return new_car_cnt, car_detect_queue
    except Exception as e:
        _, _, tb = sys.exc_info()
        print("process error = {}, error line = {}".format(e, tb.tb_lineno))
        print()
def left_or_right(lidar_module, left_cnt, right_cnt):
    scan = np.array(lidar_module.iter_scans())
    car_left_condition = lidar_condition(90, 100, 2000)
    car_right_condition = lidar_condition(-100, -90, 2000)
    
    left_cnt += len(scan[np.where(car_left_condition)])
    right_cnt += len(scan[np.where(car_right_condition)])

    
    
    
    return left_cnt, right_cnt

def near_detect_car(lidar_module):
    scan = np.array(lidar_module.iter_scans())
    near_detect_condition = lidar_condition(-100, 100, 400, scan)
    
    if len(np.where(near_detect_condition)[0]):
        return True
    else:
        return False
    
def escape(lidar_module):
    scan = np.array(lidar_module.iter_scans())
    escape_condition = lidar_condition(-100, 100, 1200, scan)
    
    if len(np.where(escape_condition)[0]) == 0:
        return True # 주변에 물체가 없으면 True 반환
    else:
        return False

def steering_parking(lidar_module, queue_key, total_array):
    scan = np.array(lidar_module.iter_scans())
    detect_condition = lidar_condition(-100, 100, 2000, scan)
    detect_scan = scan[np.where(detect_condition)]
    steering_angle = parking_steering_angle(detect_scan, queue_key, total_array)
    # print(steering_angle)
    direction = return_parking_direction(-1 * steering_angle)
    
    return direction

def rest(serial, sleep_time):
    message = 'a0s0o0'
    serial.write(message.encode())
    time.sleep(sleep_time)
    return True

def parking_steering_angle(scan, queue_key, total_array):
    delta_threshold = 10
    
    queue_key_arr = (np.ones(len(scan))*queue_key).reshape(-1, 1)
    concat_scan = np.concatenate((queue_key_arr, scan), axis=1)


    try:
        total_array = total_array[np.where(total_array[:,0] != queue_key)]
        total_array = np.concatenate((total_array, concat_scan), axis = 0)
        total_array = total_array[np.where(total_array[:,0] != -1)]

        theta = total_array[:,1]
        theta = np.sort(theta)
        
        theta_1 = np.zeros(theta.shape)
        theta_1[:len(theta)-1] = theta[1:]
        theta_1[len(theta)-1] = theta[0]
        delta_theta = np.abs((theta - theta_1)[1:len(theta)-1]) # delta theta가 너무 작은 경우 threshold로 걸러내는 작업 필요
        
        
        
        
        ret_idx = np.argmax(delta_theta)
        
        if delta_theta[ret_idx] < delta_threshold:
            pass
        
        if len(delta_theta) < 5:
            print("Delta error")
            return 0
        # return
        return (theta[ret_idx+1] + theta[ret_idx+2])/2
    except Exception as e:
        print("steering angle error")
        
        return 0

def good_parking(scan, queue_key, total_array):
    
    queue_key_arr = (np.ones(len(scan))*queue_key).reshape(-1, 1)
    concat_scan = np.concatenate((queue_key_arr, scan), axis=1)

    try:
        total_array = total_array[np.where(total_array[:,0] != queue_key)]
        total_array = np.concatenate((total_array, concat_scan), axis = 0)
        total_array = total_array[np.where(total_array[:,0] != -1)]
        ret_idx = np.argmin(total_array[:,2])

        if len(total_array) < 5:
            print("Too short array error")
            return 0
        # return
        return total_array[ret_idx][1]
    except Exception as e:
        print("good parking error")
        
        return 0

def return_parking_direction(parking_gradient):
    f = lambda x :  7/20 * x
    ret_direction = int(f(parking_gradient))
    
    ret_direction = 7 if ret_direction >= 7 else ret_direction
    ret_direction = -7 if ret_direction <= -7 else ret_direction
    return ret_direction

def calculate_distance(lidar_module):
    total_left_scan = []
    total_right_scan = []
    for i in range(5):
        scan = np.array(lidar_module.iter_scans())

        left_condition = lidar_condition(-100, -80, 1000, scan)
        right_condition = lidar_condition(80, 100, 1000, scan)

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
    
    return left_distance, right_distance

def detailed_parking(lidar_module, queue_key, total_array):
    scan = np.array(lidar_module.iter_scans())
    left_condition = lidar_condition(-100, 80, 1000, scan)
    right_condition = lidar_condition(80, 100, 1000, scan)
    
    left_scan = scan[np.where(left_condition)]
    right_scan = scan[np.where(right_condition)]
    
    left_angle = good_parking(left_scan, queue_key, total_array)
    right_angle = good_parking(right_scan, queue_key, total_array)

    steering_angle = (left_angle + right_angle) / 2

    parking_speed = -1 * parking_speed

    direction = return_parking_direction(steering_angle)

    return direction, total_array

def stop(lidar_module, stop_cnt):
    scan = np.array(lidar_module.iter_scans())
    
    stop_condition = lidar_condition2(-90, -80, 80, 90, 1000, scan)
    
    if len(np.where(stop_condition)[0]) == 0:
        stop_cnt += 1
    else:
        if stop_cnt > 0:
            stop_cnt -= 1
    
    if stop_cnt >= 5:
        return True
    else:
        return False

def search_left_right(lidar_module, cnt):
    scan = np.array(lidar_module.iter_scans())
    left_condition = lidar_condition(-100, 80, 1000, scan)
    right_condition = lidar_condition(80, 100, 1000, scan)
    
    if len(np.where(left_condition)[0]) and len(np.where(right_condition)[0]):
        cnt += 1
    else:
        if cnt > 0:
            cnt -= 1
    
    if cnt >= 5:
        return True
    else:
        return False
    
    
    
    