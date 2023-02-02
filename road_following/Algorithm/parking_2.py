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

def detect_parking_car(lidar_module, c):
    try:
        scan = np.array(lidar_module.iter_scans())
        car_search_condition = lidar_condition(100, 110, 2000, scan)
        
        if detect_counting(car_search_condition, c)[0] == True:
            c.obj = True
        else:
            if c.obj == True:
                c.new_car_cnt += 1
            c.obj = False
            
        return True
            
        
        
        
        
        
    except Exception as e:
        _, _, tb = sys.exc_info()
        print("car detect error = {}, error line = {}".format(e, tb.tb_lineno))
        
# def left_or_right(lidar_module, c):
#     scan = np.array(lidar_module.iter_scans())
#     car_left_condition = lidar_condition(90, 100, 2000, scan)
#     car_right_condition = lidar_condition(-100, -90, 2000, scan)
    
#     c.left_cnt += len(scan[np.where(car_left_condition)])
#     c.right_cnt += len(scan[np.where(car_right_condition)])

#     return c

def near_detect_car(lidar_module):
    scan = np.array(lidar_module.iter_scans())
    near_detect_condition = lidar_condition(-100, 100, 500, scan)
    
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

def steering_parking(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    detect_condition = lidar_condition(-100, 100, 2000, scan)
    detect_scan = scan[np.where(detect_condition)]
    steering_angle, c = parking_steering_angle(detect_scan, c)
    # print(steering_angle)
    direction = return_parking_direction(-1 * steering_angle)
    
    return direction, c

def rest(serial, sleep_time):
    message = 'a0s0o0'
    serial.write(message.encode())
    time.sleep(sleep_time)
    return True

def parking_steering_angle(scan, c, mode = 'angle'):
    delta_threshold = 10
    
    queue_key_arr = (np.ones(len(scan))* c.queue_key).reshape(-1, 1)
    concat_scan = np.concatenate((queue_key_arr, scan), axis=1)

    if mode == 'angle':
        try:
            c.total_array = c.total_array[np.where(c.total_array[:,0] != c.queue_key)]
            c.total_array = np.concatenate((c.c.total_array, concat_scan), axis = 0)
            c.total_array = c.total_array[np.where(c.total_array[:,0] != -1)]

            theta = c.c.total_array[:,1]
            theta = np.sort(theta)
            
            theta_1 = np.zeros(theta.shape)
            theta_1[:len(theta)-1] = theta[1:]
            theta_1[len(theta)-1] = theta[0]
            delta_theta = np.abs((theta - theta_1)[1:len(theta)-1]) # delta theta가 너무 작은 경우 threshold로 걸러내는 작업 필요

            ret_idx = np.argmax(delta_theta)
            
            if delta_theta[ret_idx] < delta_threshold:
                pass
            
            if len(delta_theta) < 3:
                print("Delta error")
                return 0, c
            # return
            print("steering angle : {}".format((theta[ret_idx+1] + theta[ret_idx+2])/2))
            return (theta[ret_idx+1] + theta[ret_idx+2])/2, c
        except Exception as e:
            print("steering angle error")
            
            return 0, c
    if mode == 'distance':
        try:
            c.total_array = c.total_array[np.where(c.total_array[:,0] != c.c.queue_key)]
            c.total_array = np.concatenate((c.total_array, concat_scan), axis = 0)
            c.total_array = c.total_array[np.where(c.total_array[:,0] != -1)]
            ret_idx = np.argmin(c.total_array[:,2])
            if len(c.total_array) < 3:
                print("Too short array error")
                return 0, c
            # return
            return c.total_array[ret_idx][1], c
        except Exception as e:
            print("good parking error")
            
            return None, c

# def good_parking(scan, c):
    
#     queue_key_arr = (np.ones(len(scan))*c.queue_key).reshape(-1, 1)
#     concat_scan = np.concatenate((queue_key_arr, scan), axis=1)

#     try:
#         c.total_array = c.total_array[np.where(c.total_array[:,0] != c.c.queue_key)]
#         c.total_array = np.concatenate((c.total_array, concat_scan), axis = 0)
#         c.total_array = c.total_array[np.where(c.total_array[:,0] != -1)]
#         ret_idx = np.argmin(c.total_array[:,2])
#         if len(c.total_array) < 3:
#             print("Too short array error")
#             return 0, c
#         # return
#         return c.total_array[ret_idx][1], c
#     except Exception as e:
#         print("good parking error")
        
#         return None, c

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

def detailed_parking(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    left_condition = lidar_condition(-100, -80, 1000, scan)
    right_condition = lidar_condition(80, 100, 1000, scan)
    
    left_scan = scan[np.where(left_condition)]
    right_scan = scan[np.where(right_condition)]

    
    left_angle, c = parking_steering_angle(left_scan, c, 'distance')
    right_angle, c = parking_steering_angle(right_scan, c, 'distance')
    print(left_angle, right_angle)
    try: 
        steering_angle = (left_angle + right_angle) / 2
        print("steering angle : {}".format(steering_angle))
        direction = return_parking_direction(steering_angle)    
    except Exception as e:
        
        if left_angle == None:
            direction = 7
        elif right_angle == None:
            direction = -7
        else:
            direction = 0

    return direction

def stop(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    
    stop_condition = lidar_condition2(-90, -80, 80, 90, 1000, scan)
    
    return detect_counting(stop_condition, c)

def search_left_right(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    left_condition = lidar_condition(-100, -70, 1000, scan)
    right_condition = lidar_condition(70, 100, 1000, scan)
    
    return detect_counting(condition1=left_condition, c = c, condition2=right_condition)
    
    
def escape_parking(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    left_condition = lidar_condition(-80, -70, 1000, scan)
    right_condition = lidar_condition(70, 80, 1000, scan)

    return detect_counting(condition1=left_condition, c = c, condition2=right_condition)

def escape_parking2(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    condition = lidar_condition(-90, 90, 1200, scan)
    
    return detect_counting(condition, c)

def escape_parking3(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())

    rear_condition = lidar_condition(-10, 10, 2000, scan)
    return detect_counting(rear_condition, c)
    
    
def detect_counting(condition1, c, condition2 = None): # 3번 연속 detect 했을때 True return
    if condition1 != None and condition2 != None:
        if len(np.where(condition1)[0]) and len(np.where(condition2)[0]):
            if c.detect_cnt < 3:
                c.detect_cnt += 1
        else:
            if c.detect_cnt > 0:
                c.detect_cnt -= 1
            
        if c.detect_cnt == 3:
            return True, c
        else:
            return False, c
    else:
        if len(np.where(condition1)[0]):
            if c.detect_cnt < 3:
                c.detect_cnt += 1
        else:
            if c.detect_cnt > 0:
                c.detect_cnt -= 1
            
        if c.detect_cnt == 3:
            return True, c
        else:
            return False, c
        