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
        # print(scan)
        car_search_condition = lidar_condition(-110, -100, 1500, scan)
        print(scan[car_search_condition])
        c = detect_counting(car_search_condition, c)
        print(c.detect_cnt)
        if c.flag == True:
            print("Detect!")
            if c.obj == False:
                c.new_car_cnt += 1
            c.obj = True
        else:
            print("not detect!")
            c.obj = False

        return c
        
    except Exception as e:
        _, _, tb = sys.exc_info()
        print("car detect error = {}, error line = {}".format(e, tb.tb_lineno))

def escape_stage1(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    near_detect_condition = lidar_condition(-100, 100, 1200, scan)
    
    detect_counting(near_detect_condition, c)
    
    if len(np.where(near_detect_condition)[0]) == 0:
        return True
    else:
        return False
    
def escape_stage2(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    left_detect_condition = lidar_condition(80, 100, 1000, scan)
    right_detect_condition = lidar_condition(-100, -80, 1000, scan)
    
    detect_counting2(left_detect_condition, right_detect_condition, c)
    
    return c
    
def stay_with_lidar(lidar_module, serial, speed, direction, rest_time = 1):
    start_time = time.time()
    end_time = time.time()
    while(end_time - start_time >= rest_time):
        scan = np.array(lidar_module.iter_scans())
        message = "a" + str(direction) + "s" + str(speed) + "o0"
        serial.write(message.encode())
        end_time = time.time()
    

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
    detect_condition = lidar_condition(-100, 100, 3000, scan)
    detect_scan = scan[np.where(detect_condition)]
    print(detect_scan)
    steering_angle, distance_bias, c = parking_steering_angle(detect_scan, c)
    # print(steering_angle)
    direction = return_parking_direction(-1 * steering_angle) + int(distance_bias * 4/700)
    
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
            c.total_array = np.concatenate((c.total_array, concat_scan), axis = 0)
            c.total_array = c.total_array[np.where(c.total_array[:,0] != -1)]

        #     min_idx = np.argmin(c.total_array[:,2])
        #     max_idx = np.argmax(c.total_array[:,2])

        #     min_angle, min_dist = c.total_array[min_idx][1], c.total_array[min_idx][2]
        #     max_angle, max_dist = c.total_array[max_idx][1], c.total_array[max_idx][2]

        #     print("min angle : {}, min dist : {}".format(min_angle, min_dist))
        #     print("max angle : {}, max dist : {}".format(max_angle, max_dist))

        #     return max_angle, c

        # except Exception as e:
        #     return 0, c
            theta = c.total_array[:,1]
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
            distance_bias = c.total_array[ret_idx+1, 2] - c.total_array[ret_idx+2, 2]
            print("distance_bias : ", distance_bias)
            return (theta[ret_idx+1] + theta[ret_idx+2])/2, distance_bias,  c
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("parking steering angle error = {}, error line = {}".format(e, tb.tb_lineno))
            
            return 0, 0, c
    if mode == 'distance':
        try:
            c.total_array = c.total_array[np.where(c.total_array[:,0] != c.queue_key)]
            c.total_array = np.concatenate((c.total_array, concat_scan), axis = 0)
            c.total_array = c.total_array[np.where(c.total_array[:,0] != -1)]
            ret_idx = np.argmin(c.total_array[:,2])


            if len(c.total_array) < 3:
                print("Too short array error")
                return 0, c
            # return
            return c.total_array[ret_idx][1], c
        except Exception as e:
            _, _, tb = sys.exc_info()
            print("parking steering distance error = {}, error line = {}".format(e, tb.tb_lineno))
            
            return None, c



def return_parking_direction(parking_gradient):
    f = lambda x :  7/20 * x
    ret_direction = int(f(parking_gradient))
    
    ret_direction = 7 if ret_direction >= 7 else ret_direction
    ret_direction = -7 if ret_direction <= -7 else ret_direction
    return ret_direction

def calculate_distance(lidar_module):

    try:
        total_left_scan = []
        total_right_scan = []
        for i in range(5):
            scan = np.array(lidar_module.iter_scans())

            right_condition = lidar_condition(-100, -70, 1000, scan)
            left_condition = lidar_condition(70, 100, 1000, scan)

            left_scan = scan[np.where(left_condition)]
            right_scan = scan[np.where(right_condition)]
            
            if i == 0:
                total_left_scan = left_scan
                total_right_scan = right_scan
            else:
                total_left_scan = np.concatenate((total_left_scan, left_scan), axis = 0)
                total_right_scan = np.concatenate((total_right_scan, right_scan), axis = 0)
                
        print("total left scan : ", total_left_scan)
        print("total right scan : ", total_right_scan)     
        left_distance = np.min(total_left_scan[:,1])
        right_distance = np.min(total_right_scan[:,1])

        return left_distance, right_distance
    except Exception as e:
        _, _, tb = sys.exc_info()
        print("calculate distance error = {}, error line = {}".format(e, tb.tb_lineno))


def detailed_parking(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    right_condition = lidar_condition(-100, -30, 1000, scan)
    left_condition = lidar_condition(30, 100, 1000, scan)
    
    left_scan = scan[np.where(left_condition)]
    right_scan = scan[np.where(right_condition)]

    
    left_angle, c = parking_steering_angle(left_scan, c, 'distance')
    right_angle, c = parking_steering_angle(right_scan, c, 'distance')

    
    try: 
        if 80 <= left_angle and left_angle <= 100:
            direction = 0
        else:
            direction = int((80 - left_angle) * 7/30)
        
        if -100 <= right_angle and right_angle <= -80:
            direction = 0
        else:
            direction = int((-80 - right_angle) * 7/30)
    except Exception as e:
        _, _, tb = sys.exc_info()
        print("detailed parking error = {}, error line = {}".format(e, tb.tb_lineno))
        direction = 0
        c.stop = True

    return direction, c

def stop(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    
    stop_condition = lidar_condition2(-90, -80, 80, 90, 1000, scan)
    
    return detect_counting(stop_condition, c)

def search_left_right(lidar_module, c):
    scan = np.array(lidar_module.iter_scans())
    left_condition = lidar_condition(-100, -80, 1000, scan)
    right_condition = lidar_condition(80, 100, 1000, scan)
    
    return detect_counting2(left_condition, right_condition, c)
    
    
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
    
    
def detect_counting(condition1, c): # 3번 연속 detect 했을때 True return
    
    if len(np.where(condition1)[0]):
        if c.detect_cnt < 3:
            c.detect_cnt += 1
    else:
        if c.detect_cnt > 0:
            c.detect_cnt -= 1
        
    if c.detect_cnt == 3:
        c.flag = True
        return c
    elif c.detect_cnt == 0:
        c.flag = False
        return c
    else:
        return c

def detect_counting2(condition1, condition2, c): # 3번 연속 detect 했을때 True return
    
    if len(np.where(condition1)[0]) and len(np.where(condition2)[0]):
        if c.detect_cnt < 3:
            c.detect_cnt += 1
    else:
        if c.detect_cnt > 0:
            c.detect_cnt -= 1
        
    if c.detect_cnt == 3:
        c.flag = True
        return c
    elif c.detect_cnt == 0:
        c.flag = False
        return c
    else:
        return c