from utility import find_nearest, box_area
import os
from pathlib import Path

def control_correction(road_direction, model_direction): # 예측 값과 이미지 기울기 값이 차이가 너무 많이 날 경우 보정
    
    if abs(road_direction - model_direction) <= 2:
        direction = model_direction
    else:
        if road_direction < model_direction:
            direction = road_direction + (abs(road_direction - model_direction) - 2)
        else:
            direction = road_direction - (abs(road_direction - model_direction) - 2)
    
    return direction

def strengthen_control(road_direction, bottom_value): # 차선에 너무 근접한 경우 방향 수정값 증가
    right_threshold = (370, 450, 530) ## threshold 값을 4등분해서 각 구간에 들어가면 weight값에 따라 방향 보정
    left_threshold = (100, 180, 250)
    
    middle_threshold = (260, 280, 300, 340, 360, 380)
    left_idx, right_idx = find_nearest(bottom_value)
    if road_direction < 0:
        if left_idx == None or right_idx == None:
            direction = -7
        else:
            if right_idx > right_threshold[2]:
                direction = int(road_direction * 0.5)
            elif right_idx > right_threshold[1] and right_idx <= right_threshold[2]:
                direction = int(road_direction * 1.0)
            elif right_idx > right_threshold[0] and right_idx <= right_threshold[1]:
                direction = int(road_direction * 1.5)
            else:
                direction = -7

            """차선 근접도를 이용해서 direction 값 조절"""
            
    elif road_direction > 0:
        if left_idx == None or right_idx == None:
            direction = 7
        else:
            if left_idx < left_threshold[0]:
                direction = int(road_direction * 0.5)
            elif left_idx >= left_threshold[0] and left_idx < left_threshold[1]:
                direction = int(road_direction * 1.2)
            elif left_idx >= left_threshold[1] and left_idx < left_threshold[2]:
                direction = int(road_direction * 1.6)    
            else:
                direction = 7
            """차선 근접도를 이용해서 direction 값 조절"""
    else:
        if left_idx == None or right_idx == None:
            if left_idx != None:
                direction = 7
            else:
                direction = -7
        else:
            middle_lane = (left_idx + right_idx)/2
            
            if middle_threshold[0] > middle_lane:
                direction = road_direction - 3
            elif middle_threshold[0] <= middle_lane and middle_lane < middle_threshold[1]:
                direction = road_direction - 2
            elif middle_threshold[1] <= middle_lane and middle_lane < middle_threshold[2]:
                direction = road_direction - 1
            elif middle_threshold[2] <= middle_lane and middle_lane < middle_threshold[3]:
                direction = road_direction + 0
            elif middle_threshold[3] <= middle_lane and middle_lane < middle_threshold[4]:
                direction = road_direction + 1
            elif middle_threshold[4] <= middle_lane and middle_lane < middle_threshold[5]:
                direction = road_direction + 2
            elif middle_threshold[5] <= middle_lane:
                direction = road_direction + 3
            
            
        
    direction = 7 if direction >= 7 else direction
    direction = -7 if direction <= -7 else direction
    
    return direction

def total_control(road_direction, model_direction, bottom_value):
    road_direction = strengthen_control(road_direction, bottom_value)
    final_direction = control_correction(road_direction, model_direction)
    
    return final_direction

def smooth_direction(bef1, bef2, bef3, cur):
    average = bef3 * 0.1 + bef2 * 0.2 + bef1 * 0.3 + cur * 0.4
    return round(average)

def moving_log(log): # road change to inside
    LOG_PATH = os.path.join(str(Path(os.path.dirname(os.path.abspath(__file__))).parent),log)
    
    try:
        FILE = open(LOG_PATH, 'r')
        messages = FILE.readlines()
        messages = list(map(lambda s: s.strip(), messages))
        FILE.close()
        return messages
            
    except Exception as e:
        print("Cannot find {}".format(log))
        return None
    
    
    pass