from utility import find_nearest 
import numpy as np

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
    right_threshold = (370, 420, 530) ## threshold 값을 4등분해서 각 구간에 들어가면 weight값에 따라 방향 보정
    left_threshold = (100, 200, 250)
    
    middle_threshold = (100, 200, 300, 340, 440, 540)
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
                direction = int(road_direction * 1.0)
            elif left_idx >= left_threshold[1] and left_idx < left_threshold[2]:
                direction = int(road_direction * 1.5)    
            else:
                direction = 7
            """차선 근접도를 이용해서 direction 값 조절"""
    else:
        if left_idx == None or right_idx == None:
            pass
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
    return (bef1 + bef2 + bef3 + cur) / 4

    
        