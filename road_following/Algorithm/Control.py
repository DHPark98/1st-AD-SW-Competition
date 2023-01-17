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
    right_threshold = (420, 530) ## threshold 값을 3등분해서 각 구간에 들어가면 weight값에 따라 방향 보정
    left_threshold = (100, 200)
    
    left_idx, right_idx = find_nearest(bottom_value)
    if road_direction < 0:
        if left_idx == None or right_idx == None:
            return -7
        else:
            if right_idx > right_threshold[1]:
                direction = int(road_direction * 0.5)
            elif right_idx > right_threshold[0] and right_idx <= right_threshold[1]:
                direction = int(road_direction * 1.5)
            else:
                direction = int(road_direction * 2.0)

            """차선 근접도를 이용해서 direction 값 조절"""
            
    else:
        if left_idx == None or right_idx == None:
            return 7
        else:
            if left_idx > left_threshold[1]:
                direction = int(road_direction * 0.5)
            elif left_idx > left_threshold[0] and left_idx <= left_threshold[1]:
                direction = int(road_direction * 1.5)
            else:
                direction = int(road_direction * 2.0)
            """차선 근접도를 이용해서 direction 값 조절"""
    
    direction = 7 if direction >= 7 else direction
    direction = -7 if direction <= -7 else direction
    
    return direction

def total_control(road_direction, model_direction, bottom_value):
    road_direction = strengthen_control(road_direction, bottom_value)
    final_direction = control_correction(road_direction, model_direction)
    
    
    return final_direction

def smooth_direction(bef1, bef2, bef3, cur):
    return (bef1 + bef2 + bef3 + cur) / 4

    
        