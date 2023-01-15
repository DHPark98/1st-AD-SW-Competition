import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

image_width = 640
image_height = 480
direction_div = 12



def color_filter(image):
    HSV_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(HSV_frame)
    
   
    # green
    H_green_condition = (40<H) & (H<80)
    S_green_condition = (S>30)
    V_green_condition = V>100
    green_condition = H_green_condition & S_green_condition & V_green_condition
    H[green_condition] = 50
    S[green_condition] = 100
    V[green_condition] = 100
    # white
    V_white_condition = V>150
    H[V_white_condition] = 0
    S[V_white_condition] = 0
    V[V_white_condition] = 255
    # black -> blue (road)
    road_condition = True^ (green_condition | V_white_condition)
    H[road_condition] = 120
    S[road_condition] = 150
    V[road_condition] = 150
 
    HSV_frame[:,:,0] = H
    HSV_frame[:,:,1] = S
    HSV_frame[:,:,2] = V
    frame_filtered = cv2.cvtColor(HSV_frame, cv2.COLOR_HSV2BGR)
    
    return frame_filtered

def remove_black(image):
    x = np.linspace(0,639,640)
    y = np.linspace(0,479,480)
    X,Y = np.meshgrid(x,y)

    left_bottom = -123*X + 69*Y - 69*354
    left_bottom = left_bottom>0
    right_bottom = 173*(X-639) + 93*Y - 93*304
    right_bottom = right_bottom>0

    B, G, R = cv2.split(image)
    left_color = image[479, 70]
    right_color = image[479, 545]

    B[left_bottom] = left_color[0]
    G[left_bottom] = left_color[1]
    R[left_bottom] = left_color[2]

    B[right_bottom] = right_color[0]
    G[right_bottom] = right_color[1]
    R[right_bottom] = right_color[2]

    image[:,:,0] = B
    image[:,:,1] = G
    image[:,:,2] = R

    return image

def only_stadium(image):    # 경기장 밖 지우는 함수
    HSV_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(HSV_frame)

    bottom_green_x = -1
    top_green_x = -1

    # 이미지 하단에 초록색 픽셀 있는지 확인
    if(0):
        for x in [544, 540]:
            # print([479, x])   # 왜 S는 100 + 2인지 확인
            H_condition = (30 < H[479, x]) & (H[479, x]<80)     # 조건 1: 해당 픽셀의 Hue가 초록색 범위
            S_condition = S[479, x]==100+2                      # 조건 2: 해당 픽셀의 Saturation이 100임
            V_condition = V[479, x]==100                      # 조건 3: 해당 픽셀의 Value가 100임
            if H_condition and S_condition and V_condition:
                bottom_green_x = x
                break

    # 이미지 상단에 초록색 픽셀 있는지 확인
    
    #HHK CODE-------------------------------------------------------------------------------------
    up_start_time = time.time()
    H_satisfied = (30 < H) & (H<80)
    S_satisfied = S==100+2
    V_satisfied = V==100
    satisfied = H_satisfied & S_satisfied & V_satisfied
    check_top_green = len(np.where(satisfied[0])[0])
    check_top_green
    first_green_x = np.argmax(satisfied, axis = 1).reshape(480, 1)
    
    x = np.linspace(0,639,640)
    y = np.linspace(0,479,480)
    X,Y = np.meshgrid(x,y)
    if check_top_green == 0:    # 제일 상단 row에 초록색 없음
        green_area = (X > first_green_x) & (first_green_x != 0)
    else:    # 제일 상단 row에 초록색 있음
        green_area = (X > first_green_x)
        
    HSV_frame[green_area] = [50, 100, 100]
    
    white_scene = np.ones((480,640))
    #cv2.imshow("satisfied", white_scene)
    #---------------------------------------------------------------------------------------------

    '''
    # 이미지 상단에 초록색 픽셀 있는지 확인
    up_start_time = time.time()
    for x in range(620, 20, -25):
        H_condition = (30 < H[0, x]) & (H[0, x] < 80)     # 조건 1: 해당 픽셀의 Hue가 초록색 범위
        S_condition = S[0, x]==100+2                      # 조건 2: 해당 픽셀의 Saturation이 100임
        V_condition = V[0, x]==100                      # 조건 3: 해당 픽셀의 Value가 100임
        if H_condition and S_condition and V_condition:   
            top_green_x = x
            break
    up_finish_time = time.time()
    print(up_finish_time-up_start_time)
    print(bottom_green_x, top_green_x)

    # 이미지 하단, 상단 모두에 초록색 픽셀이 있는 경우
    if (bottom_green_x != -1) and (top_green_x != -1):
        x = np.linspace(0,639,640)
        y = np.linspace(0,479,480)
        X,Y = np.meshgrid(x,y)

        green_boundary = -479*(X- top_green_x + 2) + (bottom_green_x-top_green_x)*Y
        green_boundary = green_boundary<0

        H[green_boundary] = 50
        S[green_boundary] = 100
        V[green_boundary] = 100
        
        HSV_frame[:,:,0] = H
        HSV_frame[:,:,1] = S
        HSV_frame[:,:,2] = V
    '''
    
    frame_stadium = cv2.cvtColor(HSV_frame, cv2.COLOR_HSV2BGR)
    return frame_stadium

def hide_car_head(image):
    HSV_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(HSV_frame)

    x = np.linspace(0,639,640)
    y = np.linspace(0,479,480)
    X,Y = np.meshgrid(x,y)
    
    a = 132
    b = 50
    elipse_eq = b*b*(X-320)*(X-320) + a*a*(Y-479)*(Y-479) < a*a*b*b
 
    H[elipse_eq] = 120
    S[elipse_eq] = 150
    V[elipse_eq] = 150
    
    HSV_frame[:,:,0] = H
    HSV_frame[:,:,1] = S
    HSV_frame[:,:,2] = V
    car_hidden_img = cv2.cvtColor(HSV_frame, cv2.COLOR_HSV2BGR)
    
    return car_hidden_img


def total_function(image):
    image_blured = cv2.GaussianBlur(image, (0,0), 5)
    image_filtered = color_filter(image_blured)
    image_no_black = remove_black(image_filtered)
    image_stadium = only_stadium(image_no_black)
    car_hidden = hide_car_head(image_stadium)
    image_gray = cv2.cvtColor(car_hidden, cv2.COLOR_BGR2GRAY)
    
    #ret, thresh = cv2.threshold(image_gray, 20, 255, cv2.THRESH_BINARY) # thresh : 160
    
    

    #image_edge = cv2.Canny(image_gray, 110,180)
    #cv2.imshow('edge', image_edge) 


    return car_hidden 


