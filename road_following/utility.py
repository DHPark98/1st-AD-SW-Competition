import os
import torch
from torchvision import transforms
import PIL.Image
import numpy as np
import cv2
import random
import numpy as np
from Algorithm.outdoor_lane_detection import *
import time
from Algorithm.img_preprocess import cvt_binary, total_function
import matplotlib.pyplot as plt
import uuid
import sys
from datetime import datetime
from Algorithm.BirdEyeConverter import bird_convert

def get_resistance_value(file):
    dir, filename = os.path.split(file)
    
    if filename.split("--")[1][1] == "-":
        return float(filename.split("--")[1][1:3])
    else:
        return float(filename.split("--")[1][1])

    
def train_test_split(dataset, test_percent = 0.1):
    
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
    
    return train_dataset, test_dataset

def DatasetLoader(dataset, batch_size = 128):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return data_loader
    
    
def return_augmented_images(image, style):
    """_summary_
    Args:
        image (_type_): numpy array
        style (_type_): noise, brightness, saturation

    Returns:
        augmented_image
    """
    
    if style == "noise":
        img = np.array(image)/255.0
        row,col,ch= img.shape
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        
        noisy_img = (img + gauss) * 255
        return noisy_img
    
    elif style == "brightness":
        aug_f = transforms.ColorJitter(brightness=(0.2, 2))
        augmented_image = aug_f(image)
        return np.array(augmented_image)
    
    elif style == "saturation":
        aug_f = transforms.ColorJitter(saturation=(0.2, 0.21))
        augmented_image = aug_f(image)
        return np.array(augmented_image)
    
    
def roi_cutting(image):
    image = image[200:]
    return image
    # x = np.linspace(0,639,640)
    # y = np.linspace(0,479,480)
    # X,Y = np.meshgrid(x,y)    
    # equation = 200*X - 640 * Y + 640*100 > 0
    # image[equation] = 0

    return image

                                                                                                             

def preprocess(image, mode, device = "cuda"):
    
    if mode == "train":
        image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image
    if mode == "test":
        image = transforms.functional.to_tensor(image).to(device)
        # image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = image[None, ...]
        return image


def box_center(box):
    if box == None:
        return None
    
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    return (int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2))

def box_area(box):
    if box == None:
        return 0
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    box_area = (p2[0] - p1[0]) * (p2[1] - p1[1])
    return box_area

def center_inside(center):
    x = center[0]
    y = center[1]
    
    if x > 550 or x < 90 or y < 270:
        return False
    else:
        return True

def show_bounding_box(image, pred):
    labels_to_names = {0 : "Crosswalk", 1 : "Green", 2 : "Red", 3 : "Car"}
    
    
    for *box, cf, cls in pred:
        cf = cf.item()
        cls = int(cls.item())
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        caption = "{}: {:.4f}".format(labels_to_names[cls], cf)
        cv2.rectangle(image, p1, p2, color = (0, 255, 0), thickness = 2)
        cv2.putText(image, caption, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness = 1)
        
    
    return image

def object_detection(pred): # pred 중 class별로 가장 큰 bbox return
    pred_array = [None, None, None, None] # 0:Crosswalk, 1:Green, 2:Red, 3:Car
    bbox_threshold = [30000, 10000, 10000, 20000] # bbox area
    
    for *box, cf, cls in pred:
        bbox_area = box_area(box)
        cls = int(cls)
        if bbox_area < bbox_threshold[cls] and cls != 3 : # find object
            if pred_array[cls] != None and box_area(pred_array[cls]) > bbox_area: 
                pass
            else:
                pred_array[cls] = box
                
        elif (cls == 3 and center_inside(box_center(box)) and
                box_area(box) > bbox_threshold[cls]): # find object(car)
            pred_array[cls] = box
                
    if pred_array[0] != None and pred_array[2] != None:
        order_flag = 0
    elif pred_array[3] != None:
        order_flag = 2
    else:
        order_flag = 1
    return pred_array, order_flag

            
            
            
    

def dominant_gradient(image, pre_image): # 흑백 이미지에서 gradient 값, 차선 하단 값 추출

    image_original = image.copy()

    ##Canny
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        img_blur = cv2.GaussianBlur(image_original, (0,0),1)
        img_edge = cv2.Canny(img_blur, 110,180)
    except Exception as e:
        _, _, tb = sys.exc_info()
        print("image preprocess(gradient) error = {}, error line = {}".format(e, tb.tb_lineno))
        
        exception_image_path = "./exception_image/"
        
        try:
            if not os.path.exists(exception_image_path):
                os.mkdir(exception_image_path)    
        except OSError:
            print('Error: Creating dirctory. ' + exception_image_path)
        
        cv2.imwrite(os.path.join(exception_image_path, "exception_image--{}.png".format(datetime.now())), pre_image)
        return None, None
        
    try:
        lines = cv2.HoughLines(img_edge,1,np.pi/180,30)

        angles = []
        bottom_flag = np.zeros((640,))
        bottom_idx = 280
        
        if(not isinstance(lines, type(None))):
            
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0+1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 -1000*(a))
                    
                    if y1 > 120 or y2 > 120:
                        flag_idx = int((x1-x2)/(y1-y2) * (bottom_idx - 1 - y1) + x1)
                        if flag_idx < 0 or flag_idx >= 640:
                            continue
                        bottom_flag[flag_idx] = 1

                    
                    
                    
                    if(theta < 1.87 and theta > 1.27):
                        continue
                    else:
                        if y1 == y2:
                            angle = 'inf'
                        else:
                            angle = np.arctan((x2-x1)/(y1-y2))*180/np.pi
                        angles.append(angle)
        result_idx = np.where(bottom_flag == 1)[0]
        if len(angles) == 0:
            result = 0
        else:
            result = np.median(angles)

        #print(angles)
        return result, result_idx               
        
    except Exception as e:
        _, _, tb = sys.exc_info()
        print("gradient detection error = {}, error line = {}".format(e, tb.tb_lineno))
        exception_image_path = "./exception_image/"
        try:
            if not os.path.exists(exception_image_path):
                os.mkdir(exception_image_path)    
        except OSError:
            print('Error: Creating dirctory. ' + exception_image_path)
        cv2.imwrite(os.path.join(exception_image_path, "exception_image--{}.png".format(str(uuid.uuid1()))), pre_image)
        return None, None
    


def return_road_direction(road_gradient):
    f = lambda x : 7/64000*x**3
    ret_direction = int(f(road_gradient))
    
    ret_direction = 7 if ret_direction >= 7 else ret_direction
    ret_direction = -7 if ret_direction <= -7 else ret_direction
    
    return ret_direction
        

def find_nearest(array, value=315):
    array = np.asarray(array)
    left_val = array[np.max(np.where(array <= value)[0])] if len(np.where(array <= value)[0]) != 0 else None
    right_val = array[np.min(np.where(array > value)[0])] if len(np.where(array > value)[0]) != 0 else None
    
    return left_val, right_val

def is_outside(image): # Is current line outside?
    HSV_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(HSV_frame)

    bottom_green_x = -1
    top_green_x = -1
    up_start_time = time.time()

    H_satisfied = (30 < H) & (H<80)
    S_satisfied = S==100+2
    V_satisfied = V==100
    satisfied = H_satisfied & S_satisfied & V_satisfied
    satisfied[:,639] = True
    check_top_green = len(np.where(satisfied[0])[0])
    first_green_x = np.argmax(satisfied, axis = 1).reshape(480, 1)
    if np.percentile(first_green_x,5) == 639:
        return 0

    else:
        return 1
    #pass

def front_line_detect(image):
    image_original = image.copy()
    try:
        img_blur = cv2.GaussianBlur(image_original, (0,0),1)
        img_edge = cv2.Canny(img_blur, 110,180)
        lines = cv2.HoughLines(img_edge,1,np.pi/180,30)

        angles = []
        
        if(not isinstance(lines, type(None))):
            
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0+1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 -1000*(a))
                if y1 == y2:
                    angle = 'inf'
                else:
                    angle = np.arctan((x2-x1)/(y1-y2))*180/np.pi
                if -45 < angle and angle < 45:
                    pass
                else:
                    angles.append(angle)
        
        return np.median(angles)
                        

    except Exception as e:
        _, _, tb = sys.exc_info()
        print("front line detection error = {}, error line = {}".format(e, tb.tb_lineno))
        return None


def total_process(image, mode = "FRONT"):
    # original image => binary bev image
    
    bev = bird_convert(image, mode)
    prep_img = total_function(bev)
    binary_img = cvt_binary(prep_img)
    
    return binary_img
    
    
    
    