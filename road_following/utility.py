import os
import torch
from torchvision import transforms
import PIL.Image
import numpy as np
import cv2
import random
import numpy as np
from outdoor_lane_detection import *
import time


def get_resistance_value(file):
    """_summary_

    Args:
        file (_type_): _description_
        ex) ./f_bird--a-1s20--1673179941.70651--b3ccbac7-8f4d-11ed-98fa-c3ae2a3ec2c8.png

    Returns:
        Image's var_resistance value
    """
    dir, filename = os.path.split(file)
    
    if filename.split("--")[1][1] == "-":
        return float(filename.split("--")[1][1:3])
    else:
        return float(filename.split("--")[1][1])

    
def train_test_split(dataset, test_percent = 0.1):
    
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
    
    return train_dataset, test_dataset

def DatasetLoader(dataset, batch_size = 32):
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
    image = image[100:350]
    return image

def preprocess(image, mode, device = "cuda"):
    if mode == "train":
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image.half()
    if mode == "test":
        image = transforms.functional.to_tensor(image).to(device)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = image[None, ...]
        return image.half()

def show_bounding_box(image, pred):
    labels_to_names = {0 : "Crosswalk", 1 : "Green", 2 : "Red", 3 : "Car"}
    
    for *box, cf, cls in pred:
        cf = cf.item()
        cls = cls.item()
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        caption = "{}: {:.4f}".format(labels_to_names[cls], cf)
        cv2.rectangle(image, p1, p2, color = (0, 255, 0), thickness = 2)
        cv2.putText(image, caption, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness = 1)
    
    return image

def object_detection(pred):
    pred_array = [False, False, False, False] # 0:Crosswalk, 1:Green, 2:Red, 3:Car
    bbox_threshold = [0, 0, 0, 0]
    
    for *box, cf, cls in pred:
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        bbox_area = (p2[0] - p1[0]) * (p2[1] - p1[1])
        
        if bbox_area > bbox_threshold[cls] : # find object
            pred_array[cls] = True
    
    if pred_array[0] and pred_array[2]: # stop
        return 0
    
    elif pred_array[0] and pred_array[1]: # go
        return 1
    elif pred_array[3]: # 차선 변경
        return 2
    else:
        return 1

            
            
            
    

def dominant_gradient(image):

    image_original = image.copy()

    ##Canny
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (0,0),1)
    img_edge = cv2.Canny(img_blur, 110,180)

    #cv2.imshow('gray', img_gray)
    #cv2.imshow('blur', img_blur)
    # cv2.imshow('edge', img_edge)
    

    #ppp = True 
    ppp = False
    
    prev = time.time()
    if(not ppp):

        lines = cv2.HoughLines(img_edge,1,np.pi/180,50)
        angles = []
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                print('theta: ', end = '')
                print(theta)
                print('slope: ', end = '')

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0+1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 -1000*(a))
                #if(x1==x2):
                #    print('inf')
                #else:
                #    print((y2-y1)/(x2-x1))
                if(theta < 1.87 and theta > 1.27):
                    continue
                else:
                    angles.append(theta)
                #if ( (x2 != x1) and (np.abs((y2-y1)/(x2-x1))<0.3)):
                #    print(theta)
                #    continue
                cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        print(angles)
        res = np.vstack((image_original,image))
    if(ppp):
        minLineLength = 100
        maxLineGap = 0
        lines = cv2.HoughLinesP(img_edge,1,np.pi/360,100,minLineLength,maxLineGap)
        if((lines) == None):
                res = image.copy()
                return res
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)
                print((x2-x1)/(y2-y1))
                res = image.copy()
    #lines = cv2.HoughLinesP(img_edge, 2, np.pi/180., 50, minLineLength = 40, maxLineGap = 5)
    
    #lane = lane_detect(image)
    
    after = time.time()
    #print(after - prev)
    return res
    #return lane 
    #return img_edge

