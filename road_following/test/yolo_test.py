import cv2
import os
import sys
import torch
import numpy as np
import torchvision.transforms as transform
<<<<<<< HEAD:road_following/yolo_test.py
path_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_cur + "/yolov5")
=======
import matplotlib.pyplot as plt
<<<<<<< HEAD

path_rf = os.path.dirname(__file__)
print(path_rf)
sys.path.append(path_rf + "/yolov5")
>>>>>>> 98bd515cde1da8e7bccd4a10c4c2cd44a2580c26:road_following/test/yolo_test.py
=======
from pathlib import Path
rf_path = str(Path(os.getcwd()).parent)
sys.path.append(rf_path)
>>>>>>> d558718f354929b33466f91784e7854e42cf28f7
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from utility import preprocess, show_bounding_box
labels_to_names = {0 : "Crosswalk", 1 : "Green", 2 : "Red"}

weight_file_path = os.path.join(rf_path, 'model_weight_file', 'yolo_weight.pt')
model = DetectMultiBackend(weights = weight_file_path)

img_path = os.path.join(rf_path, 'test_image', 'trafficlight.png')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
img = cv2.imread(img_path)
draw_img = img.copy()

<<<<<<< HEAD
image = transform.functional.to_tensor(img)
print("shape:", image.shape)
image = image[None, ...]
=======
image = preprocess(img, "test")
image = image.cpu()
>>>>>>> e8e61532ae354662fa24e87293a5fdbeeaddf0f2
pred = model(image)
pred = non_max_suppression(pred)[0]

draw_img, detect_list= show_bounding_box(draw_img, pred)

print(detect_list)