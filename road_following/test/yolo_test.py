import cv2
import os
import sys
import torch
import numpy as np
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from pathlib import Path
rf_path = str(Path(os.getcwd()).parent)
sys.path.append(rf_path)
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

image = preprocess(img, "test")
image = image.cpu()
pred = model(image)
pred = non_max_suppression(pred)[0]

draw_img, detect_list= show_bounding_box(draw_img, pred)

print(detect_list)