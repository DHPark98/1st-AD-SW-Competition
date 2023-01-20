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
print(rf_path)
sys.path.append(rf_path)
>>>>>>> d558718f354929b33466f91784e7854e42cf28f7
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

labels_to_names = {1 : "Green", 2 : "Red", 0 : "Crosswalk"}

weight_file_path = os.path.join(rf_path, 'model_weight_file', 'yolo_weight.pt')
model = DetectMultiBackend(weights = weight_file_path)

img_path = os.path.join(rf_path, 'test_images', 'trafficlight.png')
device = "cuda" if torch.cuda.is_available() else "cpu"
img = cv2.imread(img_path)
draw_img = img.copy()

image = transform.functional.to_tensor(img)
print("shape:", image.shape)
image = image[None, ...]
pred = model(image)

pred = non_max_suppression(pred)[0]
green_color = (0, 255, 0)
red_color = (255, 0, 0)
print(pred)

# image drawing
for *box, cf, cls in pred:
    cf = cf.item()
    cls = cls.item()

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

    bbox_area = (p2[0] - p1[0]) * (p2[1] - p1[1])

    caption = "{}: {:.4f}".format(labels_to_names[cls], cf)
    cv2.rectangle(draw_img, p1, p2, color = green_color, thickness = 2)
    cv2.putText(draw_img, caption, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, thickness = 1)
    print(caption)
    
cv2.imwrite("image01.jpg", draw_img)