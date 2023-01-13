from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.general import scale_coords, non_max_suppression
import cv2
import os
import sys
model = DetectMultiBackend(weights = "yolo_weight.pt")

img_path = "../trafficlight.png"

img = cv2.imread(img_path)

pred = model(img)

print(pred)
