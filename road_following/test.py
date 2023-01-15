from color_filter.preprocess_pdh import total_function
import os
import cv2
import matplotlib.pyplot as plt
img_path = "/hdd/woonho/autonomous_driving/rfdata/0113/"

img_list = os.listdir(img_path)

for img_idx in range(0, 20000):
    image = cv2.imread(os.path.join(img_path, img_list[img_idx]))
    transform_img = total_function(image)
    cv2.imshow("transform_img", transform_img)



