from Dataset.preprocess_pdh import total_function
import os
import cv2
import matplotlib.pyplot as plt
# img_path = "/hdd/woonho/autonomous_driving/rfdata/0113/"
img_path = "/home/skkcar/Desktop/contest/data_img/0114/"

img_list = os.listdir(img_path)

brk = 0
for img_idx in range(0, 20000):
    if brk == 1:
        break
    
    while True:
        image = cv2.imread(os.path.join(img_path, img_list[img_idx]))
        transform_img = total_function(image)
        cv2.imshow("transform_img", transform_img)

        if cv2.waitKey(1) == ord('f'):
            break
        elif cv2.waitKey(1) == ord('q'):
            brk = 1
            break


