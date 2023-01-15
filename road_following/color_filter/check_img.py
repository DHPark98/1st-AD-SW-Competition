import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
#from custom_code import *
from preprocess_pdh import *


#imgs = glob.glob("./img/*.png")
imgs = glob.glob("../../data_img/0114/*.png")
for iname in imgs:
    while True:
        img = cv2.imread(iname, 1)
        img_gray = total_function(img)

        cv2.imshow('original', img)

        
        if cv2.waitKey(1) & 0xFF == ord('f'):
            cv2.destroyAllWindows()
            break
