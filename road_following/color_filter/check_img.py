import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
#from custom_code import *
from preprocess_pdh import *
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utility import dominant_gradient

#imgs = glob.glob("./img/*.png")
imgs = glob.glob("../../../data_img/0114/*.png")
for inum ,iname in enumerate(imgs):
    while True:
        img = cv2.imread(iname, 1)
        img_stadium = total_function(img)
        img_gradient = dominant_gradient(img_stadium)
        cv2.imshow('grad'+str(inum), img_gradient)
        cv2.imshow('original'+str(inum), img)

        
        if cv2.waitKey(1) & 0xFF == ord('f'):
            cv2.destroyAllWindows()
            break
