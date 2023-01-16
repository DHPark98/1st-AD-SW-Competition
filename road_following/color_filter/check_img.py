import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import sys

path_rf = os.path.dirname(os.path.dirname(__file__))
print(path_rf)
sys.path.append(path_rf)
from Dataset.preprocess_pdh import *
from utility import dominant_gradient

#imgs = glob.glob("./img/*.png")
imgs = glob.glob("../../../data_img/0115/*.png")
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
