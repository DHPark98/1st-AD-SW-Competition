import os
import cv2
from utility import dominant_gradient

path = '../../data_img/0114/'

file_list = os.listdir(path)


if __name__ == '__main__':
    #print(file_list)

    img = cv2.imread(path + file_list[3])
    edge_img = dominant_gradient(img)
    cv2.imshow('img', img)
    cv2.imshow('edge_img', edge_img)
    cv2.waitKey()

