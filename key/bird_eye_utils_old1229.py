### custom_code.py 수정

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

image_width = 640
image_height = 480
direction_div = 12
_shape = np.array([
    [(int(0.05 * image_width), int(0.20 * image_height)), (int(0.495 * image_width), int(0.20 * image_height)), 
    (int(0.495 * image_width), int(0.95 * image_height)), (int(0.05 * image_width), int(0.95 * image_height))], 
    [(int(0.505 * image_width), int(0.20 * image_height)), (int(0.95 * image_width), int(0.20 * image_height)), 
    (int(0.95 * image_width), int(0.95 * image_height)), (int(0.505 * image_width), int(0.95 * image_height))]
    ])

# Calibrate
def calibrate(image_dist):
    with open('calib_param.pkl', 'rb') as f:   
        dic_param = pickle.load(f)
    mtx = dic_param['mtx']
    rvecs = dic_param['rvecs']
    tvecs = dic_param['tvecs']
    
    h, w = image_dist.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,image_dist,(w,h),1,(w,h))
    
    image_undist = cv2.undistort(img, mtx, image_dist, None, newcameramtx)
    return image_undist


# Warpping (Bird Eye View)
def warpping(image, source):
    (h, w) = (image.shape[0], image.shape[1])
    ###source = np.float32([[223, 349], [461, 336], [622, 426], [102, 445]])
    destination = np.float32(
        [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], 
        [round(w * 0.7), h], [round(w * 0.3), h]])
    
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (640,480))

    return _image, minv


# Color Filter (HLS)
def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    # white Filter
    white_lower = np.array([20, 150, 20])   # E: 23, S: 31, L: 170
    white_upper = np.array([255, 255, 255])
    '''
    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])
    
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, white_lower, white_upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # Black Filter
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([40, 40, 35])

    mask = cv2.inRange(image, black_lower, black_upper)
    '''
    mask = cv2.inRange(hls, white_lower, white_upper)
    masked = cv2.bitwise_and(image, image, mask = mask)
    
    return masked


def total_function(img_l, img_r, mat_l, mat_r):
    warpped_l, minverse_l = warpping(img_l, mat_l)
    warpped_r, minverse_r = warpping(img_r, mat_r)
    
    warpped_img = np.concatenate(
        (warpped_l[ :, : round(0.7 * image_width), : ], 
        warpped_r[ :, round(0.3 * image_width): , : ]),
        axis = 1)
    warpped_img = cv2.resize(warpped_img, (image_width, image_height))
    warpped_roi = cv2.polylines(warpped_img, _shape, True, (0, 0, 255))     # 빨간색 박스 2개
    
    w_f_img = color_filter(warpped_img)
    _gray = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('warp', warpped_img)    ###
    cv2.imshow('filter', w_f_img)    ###
    return _gray


if __name__ == '__main__':
    cap = cv2.VideoCapture('drving_video_480.mp4')
    i = 0
    while True:
        retval, img = cap.read()
        img = cv2.resize(img, (image_width, image_height))

        if not retval:
            break

        if i == 0:
            #print(img.shape)
            i = 1

        warpped_img, minverse = warpping(img)
        
        w_f_img = color_filter(warpped_img)
        _gray = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(_gray, 120, 255, cv2.THRESH_BINARY) # thresh : 160
        
        cv2.imshow('thresh', thresh)
        cv2.imshow("video", img)
        cv2.imshow("test1", _gray)
        cv2.imshow("test2", thresh)

        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
