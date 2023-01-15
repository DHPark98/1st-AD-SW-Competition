
### custom_code.py 수정

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


image_width = 640
image_height = 480
direction_div = 12
_shape = np.array([
    [(int(0.05 * image_width), int(0.20 * image_height)), (int(0.495 * image_width), int(0.20 * image_height)), 
    (int(0.495 * image_width), int(0.95 * image_height)), (int(0.05 * image_width), int(0.95 * image_height))], 
    [(int(0.505 * image_width), int(0.20 * image_height)), (int(0.95 * image_width), int(0.20 * image_height)), 
    (int(0.95 * image_width), int(0.95 * image_height)), (int(0.505 * image_width), int(0.95 * image_height))]
    ])
mat_pts_src = np.float32([[263, 31], [582, 47], [557, 349], [8, 250]])  ###

# Calibrate
def calibrate(image_dist):
    path_calib = os.path.dirname(os.path.abspath(__file__)) + '/calibration'
    with open(path_calib + '/calib_param.pkl', 'rb') as f:   
        dic_param = pickle.load(f)
    mtx = dic_param['mtx']
    
    coeff_dist = dic_param['coeff_dist']
    rvecs = dic_param['rvecs']
    tvecs = dic_param['tvecs']
    
    h, w = image_dist.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,coeff_dist,(w,h),1,(w,h))
    
    image_undist = cv2.undistort(image_dist, mtx, coeff_dist, None, newcameramtx)
    return image_undist

# Source Points of Front / Back Camera
def which_srcmat(FB):
    dic_param = {}
    if FB == 'front':
        path_perspect = os.path.dirname(os.path.abspath(__file__)) + '/find_srcmat'
        with open(path_perspect + '/front_perspect_param.pkl', 'rb') as f:
            # print('perspect.pkl opened!')   
            dic_param = pickle.load(f)
    elif FB == 'back':
        path_perspect = os.path.dirname(os.path.abspath(__file__)) + '/find_srcmat'
        with open(path_perspect + '/back_perspect_param.pkl', 'rb') as f:   
            # print('perspect.pkl opened!')   
            dic_param = pickle.load(f)
            
    if len(dic_param) != 0:
        return dic_param['pts_src']
    else:
        return None
    
    
# Warpping (Bird Eye View)
def warpping(image, pts_src):
    (h, w) = (image.shape[0], image.shape[1])
    ###source = np.float32([[223, 349], [461, 336], [622, 426], [102, 445]])
    destination = np.float32(
        [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], 
        [round(w * 0.7), h], [round(w * 0.3), h]])
    
    transform_matrix = cv2.getPerspectiveTransform(pts_src, destination)
    minv = cv2.getPerspectiveTransform(destination, pts_src)
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


def total_function(img, FB):
    #img_undist = calibrate(img)
    srcmat = which_srcmat(FB)
    img_warpped, minverse = warpping(img, srcmat)
    # warpped_roi = cv2.polylines(img_warpped, _shape, True, (0, 0, 255))    
    
    # img_w_f = color_filter(img_warpped)
    # img_gray = cv2.cvtColor(img_w_f, cv2.COLOR_BGR2GRAY)
    # ret, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

    #cv2.imshow('warp', img_warpped)    ###
    # cv2.imshow('filter', img_w_f)    ###
    # cv2.imshow('gray', img_gray)
    
    ##Canny
    img_gray = cv2.cvtColor(img_warpped, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (0,0),1)
    img_edge = cv2.Canny(img_blur, 110,180)
    
    #cv2.imshow('gray', img_gray)
    #cv2.imshow('blur', img_blur)
    cv2.imshow('edge', img_edge)
    return img_warpped
    #return img_edge


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