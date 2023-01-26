import os
import sys
from pathlib import Path
from Lidar import LidarModule
import numpy as np
from Camera import CameraModule
import cv2
import time
lidar_module = LidarModule()
camera_module = CameraModule(width=640, height=480)
camera_module.open_cam(0)

# for i, scan in enumerate(lidar_module.iter_scans()):
#     print(type(scan))
#     scan = np.array(scan, dtype = np.int16)
#     print(scan.shape)
    
    
#     if i>50:
#         break

i = 1
while True:
    try:
        cam_img = camera_module.read()
        scan = lidar_module.iter_scans()
        scan = np.array(scan)
        lidar_detect_condition = ((-90 < scan[:,0]) & (scan[:,0] < 90)) & (scan[:,1] < 1000)
        # print(scan[np.where(lidar_detect_condition)])
        print(scan[np.where(lidar_detect_condition)])
        cv2.imshow("video", cam_img)
        i+=1
        if i == 1000:
            break
        
        if cv2.waitKey(25) == ord('f'):
            camera_module.close_cam()
            cv2.destroyAllWindows()
            
            lidar_module.scanning_stop()
            lidar_module.stop_motor()
            lidar_module.disconnect()
            print("Program Finish")
            break
    except Exception as e:
        print("Error : {}".format(e))
        break
    
    time.sleep(0.0001)
    
lidar_module.scanning_stop()
lidar_module.stop_motor()
lidar_module.disconnect()
    
    