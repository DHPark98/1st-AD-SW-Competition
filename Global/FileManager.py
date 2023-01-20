#!/usr/bin/env python
import os
from datetime import datetime
import cv2
import sys
import numpy as np
import shutil
import time

IMAGELOGPATH = os.path.expanduser('~') + "/ImageLog/"
TOPICLOGPATH = os.path.expanduser('~') + "/TopicLog/"
IMAGE_SAVE = True
def delete_files(vision_type, time_elapsed, time_type, log_type):
    """
    :param directory_path:
    :param vision_type:
    :param time_elapsed:
    :param time_type: 0 --- seconds
                      1 --- minutes
                      2 --- hours
                      3 --- days
    :return: x
    """
    offset_time = np.nan
    if time_type == 0:
        offset_time = (time_elapsed)
    elif time_type == 1:
        offset_time = (time_elapsed * 60)
    elif time_type == 2:
        offset_time = (time_elapsed * 60 * 60)
    elif time_type == 3:
        offset_time = (time_elapsed * 60 * 60 * 24)

    log_path = None
    if log_type == 0:
        log_path = IMAGELOGPATH
    elif log_type == 1:
        log_path = TOPICLOGPATH

    delete_path = log_path + vision_type

    if os.path.exists(delete_path):
        for f in os.listdir(delete_path):
            f = os.path.join(delete_path, f)
            if os.path.exists(f):
                timestamp_now = time.time()
                is_old = os.stat(f).st_mtime < timestamp_now - offset_time
                if is_old:
                    try:
                        if os.path.isfile(f):
                            os.remove(f)
                        if os.path.isdir(f):
                            shutil.rmtree(f)
                        # print(f, 'is deleted')
                    except OSError as e:
                        print("error : {}".format(e))
                        print(f, 'can not delete')

def save_vision_image(vision_type, image_name, image, now_time):
    """
    :param path_target: save dir
    :param image: save img
    :return:
    """
    try:
        save_path = IMAGELOGPATH + vision_type
        if not (os.path.isdir(save_path)):
            os.makedirs(os.path.join(save_path))
            pass
        image_path = save_path + "/" + now_time + "_" + image_name + ".jpg"
        if IMAGE_SAVE == True:
            resize_img = cv2.resize(image,
                                         dsize=(int(image.shape[1] / 2), int(image.shape[0] / 2)),
                                         interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_path, np.uint8(resize_img))

    except Exception as e:
        _, _, tb = sys.exc_info()
        print('{0} image save error = {1}, error line = {2}'.format(__file__, e, tb.tb_lineno))

def save_topic_csv(topic_data, vision_type, date):
    try:
        csv_folder = TOPICLOGPATH + vision_type
        csv_name = date + ".csv"
        csv_path = csv_folder + "/" + csv_name

        if not (os.path.isdir(csv_folder)):
            os.makedirs(os.path.join(csv_folder))
            pass

        if not os.path.exists(csv_path):
            topic_data.to_csv(csv_path, index=False, mode='w', encoding='utf-8-sig')
        else:
            topic_data.to_csv(csv_path, index=False, mode='a', encoding='utf-8-sig', header=False)
            pass
        pass
        return True

    except Exception as e:
        _, _, tb = sys.exc_info()
        print('{0} image save error = {1}, error line = {2}'.format(__file__, e, tb.tb_lineno))
        return False