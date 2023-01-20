import sys
import os
from pathlib import Path
PATH = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
sys.path.append(PATH)
from utility import is_outside
from Algorithm.Control import moving_log

class avoidance():
    def __init__(self, serial, left_log, right_log):
        self.serial = serial
        self.left_log = left_log
        self.right_log = right_log
        pass
    def action(self, cam_img):
        try:
            if is_outside(cam_img) == True:
                messages = moving_log(self.left_log)
            else:
                messages = moving_log(self.right_log)
                
            for message in messages:
                self.serial.write(message.encode())
                print(message)
            return True
            pass
        except Exception as e:
            print("Exception in avoidance")
            return False
            pass
