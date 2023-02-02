import numpy as np
class Parking_constant():
    def __init__(self):
        self.detect_cnt = 0
        self.new_car_cnt = 0
        self.obj = False
        self.car_detect_queue = 0
        self.queue_key = 0
        self.total_array = np.array([[-1, -1, -1]])
        self.stop_cnt = 0
        self.left_right_cnt = 0
