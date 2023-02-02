import numpy as np
class Parking_constant():
    def __init__(self):
        self.detect_cnt = 0
        self.new_car_cnt = 0
        self.obj = False
        self.queue_key = 0
        self.total_array = np.array([[-1, -1, -1]])
        
    def initialize(self):
        self.detect_cnt = 0
        self.new_car_cnt = 0
        self.obj = False
        self.queue_key = 0
        self.total_array = np.array([[-1, -1, -1]])