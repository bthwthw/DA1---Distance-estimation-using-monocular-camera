# modules/kalman.py
import numpy as np

class KalmanFilter1D:
    def __init__(self, process_variance, measurement_variance, initial_state=0):
        """
        :param process_variance (Q): Độ nhiễu của mô hình vật lý (xe di chuyển mượt hay giật).
        :param measurement_variance (R): Độ nhiễu của phép đo (YOLO bắt box rung cỡ nào).
        :param initial_state: Giá trị khởi tạo ban đầu (ví dụ: v_bottom đầu tiên).
        """
        self.Q = process_variance
        self.R = measurement_variance
        
        self.x = initial_state # Giá trị được làm mượt
        self.P = 1.0           # Sai số ước tính ban đầu

    def update(self, measurement):
        """
        :param measurement: Giá trị 'thô' đo được từ cảm biến (v_bottom của YOLO).
        :return: Giá trị đã được làm mượt (x).
        """
        
        self.x = self.x # Trạng thái dự đoán vẫn là trạng thái cũ
        
        # mô hình không hoàn hảo
        self.P = self.P + self.Q
        
        # Tính Kalman Gain K
        # K tiến về 0: Tin mô hình hơn.
        # K tiến về 1: Tin YOLO hơn.
        S = self.P + self.R
        K = self.P / S
        
        self.x = self.x + K * (measurement - self.x)
        
        self.P = (1 - K) * self.P
        
        return self.x