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
    
class KalmanFilter2D:
    def __init__(self, dt=0.1, process_noise=0.5, measurement_noise=1.5, initial_pos=0):
        self.dt = dt
        self.x = np.array([[initial_pos], [0.0]])
        self.A = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.P = np.eye(2) * 10.0
        self.Q = np.array([[0.25*(self.dt**4), 0.5*(self.dt**3)],
                           [0.5*(self.dt**3), self.dt**2]]) * process_noise
        self.R = np.array([[measurement_noise]])

        self.MAX_SPEED = 33.3  
        self.MAX_ACCEL = 8.0   
        
        self.last_valid_dist = initial_pos

    def update(self, measurement, box_w=None, box_h=None):
        """
        measurement: Khoảng cách tính từ YOLO (m)
        box_w, box_h: Chiều rộng và cao của Bounding Box để check lỗi mất bánh xe
        """
        # Predict
        x_prior = np.dot(self.A, self.x)
        P_prior = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        is_glitch = False
        # aspect ratio check
        # if box_w is not None and box_h is not None:
        #     ratio = box_w / box_h
        #     if ratio > 2.8: 
        #         is_glitch = True

        # Innovation Gating
        innovation = measurement - np.dot(self.H, x_prior)
        # if innovation big, glitch 
        if abs(innovation) > 10.0:
            is_glitch = True

        if is_glitch:
            # if glitch, use prediction as smoothed value, not update P 
            self.x = x_prior
            self.P = P_prior 
        else:
            S = np.dot(self.H, np.dot(P_prior, self.H.T)) + self.R
            K = np.dot(np.dot(P_prior, self.H.T), np.linalg.inv(S))
            self.x = x_prior + np.dot(K, innovation)
            self.P = np.dot((np.eye(2) - np.dot(K, self.H)), P_prior)

        # speed, acceleration limit check
        self.x[1, 0] = np.clip(self.x[1, 0], -self.MAX_SPEED, self.MAX_SPEED)
        v_diff = self.x[1, 0] - x_prior[1, 0]
        max_v_change = self.MAX_ACCEL * self.dt
        if abs(v_diff) > max_v_change:
            self.x[1, 0] = x_prior[1, 0] + np.sign(v_diff) * max_v_change

        return float(self.x[0, 0])
    
class DistanceSmoother:
    def __init__(self, ema_alpha=0.25, median_window=3):
        """
        :param ema_alpha:     Hệ số EMA (0 < alpha < 1).
        :param median_window: Kích thước cửa sổ median (số lẻ, >= 3).
        """
        assert median_window % 2 == 1 and median_window >= 3, "median_window must be odd >= 3"
        self.alpha = ema_alpha
        self.win = median_window
        self._buf = []       # buffer cho median
        self._ema = None     # giá trị EMA hiện tại

    def update(self, distance: float) -> float:
        # Median 
        self._buf.append(distance)
        if len(self._buf) > self.win:
            self._buf.pop(0)
        median_val = sorted(self._buf)[len(self._buf) // 2]

        # EMA 
        if self._ema is None:
            self._ema = median_val          # khởi tạo lần đầu = giá trị đầu tiên
        else:
            self._ema = self.alpha * median_val + (1 - self.alpha) * self._ema

        return self._ema