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
        
        # --- THÊM BIẾN LƯU TRỮ TỈ LỆ BOX ---
        self.last_valid_dist = initial_pos

    def update(self, measurement, box_w=None, box_h=None):
        """
        measurement: Khoảng cách tính từ YOLO (m)
        box_w, box_h: Chiều rộng và cao của Bounding Box để check lỗi mất bánh xe
        """
        # 1. Dự đoán (Predict)
        x_prior = np.dot(self.A, self.x)
        P_prior = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        # 2. KIỂM TRA LỖI HÌNH HỌC (Aspect Ratio Check)
        is_glitch = False
        if box_w is not None and box_h is not None:
            ratio = box_w / box_h
            # Xe con thường ratio ~1.5-2.0. Nếu ratio > 2.8 là mất bánh xe chắc chắn
            if ratio > 2.8: 
                is_glitch = True

        # 3. KIỂM TRA LỖI VẬT LÝ (Innovation Gating)
        innovation = measurement - np.dot(self.H, x_prior)
        # Nếu khoảng cách nhảy vọt > 5m trong 0.1s (phi thực tế)
        if abs(innovation) > 5.0:
            is_glitch = True

        # --- CHIẾN THUẬT CỨU VÃN ---
        if is_glitch:
            # Nếu lỗi, ta không cập nhật YOLO, chỉ lấy dự đoán x_prior
            self.x = x_prior
            self.P = P_prior 
        else:
            # Nếu ổn, chạy Update Kalman bình thường
            S = np.dot(self.H, np.dot(P_prior, self.H.T)) + self.R
            K = np.dot(np.dot(P_prior, self.H.T), np.linalg.inv(S))
            self.x = x_prior + np.dot(K, innovation)
            self.P = np.dot((np.eye(2) - np.dot(K, self.H)), P_prior)

        # 4. Ràng buộc vận tốc & gia tốc
        self.x[1, 0] = np.clip(self.x[1, 0], -self.MAX_SPEED, self.MAX_SPEED)
        v_diff = self.x[1, 0] - x_prior[1, 0]
        max_v_change = self.MAX_ACCEL * self.dt
        if abs(v_diff) > max_v_change:
            self.x[1, 0] = x_prior[1, 0] + np.sign(v_diff) * max_v_change

        return float(self.x[0, 0])