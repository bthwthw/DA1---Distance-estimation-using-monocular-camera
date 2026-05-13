# modules/estimator.py
import numpy as np
import cv2


class DistanceEstimator:
    def __init__(self, focal_length_y, camera_height_m, horizon_y):
        """
        :param focal_length_y: Tiêu cự trục dọc của camera (pixel) - f_y
        :param camera_height_m: Chiều cao lắp đặt camera so với mặt đất (mét) - C_h
        :param horizon_y: Tọa độ pixel trục dọc của đường chân trời - V_horizon - C_y
        """
        self.f_y = focal_length_y
        self.c_h = camera_height_m
        self.v_horizon = horizon_y

    def estimate_ground(self, v_bottom_px):
        """
        Z = (f_y * C_h) / (V_bottom - V_horizon)
        """
        if v_bottom_px <= self.v_horizon: 
            return -1.0 
        
        distance_m = (self.f_y * self.c_h) / (v_bottom_px - self.v_horizon)
        return distance_m
    
    def estimate_geometry(self, bbox_h_px):
        """
        Z = (f * H) / h
        """
        if bbox_h_px <= 0:
            return -1.0
        distance_m = (self.f_y * self.PHYSICAL_CAR_HEIGHT) / bbox_h_px
        return distance_m
    
    def refine_v_bottom(self, frame, x1, y1, x2, y2):
        """
        Dò điểm tâm và hạ xuống tìm viền gầm xe thực tế bằng gradient ngang.
        """
        h_img, w_img = frame.shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1
        center_x = (x1 + x2) // 2

        # 1. Xác định ROI (Region of Interest) ở khu vực đáy box
        # Quét từ 15% phía trên đáy hộp xuống 5% phía dưới đáy hộp 
        # (Vì bóng đổ thường nằm dưới đáy hộp một chút, và mép cản nằm trên)
        scan_y_start = max(0, y2 - int(box_h * 0.15))
        scan_y_end = min(h_img - 1, y2 + int(box_h * 0.05))

        # Lấy một dải hẹp quanh tâm X (Rộng 20% chiều ngang box) để tránh bánh xe hai bên làm nhiễu
        scan_x_start = max(0, center_x - int(box_w * 0.1))
        scan_x_end = min(w_img - 1, center_x + int(box_w * 0.1))

        # Kểm tra ROI hợp lệ
        if scan_y_end <= scan_y_start or scan_x_end <= scan_x_start:
            return center_x, y2 
            
        roi = frame[scan_y_start:scan_y_end, scan_x_start:scan_x_end]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 2. Dùng Sobel tìm đường cắt ngang (mép gầm xe / lốp tiếp xúc đường)
        sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)

        # 3. Tính trung bình cường độ cạnh theo từng hàng ngang
        row_edges = np.mean(abs_sobel_y, axis=1)

        # 4. Tìm vị trí có cạnh ngang mạnh nhất
        # Ngưỡng động: 1.5 lần độ nhiễu trung bình của cả vùng
        threshold = np.mean(row_edges) * 1.5 

        best_y_local = y2 - scan_y_start # Mặc định là y2 ban đầu

        # Dò từ dưới lên để lấy mép ngoài cùng (cắt bóng đổ, chạm vào mặt đường)
        for i in range(len(row_edges)-1, -1, -1):
            if row_edges[i] > threshold:
                best_y_local = i
                break
                
        # Đưa tọa độ từ ROI về lại toàn bức ảnh
        v_bottom_refined = scan_y_start + best_y_local

        return center_x, v_bottom_refined