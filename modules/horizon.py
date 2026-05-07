import cv2
import numpy as np
import random

class HorizonDetector:
    def __init__(self, default_horizon=200, roi_top_pct=0.6, ransac_iters=1000, inlier_thresh=10.0):
        """
        :param ransac_iters: Số vòng lặp lấy mẫu ngẫu nhiên. Càng cao càng chính xác nhưng chậm hơn.
        :param inlier_thresh: Ngưỡng khoảng cách vuông góc (pixel) để coi một đường thẳng là inlier.
        """
        self.default_horizon = default_horizon
        self.roi_top_pct = roi_top_pct
        self.ransac_iters = ransac_iters
        self.inlier_thresh = inlier_thresh

    def detect(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        roi_top = int(h * self.roi_top_pct)
        mask = np.zeros_like(gray)
        mask[roi_top:h, :] = 255
        roi_gray = cv2.bitwise_and(gray, mask)
        
        blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, 
                                minLineLength=40, maxLineGap=20)
        
        if lines is None:
            return None
            
        left_lines = []
        right_lines = []
        all_line_eqs = [] # Lưu hệ số chuẩn hóa (a, b, c) của tất cả các đường
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue 
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Tính a, b, c cho phương trình: (y1-y2)x + (x2-x1)y + (x1*y2 - x2*y1) = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            norm_factor = np.sqrt(a**2 + b**2)
            
            if norm_factor == 0: continue
            
            # Chuẩn hóa để dễ tính khoảng cách vuông góc
            eq = (a / norm_factor, b / norm_factor, c / norm_factor)
            
            if 0.3 < slope < 3.0: 
                right_lines.append(eq)
                all_line_eqs.append(eq)
            elif -3.0 < slope < -0.3:
                left_lines.append(eq)
                all_line_eqs.append(eq)
                
        if not left_lines or not right_lines:
            return None
            
        best_y_int = None
        max_inliers = -1
        
        # --- RANSAC LOOP ---
        for _ in range(self.ransac_iters):
            l1 = random.choice(left_lines)
            l2 = random.choice(right_lines)
            
            # Giải hệ phương trình 2 đường thẳng tìm giao điểm bằng định thức (Cramer)
            D = l1[0] * l2[1] - l2[0] * l1[1]
            if abs(D) < 1e-5: continue # Hai đường gần như song song
            
            x_int = (l1[1] * l2[2] - l2[1] * l1[2]) / D
            y_int = (l2[0] * l1[2] - l1[0] * l2[2]) / D
            
            # Bỏ qua nếu giao điểm nằm ngoài giới hạn không gian hợp lý
            if not (-h <= y_int <= h * 2): 
                continue
            
            # Đếm inliers: Khoảng cách từ giao điểm (x_int, y_int) đến các đường thẳng
            inliers = 0
            for eq in all_line_eqs:
                dist = abs(eq[0] * x_int + eq[1] * y_int + eq[2])
                if dist < self.inlier_thresh:
                    inliers += 1
                    
            if inliers > max_inliers:
                max_inliers = inliers
                best_y_int = y_int
                
        return best_y_int