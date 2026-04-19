import numpy as np

import numpy as np
from collections import deque

class DistanceSmoother:
    def __init__(self, window_size=5, alpha=0.4):
        """
        window_size: Kích thước cửa sổ lọc Trung vị (Nên để số lẻ: 3 hoặc 5)
        alpha: Hệ số làm mượt EMA (Nhỏ = mượt nhiều nhưng trễ, Lớn = bám sát thực tế nhưng hơi rung)
        """
        self.history = {}
        self.window_size = window_size
        self.alpha = alpha

    def update(self, obj_id, new_distance):
        # 1. Khởi tạo cho xe mới xuất hiện
        if obj_id not in self.history:
            self.history[obj_id] = {
                'buffer': deque([new_distance] * self.window_size, maxlen=self.window_size),
                'ema': new_distance
            }
            return new_distance

        # 2. Đẩy khoảng cách mới vào cửa sổ trượt
        self.history[obj_id]['buffer'].append(new_distance)

        # 3. LỌC TRUNG VỊ (MEDIAN): Khử nhiễu nhảy vọt (Spike Killer)
        # Giả sử buffer là [20, 20.5, 45 (lỗi YOLO), 21, 21.5]
        # Median sẽ sắp xếp lại và lấy số ở giữa là 21. Số 45 bị vứt bỏ hoàn toàn!
        median_dist = np.median(self.history[obj_id]['buffer'])

        # 4. LỌC EMA: Làm mượt quỹ đạo
        prev_ema = self.history[obj_id]['ema']
        current_ema = (1 - self.alpha) * prev_ema + self.alpha * median_dist
        
        # Cập nhật lại lịch sử
        self.history[obj_id]['ema'] = current_ema

        return current_ema
    
class BoundingBoxStabilizer:
    def __init__(self):
        self.history_boxes = {} # Lưu tọa độ box của frame liền trước
    
    def calculate_iou(self, boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, obj_id, current_box):
        """
        current_box: [x1, y1, x2, y2] từ YOLO
        """
        current_box = np.array(current_box, dtype=float)
        
        if obj_id not in self.history_boxes:
            self.history_boxes[obj_id] = current_box
            return current_box

        prev_box = self.history_boxes[obj_id]
        
        iou = self.calculate_iou(prev_box, current_box)
        
        # Adaptive Alpha (Hệ số nội suy)
        # iou cao (>0.85) -> Box ít di chuyển -> Chắc chắn là rung nhiễu -> alpha nhỏ (0.2) để làm mượt mạnh
        # iou thấp (<0.6) -> Xe đang di chuyển nhanh -> alpha lớn (0.8) để bám theo xe, tránh bị lag
        if iou > 0.85:
            alpha = 0.2
        elif iou < 0.60:
            alpha = 0.8
        else:
            # Tuyến tính trong khoảng 0.6 -> 0.85
            alpha = 0.8 - ((iou - 0.6) / 0.25) * 0.6 
            
        # 2022_A Spatio-Temporal Robust Tracker with Spatial-Channel Transformer and Jitter Suppression
        smoothed_box = prev_box * (1 - alpha) + current_box * alpha
        
        self.history_boxes[obj_id] = smoothed_box
        
        return smoothed_box