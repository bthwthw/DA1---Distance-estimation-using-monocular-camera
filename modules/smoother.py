import numpy as np
import cv2
from collections import deque

class DistanceSmoother:
    def __init__(self, window_size=5, alpha=0.4):
        '''
        window_size: median filter size, odd integer >= 3
        alpha: EMA smoothing factor (0.0 - 1.0)
        '''
        self.history = {}
        self.window_size = window_size
        self.alpha = alpha

    def update(self, obj_id, new_distance):
        if obj_id not in self.history:
            self.history[obj_id] = {
                'buffer': deque([new_distance] * self.window_size, maxlen=self.window_size),
                'ema': new_distance
            }
            return new_distance

        self.history[obj_id]['buffer'].append(new_distance)
        
        median_dist = np.median(self.history[obj_id]['buffer'])
        
        prev_ema = self.history[obj_id]['ema']
        current_ema = (1 - self.alpha) * prev_ema + self.alpha * median_dist
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
    
class OpticalFlowStabilizer:
    def __init__(self):
        self.history = {} # obj_id -> {'box': [x1, y1, x2, y2], 'points': points_array}
        # Tham số cho Lucas-Kanade Optical Flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Tham số tìm góc cạnh sắc nét trên thân xe
        self.feature_params = dict(maxCorners=20, qualityLevel=0.1, minDistance=5, blockSize=5)

    def get_features(self, gray, box):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w//2, y1 + h//2
        
        # THU NHỎ VÙNG TÌM KIẾM (Chỉ lấy 40% ở giữa thân xe)
        # Tránh lấy viền ngoài vì dễ dính background đang trôi (Ego-motion)
        sx1, sy1 = int(cx - w*0.2), int(cy - h*0.2)
        sx2, sy2 = int(cx + w*0.2), int(cy + h*0.2)

        mask = np.zeros_like(gray)
        if sx2 > sx1 and sy2 > sy1:
            mask[sy1:sy2, sx1:sx2] = 255
        else:
            mask[y1:y2, x1:x2] = 255 # Fallback nếu box quá nhỏ

        points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        return points

    def update(self, obj_id, yolo_box, prev_gray, curr_gray):
        yolo_box = np.array(yolo_box, dtype=float)

        # 1. Nếu là xe mới xuất hiện, lưu điểm đặc trưng và trả về YOLO Box
        if obj_id not in self.history or self.history[obj_id]['points'] is None:
            pts = self.get_features(curr_gray, yolo_box)
            self.history[obj_id] = {'box': yolo_box, 'points': pts}
            return yolo_box

        prev_pts = self.history[obj_id]['points']
        prev_box = self.history[obj_id]['box']

        # 2. Tính Optical Flow (Luồng quang học)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **self.lk_params)

        # 3. Lọc ra các điểm theo vết thành công
        if curr_pts is not None and status is not None:
            good_curr = curr_pts[status == 1]
            good_prev = prev_pts[status == 1]
        else:
            good_curr, good_prev = [], []

        # 4. Tính toán Bounding Box mới dựa trên chuyển động của thân xe
        if len(good_curr) >= 3:
            # Tính độ dời (dx, dy) và lấy Trung vị (Median) để khử nhiễu ngoại lai
            dx_dy = good_curr - good_prev
            med_dx = np.median(dx_dy[:, 0])
            med_dy = np.median(dx_dy[:, 1])

            # Box di chuyển theo đúng pixel vật lý của thân xe
            flow_box = prev_box + np.array([med_dx, med_dy, med_dx, med_dy])

            # FUSION (Kết hợp): Flow chống rung cực tốt (80%), YOLO chống trôi (20%)
            final_box = 0.8 * flow_box + 0.2 * yolo_box
        else:
            # Nếu mất dấu quá nhiều điểm, tin tưởng YOLO hoàn toàn
            final_box = yolo_box

        # 5. Cập nhật lại điểm đặc trưng cho frame tiếp theo
        new_pts = self.get_features(curr_gray, final_box)
        self.history[obj_id] = {'box': final_box, 'points': new_pts}

        return final_box