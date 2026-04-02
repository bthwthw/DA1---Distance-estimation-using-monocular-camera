# modules/evaluator.py
import numpy as np

def calculate_iou(boxA, boxB):
    """Tính IoU giữa 2 bounding box [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

class KittiLabelReader:
    def __init__(self, label_path):
        """
        Đọc file label của KITTI và tổ chức lại theo dạng:
        { frame_id: [ {'class': 'Car', 'bbox': [x1,y1,x2,y2], 'z_gt': 25.4}, ... ] }
        """
        self.ground_truth = {}
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_id = int(parts[0])
                obj_class = parts[2]
                
                # Bỏ qua các nhãn không quan tâm
                if obj_class not in ['Car', 'Pedestrian', 'Cyclist']:
                    continue
                
                # Cột 6,7,8,9 là bbox [x1, y1, x2, y2]
                bbox = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
                
                # khoảng cách Z (mét)
                z_gt = float(parts[15])
                
                if frame_id not in self.ground_truth:
                    self.ground_truth[frame_id] = []
                    
                self.ground_truth[frame_id].append({
                    'class': obj_class,
                    'bbox': bbox,
                    'z_gt': z_gt
                })

    def get_gt_for_frame(self, frame_id):
        return self.ground_truth.get(frame_id, [])