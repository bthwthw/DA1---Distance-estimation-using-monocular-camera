import numpy as np
import time
import csv
import os

class SystemLogger:
    def __init__(self, sequence_name):
        self.sequence_name = sequence_name
        self.start_time = time.time()
        
        # Thống kê khoảng cách
        self.errors = {'near': [], 'mid': [], 'far': []}
        self.mape_list = []
        
        # Thống kê hiệu năng
        self.inference_times = []
        self.total_yolo_boxes = 0
        self.unmatched_boxes = 0
        
        # Thống kê mAP (giả lập dựa trên IoU)
        self.ious = []

    def log_frame(self, inf_time):
        """Ghi lại thời gian xử lý của mỗi frame"""
        self.inference_times.append(inf_time)

    def log_match(self, dist_pred, dist_gt, iou):
        """Ghi lại kết quả khi YOLO khớp với Label"""
        error = abs(dist_pred - dist_gt)
        percent_error = (error / dist_gt) * 100
        
        self.mape_list.append(percent_error)
        self.ious.append(iou)
        self.total_yolo_boxes += 1
        
        if dist_gt <= 20: self.errors['near'].append(error)
        elif dist_gt <= 50: self.errors['mid'].append(error)
        else: self.errors['far'].append(error)

    def log_unmatched(self):
        """Ghi lại khi YOLO bắt sai hoặc Label không có"""
        self.total_yolo_boxes += 1
        self.unmatched_boxes += 1

    def get_summary(self):
        avg_mape = np.mean(self.mape_list) if self.mape_list else 0
        avg_inf = np.mean(self.inference_times) * 1000 # đổi sang ms
        fps = 1.0 / (avg_inf / 1000) if avg_inf > 0 else 0
        mAP_approx = np.mean(self.ious) if self.ious else 0
        
        return {
            'seq': self.sequence_name,
            'mre': round(avg_mape / 100, 4),
            'inf_time': round(avg_inf, 2),
            'fps': round(fps, 1),
            'map': round(mAP_approx, 2),
            'miss_rate': round((self.unmatched_boxes / self.total_yolo_boxes)*100, 2) if self.total_yolo_boxes > 0 else 0
        }

    def save_csv(self, filename="final_results.csv"):
        s = self.get_summary()
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=s.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(s)
        print(f"Đã lưu kết quả set {self.sequence_name} vào {filename}")