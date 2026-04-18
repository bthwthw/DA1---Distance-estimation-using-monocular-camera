import numpy as np
import time
import csv
import os

class SystemLogger:
    def __init__(self, sequence_name, log_dir="logs"):
        self.sequence_name = sequence_name
        self.log_dir = log_dir
        self.start_time = time.time()
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.detail_file = os.path.join(self.log_dir, f"{sequence_name}_details_2.csv")
        self._init_detail_file()

        # Thống kê tổng hợp
        self.mape_list = []
        self.inference_times = []
        self.total_yolo_boxes = 0
        self.unmatched_boxes = 0
        self.ious = []

    def _init_detail_file(self):
        """Add header"""
        with open(self.detail_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'obj_id', 'dist_gt', 'dist_pred', 'error_pct', 'iou'])

    def log_frame(self, inf_time):
        self.inference_times.append(inf_time)

    def log_match(self, frame_idx, obj_id, dist_pred, dist_gt, iou):
        """Log matched object details"""
        error = abs(dist_pred - dist_gt)
        percent_error = (error / dist_gt) * 100
        
        self.mape_list.append(percent_error)
        self.ious.append(iou)
        self.total_yolo_boxes += 1
        
        with open(self.detail_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([frame_idx, obj_id, round(dist_gt, 3), round(dist_pred, 3), round(percent_error, 2), round(iou, 2)])

    def log_unmatched(self):
        self.total_yolo_boxes += 1
        self.unmatched_boxes += 1

    def get_summary(self):
        avg_mape = np.mean(self.mape_list) if self.mape_list else 0
        avg_inf = np.mean(self.inference_times) 
        if avg_inf < 1.0: avg_inf *= 1000
            
        fps = 1000.0 / avg_inf if avg_inf > 0 else 0
        mAP_approx = np.mean(self.ious) if self.ious else 0
        
        return {
            'seq': self.sequence_name,
            'mre': round(avg_mape / 100, 4),
            'inf_time': round(avg_inf, 2),
            'fps': round(fps, 1),
            'map': round(mAP_approx, 2),
            'miss_rate': round((self.unmatched_boxes / self.total_yolo_boxes)*100, 2) if self.total_yolo_boxes > 0 else 0
        }

    def save_csv(self, filename="final_results_2.csv"):
        s = self.get_summary()
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=s.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(s)
        print(f"--- [SUMMARY {self.sequence_name}] ---")
        print(f"MRE: {s['mre']*100:.2f}% | FPS: {s['fps']} | Miss: {s['miss_rate']}%")