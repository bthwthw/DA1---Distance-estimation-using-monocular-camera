# main.py
import cv2
import os
import glob
import numpy as np
from modules.detector import ObjectDetector
from modules.estimator import DistanceEstimator
from modules.evaluator import KittiLabelReader, calculate_iou
from modules.kalman import KalmanFilter1D, KalmanFilter2D
from modules.logger import SystemLogger
import time
from collections import deque

MODEL_PATH = 'yolov8n_openvino_model/'
CAMERA_HEIGHT = 1.65  # Vị trí camera theo kitti 

def read_kitti_calib(calib_path):
    """
    Đọc file txt của KITTI, trích xuất F_y và C_y từ ma trận P2 
    """
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                values = [float(x) for x in line.strip().split()[1:]]
                f_y = values[5] # Tiêu cự trục dọc
                c_y = values[6] # Tọa độ đường chân trời
                return f_y, c_y
    return None, None

class VisionSystem:
    def __init__(self, sequence_dir, calib_file, label_file):
        """
        :param sequence_dir: Thư mục chứa các ảnh 000000.png, 000001.png...
        :param calib_file: File txt chứa ma trận camera của sequence đó
        :param label_file: File txt chứa nhãn ground truth của sequence đó

        """
        self.sequence_dir = sequence_dir
        self.f_y, self.c_y = read_kitti_calib(calib_file)
        
        self.detector = ObjectDetector(model_path=MODEL_PATH)
        self.alert_counters = {} 
        self.estimator = DistanceEstimator(self.f_y, CAMERA_HEIGHT, self.c_y)
        self.label_reader = KittiLabelReader(label_file)

        self.image_paths = sorted(glob.glob(os.path.join(sequence_dir, '*.png')))
        
        self.history = {} 
        self.kalman_filters = {}
        self.kalman_filters_2 = {}
        self.dist_history = {}
        self.geom_cooldown = {}
        self.h_history = {}      # Lưu danh sách H của 10 frame đầu
        self.calibrated_H = {}   # Lưu con số H trung bình đã được chốt

        self.fps_assumed = 10.0 # dataset kitti 10fps 

    def run(self, headless=False):
            print(f"System started. Mode: {'Headless' if headless else 'GUI'}")
            logger = SystemLogger(self.sequence_dir.split('/')[-1])
            
            for frame_idx, img_path in enumerate(self.image_paths):
                frame = cv2.imread(img_path)
                if frame is None: continue
                start_inf = time.time()

                # Thiết lập hành lang
                h_img, w_img = frame.shape[:2]
                center_x, horizon_y = w_img // 2, int(self.c_y) - 20
                top_w, bottom_w = 50, 250
                corridor_pts = np.array([[center_x - top_w, horizon_y], [center_x + top_w, horizon_y],
                                        [center_x + bottom_w, h_img], [center_x - bottom_w, h_img]], np.int32)

                gt_objects = self.label_reader.get_gt_for_frame(frame_idx)
                results = list(self.detector.track_objects(frame))

                if results and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in [0, 2] and box.id is not None:
                            obj_id = int(box.id[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            box_h_raw = y2 - y1

                            if obj_id not in self.dist_history:
                                self.dist_history[obj_id] = deque(maxlen=10) 
                                self.h_history[obj_id] = deque(maxlen=10)    
                                self.calibrated_H[obj_id] = None             
                                self.geom_cooldown[obj_id] = 0
                                self.kalman_filters[obj_id] = KalmanFilter1D(0.1, 5.0, box_h_raw)
                            
                            box_h_muot = self.kalman_filters[obj_id].update(box_h_raw)
                            
                            # Tính sẵn 2 loại khoảng cách thô
                            u, v_bottom_raw = self.detector.get_bottom_center(box)
                            if cv2.pointPolygonTest(corridor_pts, (u, v_bottom_raw), False) < 0:
                                continue
                            raw_dist_gcp = self.estimator.estimate_ground(v_bottom_raw)
                            
                            # Nếu chưa có H chuẩn, mượn tạm 1.52m để làm gốc check lỗi
                            my_H = self.calibrated_H[obj_id] if self.calibrated_H[obj_id] else 1.52
                            raw_dist_geom = (self.f_y * my_H) / box_h_muot

                            is_glitch = False
                            if len(self.dist_history[obj_id]) > 0:
                                last_dist = self.dist_history[obj_id][-1]
                                
                                # Chọn khoảng cách để check lỗi tùy theo trạng thái
                                check_dist = raw_dist_geom if self.calibrated_H[obj_id] else raw_dist_gcp
                                
                                dynamic_threshold = max(3.0, last_dist * 0.25)
                                if abs(check_dist - last_dist) > dynamic_threshold:
                                    is_glitch = True
                                
                                if len(self.dist_history[obj_id]) >= 4:
                                    past_dists = list(self.dist_history[obj_id])
                                    deltas = [past_dists[i] - past_dists[i-1] for i in range(1, len(past_dists))]
                                    if np.std(deltas) > 1.5:
                                        is_glitch = True

                            if is_glitch:
                                self.geom_cooldown[obj_id] = 3 

                            if self.geom_cooldown[obj_id] > 0:
                                # ĐANG BỊ LỖI (CHATTERING / OCCLUSION)
                                current_distance = self.dist_history[obj_id][-1] if len(self.dist_history[obj_id])>0 else raw_dist_geom
                                method_used = f"HOLD({self.geom_cooldown[obj_id]})"
                                
                                self.h_history[obj_id].clear()
                                
                                # Ép Kalman đứng im
                                box_h_virtual = (self.f_y * my_H) / current_distance
                                self.kalman_filters[obj_id].x = box_h_virtual 
                                
                                self.geom_cooldown[obj_id] -= 1
                                
                            else:
                                if self.calibrated_H[obj_id] is None:
                                    # Giai đoạn Hiệu chuẩn: Đo bằng GCP
                                    current_distance = raw_dist_gcp
                                    
                                    if current_distance > 0:
                                        real_H = (current_distance * box_h_muot) / self.f_y
                                        self.h_history[obj_id].append(real_H)
                                        
                                    method_used = f"CALIB ({len(self.h_history[obj_id])}/10)"
                                    
                                    if len(self.h_history[obj_id]) == 10:
                                        h_std = np.std(self.h_history[obj_id])
                                        
                                        # Ngưỡng lệch chuẩn: 0.15m
                                        if h_std < 0.15: 
                                            self.calibrated_H[obj_id] = np.mean(self.h_history[obj_id])
                                        else:
                                            method_used = f"NOISY (std:{h_std:.2f})"
                                else:
                                    # Giai đoạn Tracking: Xài GEOMETRY 
                                    current_distance = raw_dist_geom
                                    method_used = "GEOM_TRACK"

                            self.dist_history[obj_id].append(current_distance)
                            
                            if current_distance < 0: continue
                            
                            # Matching Label 
                            best_iou = 0
                            best_z_gt = -1
                            for gt_obj in gt_objects:
                                iou = calculate_iou([x1, y1, x2, y2], gt_obj['bbox'])
                                if iou > best_iou:
                                    best_iou, best_z_gt = iou, gt_obj['z_gt']
                            
                            if best_iou > 0.5:
                                logger.log_match(frame_idx, obj_id, current_distance, best_z_gt, best_iou)
                                if best_z_gt != -1:
                                    error_print = best_z_gt - current_distance
                                    print(f"Frame {frame_idx} | ID {obj_id} | ERROR: {error_print:.2f}m | Method: {method_used}")
                                else:
                                    print(f"Frame {frame_idx} | ID {obj_id} | KHÔNG CÓ NHÃN GT | Pred: {current_distance:.2f}m | Method: {method_used}")

                            else:
                                logger.log_unmatched()
                            
                            # TTC
                            ttc, status, color = float('inf'), "GO", (0, 255, 0)

                            if obj_id in self.history:
                                
                                raw_v_rel = (self.history[obj_id] - current_distance) * self.fps_assumed
                                
                                if raw_v_rel > 0.5 and current_distance < 15.0: 
                                    ttc = current_distance / raw_v_rel
                                    
                                    if ttc <= 2.5 or current_distance < 2.0: 
                                        new_status = "STOP"
                                        new_color = (0, 0, 255)
                                    elif ttc <= 5.0:
                                        new_status = "WARN"
                                        new_color = (0, 255, 255)
                                    else:
                                        new_status = "GO"
                                        new_color = (0, 255, 0)
                                        
                                    if new_status == "STOP":
                                        self.alert_counters[obj_id] = self.alert_counters.get(obj_id, 0) + 1
                                    else:
                                        self.alert_counters[obj_id] = 0
                                        
                                    if self.alert_counters.get(obj_id, 0) >= 5:
                                        status, color = f"!!! {new_status}: {ttc:.1f}s !!!", new_color
                                    elif new_status == "WARN":
                                        status, color = f"{new_status}: {ttc:.1f}s", new_color
                                else:
                                    self.alert_counters[obj_id] = 0

                            self.history[obj_id] = current_distance

                            if not headless:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                label = f"ID:{obj_id}|D:{current_distance:.1f}m|GT:{best_z_gt:.1f}m"
                                cv2.putText(frame, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                                cv2.putText(frame, f"Status: {status}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                logger.log_frame(time.time() - start_inf)
                
                if not headless:
                    cv2.polylines(frame, [corridor_pts], True, (255, 100, 0), 2)
                    cv2.imshow("Robot Vision Demo", frame)
                    if cv2.waitKey(1) == ord('q'): break

            logger.save_csv()

if __name__ == "__main__":
    IMG_DIR = 'C:/Users/Thu/Downloads/data_tracking_image_2/training/image_02/0010' 
    CALIB_FILE = 'C:/Users/Thu/Downloads/data_tracking_calib/training/calib/0010.txt'
    LABEL_FILE = 'C:/Users/Thu/Downloads/data_tracking_label_2/training/label_02/0010.txt'
    app = VisionSystem(IMG_DIR, CALIB_FILE, LABEL_FILE)
    app.run()
    