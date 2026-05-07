# main.py
import cv2
import os
import glob
import numpy as np
from modules.detector import ObjectDetector
from modules.estimator import DistanceEstimator
from modules.evaluator import KittiLabelReader, calculate_iou
from modules.filter import KalmanFilter1D, KalmanFilter2D, DistanceSmoother
from modules.logger import SystemLogger
import time

MODEL_PATH = 'yolov8n_openvino_model/'
CAMERA_HEIGHT = 1.65  # Vị trí camera theo kitti 

def read_kitti_calib(calib_path):
    """Đọc file calib KITTI, trả về f_y, c_y, v_horizon tính đúng từ pitch."""
    P2 = R_rect = Tr = None
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
            elif line.startswith('R_rect'):
                R_rect = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 3)
            elif line.startswith('Tr_velo_cam'):
                Tr = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)

    f_y = P2[1, 1]
    c_y = P2[1, 2]

    # Ground normal trong velo frame = [0, 0, 1] (Z_velo = hướng lên)
    n_velo = np.array([0.0, 0.0, 1.0])
    n_cam = R_rect @ Tr[:, :3] @ n_velo
    n_cam /= np.linalg.norm(n_cam)

    # Pitch = angle của optical axis (Z_cam=[0,0,1]) với mặt phẳng ngang
    pitch_rad = np.arcsin(float(np.dot([0.0, 0.0, 1.0], n_cam)))

    # v_horizon = c_y - f_y * tan(pitch)
    # pitch > 0 (nhìn lên): horizon thấp hơn c_y
    # pitch < 0 (nhìn xuống nhẹ): horizon cao hơn c_y (đúng với dashcam)
    v_horizon = c_y - f_y * np.tan(pitch_rad)

    return f_y, c_y, v_horizon

class VisionSystem:
    def __init__(self, sequence_dir, calib_file, label_file):
        """
        :param sequence_dir: Thư mục chứa các ảnh 000000.png, 000001.png...
        :param calib_file: File txt chứa ma trận camera của sequence đó
        :param label_file: File txt chứa nhãn ground truth của sequence đó

        """
        self.sequence_dir = sequence_dir
        
        self.f_y, self.c_y, v_horizon = read_kitti_calib(calib_file)
        self.estimator = DistanceEstimator(self.f_y, CAMERA_HEIGHT, v_horizon)
        
        self.detector = ObjectDetector(model_path=MODEL_PATH)
        self.alert_counters = {} 
        self.label_reader = KittiLabelReader(label_file)

        self.image_paths = sorted(glob.glob(os.path.join(sequence_dir, '*.png')))
        
        self.history = {} 
        self.kalman_filters = {}
        self.kalman_filters_2 = {}
        self.dist_smoothers = {}    # Lọc distance 2 tầng (Median → EMA)

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
                            u, v_bottom = self.detector.get_bottom_center(box)
                            
                            if cv2.pointPolygonTest(corridor_pts, (u, v_bottom), False) < 0:
                                continue

                            # Kalman & Distance
                            if obj_id not in self.kalman_filters:
                                self.kalman_filters[obj_id] = KalmanFilter1D(2.0, 1.5, v_bottom)
                            v_bottom_muot = self.kalman_filters[obj_id].update(v_bottom)
                            raw_distance = self.estimator.estimate(v_bottom_muot)
                            if raw_distance < 0: continue

                            # if obj_id not in self.kalman_filters_2:
                            #     self.kalman_filters_2[obj_id] = KalmanFilter2D(
                            #         dt=1.0/self.fps_assumed, 
                            #         process_noise=5.0, 
                            #         measurement_noise=0.01, 
                            #         initial_pos=raw_distance
                            #     )
                            # box_w, box_h = x2 - x1, y2 - y1
                            # current_distance = self.kalman_filters_2[obj_id].update(raw_distance, box_w, box_h)
                            # Lọc 2 tầng trên miền distance: Median(w=3) → EMA(α=0.25)
                            if obj_id not in self.dist_smoothers:
                                self.dist_smoothers[obj_id] = DistanceSmoother(ema_alpha=0.25, median_window=3)
                            current_distance = self.dist_smoothers[obj_id].update(raw_distance)

                            # Matching Label 
                            best_iou = 0
                            best_z_gt = -1
                            for gt_obj in gt_objects:
                                iou = calculate_iou([x1, y1, x2, y2], gt_obj['bbox'])
                                if iou > best_iou:
                                    best_iou, best_z_gt = iou, gt_obj['z_gt']
                            
                            if best_iou > 0.5:
                                logger.log_match(frame_idx, cls_id, obj_id, current_distance, best_z_gt, best_iou)
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
    IMG_DIR = 'C:/Users/Thu/Downloads/data_tracking_image_2/training/image_02/0017' 
    CALIB_FILE = 'C:/Users/Thu/Downloads/data_tracking_calib/training/calib/0017.txt'
    LABEL_FILE = 'C:/Users/Thu/Downloads/data_tracking_label_2/training/label_02/0017.txt'
    app = VisionSystem(IMG_DIR, CALIB_FILE, LABEL_FILE)
    app.run()