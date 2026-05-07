from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolov8n-seg_openvino_model/', conf_threshold=0.6):
        self.model = YOLO(model_path, task='segment')
        self.conf = conf_threshold

    def track_objects(self, frame):
        results = self.model.track(frame, persist=True, conf=self.conf, verbose=False, retina_masks=True)
        return results

    def get_bottom_center(self, box, mask_contour=None):
        """
        Trích xuất điểm V_bottom từ viền mặt nạ
        """
        # Dùng Mask nếu có dữ liệu 
        if mask_contour is not None and len(mask_contour) > 0:
            # Lấy pixel thấp nhất (Y lớn nhất) của mặt nạ
            v_bottom = np.max(mask_contour[:, 1])
            
            # Tâm U (trục ngang) vẫn lấy từ Bounding Box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            u = (x1 + x2) / 2.0
            return int(u), int(v_bottom)

        # Quay về dùng đáy Bounding Box nếu frame này mất mask
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return int((x1 + x2) / 2.0), int(y2)