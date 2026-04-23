# modules/detector.py
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        print(f"Loading model from: {model_path}...")
        self.model = YOLO(model_path, task='detect') 
        self.conf = conf_threshold

    def track_objects(self, frame):
        """
        persist=True: Yêu cầu YOLO nhớ ID của vật thể qua từng frame liên tiếp.
        """
        results = self.model.track(frame, persist=True, stream=True, conf=self.conf, verbose=False, tracker="bytetrack.yaml")
        return results
