import cv2
import time
import psutil
import os
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

# model = YOLO("yolov8n.pt")
model = YOLO('yolo11n.pt')
# model.export(format='onnx')
model.export(format='openvino')

# --- CẤU HÌNH ---
# MODEL_TO_TEST = 'yolov8n.torchscript' 
# MODEL_TO_TEST = 'yolov8n.onnx' 
# MODEL_TO_TEST = 'yolo11n_openvino_model/' 

def main():
    print(f"Đang load model: {MODEL_TO_TEST}...")
    model = YOLO(MODEL_TO_TEST, task='detect')
    
    cap = cv2.VideoCapture(0)
    
    # Biến để tính thông số 
    frame_count = 0
    start_time = time.time()
    fps_list = []
    cpu_samples = []
    ram_samples = []
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret: break

            # Chạy nhận diện
            results = model(frame, verbose=False)

            # Tính FPS tức thời
            loop_end = time.time()
            fps = 1 / (loop_end - loop_start)
            fps_list.append(fps)
            frame_count += 1

            # Lấy thông số CPU/RAM hiện tại
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 # MB

            # Hiển thị lên màn hình
            cv2.putText(frame, f"Model: {MODEL_TO_TEST}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"CPU: {cpu_usage}% | RAM: {int(ram_usage)}MB", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow('Benchmark', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Lấy mẫu CPU/RAM sau 20 khung hình đầu tiên tranh giai doan warm-up 
            if frame_count > 20:
                cpu_samples.append(psutil.cpu_percent())
                ram_samples.append(psutil.Process().memory_info().rss / (1024 * 1024))

            # Tự động dừng sau 30 giây
            if time.time() - start_time > 30:
                print("Đã hết 30 giây test!")
                break

    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()

    # --- TỔNG KẾT ---
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"BENCHMARK: {MODEL_TO_TEST}")
    print(f"FPS Trung bình: {avg_fps:.2f}")
    print(f"Tổng số khung hình: {frame_count}")

    if cpu_samples:
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        avg_ram = sum(ram_samples) / len(ram_samples)
        print(f"CPU trung bình: {avg_cpu:.2f}%")
        print(f"RAM trung bình: {avg_ram:.2f} MB")

if __name__ == "__main__":
    main()
    print("benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device='cpu', format='onnx')")
    print("benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device='cpu', format='openvino')")