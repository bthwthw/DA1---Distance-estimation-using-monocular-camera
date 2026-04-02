import os
from main import VisionSystem

# Cấu hình đường dẫn gốc (Thu sửa lại cho đúng máy mình nhé)
BASE_PATH = 'C:/Users/Thu/Downloads/'
IMG_ROOT = BASE_PATH + 'data_tracking_image_2/training/image_02/'
CALIB_ROOT = BASE_PATH + 'data_tracking_calib/training/calib/'
LABEL_ROOT = BASE_PATH + 'data_tracking_label_2/training/label_02/'

# Danh sách các set muốn chạy (ví dụ từ 0000 đến 0020)
sequences = [f"{i:04d}" for i in range(21)]

print(f"Bắt đầu chuỗi chạy tự động cho {len(sequences)} sets...")

for seq in sequences:
    img_dir = os.path.join(IMG_ROOT, seq)
    calib_file = os.path.join(CALIB_ROOT, f"{seq}.txt")
    label_file = os.path.join(LABEL_ROOT, f"{seq}.txt")

    # Kiểm tra xem file có tồn tại không trước khi chạy
    if os.path.exists(img_dir) and os.path.exists(calib_file):
        print(f"\n>>> Đang xử lý Set: {seq}...")
        try:
            app = VisionSystem(img_dir, calib_file, label_file)
            app.run(headless=True) 
        except Exception as e:
            print(f"Lỗi ở set {seq}: {e}")
    else:
        print(f"Bỏ qua set {seq} do thiếu dữ liệu.")

print("\n✨ XONG ✨")