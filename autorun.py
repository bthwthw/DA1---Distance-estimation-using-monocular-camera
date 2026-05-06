import os
from main import VisionSystem

BASE_PATH = 'C:/Users/Thu/Downloads/'
IMG_ROOT = BASE_PATH + 'data_tracking_image_2/training/image_02/'
CALIB_ROOT = BASE_PATH + 'data_tracking_calib/training/calib/'
LABEL_ROOT = BASE_PATH + 'data_tracking_label_2/training/label_02/'

sequences = [f"{i:04d}" for i in range(21)]

print(f"Autorun: {len(sequences)} sets...")

for seq in sequences:
    img_dir = os.path.join(IMG_ROOT, seq)
    calib_file = os.path.join(CALIB_ROOT, f"{seq}.txt")
    label_file = os.path.join(LABEL_ROOT, f"{seq}.txt")

    if os.path.exists(img_dir) and os.path.exists(calib_file):
        print(f"\n>>> Set: {seq}...")
        try:
            app = VisionSystem(img_dir, calib_file, label_file)
            app.run(headless=True) 
        except Exception as e:
            print(f"[ERROR] set {seq}: {e}")
    else:
        print(f"Skip set {seq} ")

print("\n✨ XONG ✨")