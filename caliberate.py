import numpy as np
import cv2
import glob

# So goc den trang 
CHECKERBOARD = (8, 5) 

# size cua 1 o vuong (cm)
SQUARE_SIZE = 2.5 

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Tạo hệ tọa độ giả định cho bàn cờ (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

cap = cv2.VideoCapture(0)
print("Bấm 's' để chụp hình bàn cờ. Chụp đủ 15 tấm thì bấm 'q' để tính toán.")
count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tìm góc bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    # Nếu tìm thấy 
    display_frame = frame.copy()
    if ret == True:
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret)
        
    cv2.imshow('Calibration', display_frame)
    
    key = cv2.waitKey(1)
    if key == ord('s') and ret == True:
        # Lưu điểm khi bấm 's' và bàn cờ hợp lệ
        objpoints.append(objp)
        imgpoints.append(corners)
        count += 1
        print(f"Đã chụp: {count}/15 tấm")
        
    if key == ord('q') or count >= 15:
        break

cap.release()
cv2.destroyAllWindows()

# --- TÍNH TOÁN ---
if count > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Ma trận Camera (Intrinsic Matrix):")
    print(mtx)
    fx = mtx[0][0]
    fy = mtx[1][1]
    print(f"Fx = {fx:.2f}")
    print(f"Fy = {fy:.2f}")
else:
    print("Chưa chụp đủ hình!")