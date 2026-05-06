import os

# Danh sách các số sequence cần bỏ qua
skip_sequences = {5, 8, 12, 14}

# Thư mục chứa các file (để "." nếu chạy script ở cùng thư mục với file csv)
directory = "."

for i in range(22):  # Chạy từ 0 đến 21
    if i in skip_sequences:
        continue
    
    # Định dạng số thành chuỗi 4 ký tự, điền số 0 ở đầu (VD: 0 -> "0000")
    seq_name = f"{i:04d}"
    
    old_name = f"{seq_name}_details_2.csv"
    new_name = f"2_{seq_name}_details.csv"
    
    old_path = os.path.join(directory, old_name)
    new_path = os.path.join(directory, new_name)
    
    # Đổi tên nếu file tồn tại
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Đã đổi tên: {old_name} -> {new_name}")
    else:
        print(f"Bỏ qua (không tìm thấy file): {old_name}")