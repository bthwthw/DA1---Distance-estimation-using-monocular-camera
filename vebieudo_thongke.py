import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Tìm tất cả các file CSV của phiên bản RANSAC (có tiền tố 4_)
# Nếu bạn muốn phân tích các file không dùng RANSAC, đổi thành '2_*_details_1.csv' hoặc '*_details_1.csv'
file_pattern = 'logs/1_*_details.csv'
all_files = sorted(glob.glob(file_pattern))

if not all_files:
    print(f"Không tìm thấy file nào khớp với '{file_pattern}'. Vui lòng kiểm tra lại đường dẫn!")
    exit()

print(f"Đang xử lý tổng cộng {len(all_files)} file CSV...")

# 2. Đọc và gộp tất cả các file thành một DataFrame duy nhất
df_list = []
for f in all_files:
    temp_df = pd.read_csv(f)
    # Lưu tên file (chỉ lấy tên, bỏ qua đường dẫn dài)
    temp_df['source_file'] = os.path.basename(f)
    # Lưu số dòng trong CSV: index bắt đầu từ 0 + 2 (1 cho header, 1 để khớp dòng thực tế)
    temp_df['csv_line_number'] = temp_df.index + 2 
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# 3. Tính toán các cột sai số
# abs_error: Lệch bao nhiêu mét
df['abs_error'] = (df['dist_pred'] - df['dist_gt']).abs()

# error_pct: Lệch bao nhiêu phần trăm so với khoảng cách thực
df['error_pct'] = (df['abs_error'] / df['dist_gt']) * 100

# 4. Phân vùng khoảng cách (Zoning/Binning)
bins = [0, 10, 20, 30, 40, 50]
labels = ['0-10m', '10-20m', '20-30m', '30-40m', '40-50m']
# right=False nghĩa là lấy [0, 10), [10, 20)
df['dist_bin'] = pd.cut(df['dist_gt'], bins=bins, labels=labels, right=False)

# 5. Gom nhóm và tính toán các chỉ số thống kê (Mean, Median, Max)
agg_df = df.groupby('dist_bin', observed=False).agg(
    count=('dist_gt', 'count'),
    mean_abs_err=('abs_error', 'mean'),
    median_abs_err=('abs_error', 'median'),
    max_abs_err=('abs_error', 'max'),
    mean_err_pct=('error_pct', 'mean'),
    median_err_pct=('error_pct', 'median')
).reset_index()

# Làm tròn số liệu để in ra console cho đẹp
agg_df_rounded = agg_df.round(2)
print("\n=== Final Error Statistics by Distance Range (RANSAC version) ===")
print(agg_df_rounded.to_string())

# Lưu bảng thống kê này ra file CSV để làm báo cáo
agg_csv_name = 'final_ransac_aggregated_error_stats_1.csv'
agg_df.to_csv(agg_csv_name, index=False)
print(f"\nĐã lưu bảng thống kê vào file: {agg_csv_name}")

# Lấy danh sách các dải khoảng cách đã định nghĩa ở bước 4
unique_bins = df['dist_bin'].dropna().unique()
# Sắp xếp lại cho đúng thứ tự từ gần đến xa
unique_bins = sorted(unique_bins, key=lambda x: float(x.split('-')[0])) 

outliers_list = []
columns_to_print = ['source_file', 'csv_line_number', 'dist_gt', 'dist_pred', 'abs_error', 'error_pct']

for bin_label in unique_bins:
    # Lọc dữ liệu thuộc dải hiện tại
    bin_df = df[df['dist_bin'] == bin_label]
    
    if bin_df.empty:
        continue
        
    # Sắp xếp để lấy các ca có sai số tuyệt đối cao nhất
    worst_cases = bin_df.sort_values(by='abs_error', ascending=False).head(10)
    outliers_list.append(worst_cases)
    
    print(f"\n--- TOP 10 LỖI NẶNG NHẤT DẢI {bin_label} ---")
    # Tùy chỉnh format in ra cho gọn, làm tròn 2 chữ số thập phân
    print(worst_cases[columns_to_print].round(2).to_string(index=False))

# Gộp tất cả outliers của các dải lại và lưu ra file CSV
if outliers_list:
    final_outliers_df = pd.concat(outliers_list)
    outliers_csv_name = 'outliers_by_distance_bins_1_1.csv'
    final_outliers_df.to_csv(outliers_csv_name, index=False)
    print(f"\n=> Đã xuất toàn bộ danh sách phân loại ra file: {outliers_csv_name}")

# ==========================================
# 6. VẼ BIỂU ĐỒ (VISUALIZATION)
# ==========================================
plt.figure(figsize=(15, 6)) # Kích thước khung hình (rộng x cao)

# Biểu đồ 1: Scatter plot (Phân tán sai số tuyệt đối)
plt.subplot(1, 2, 1)
# s=15 là kích thước hạt, alpha=0.5 là độ trong suốt giúp nhìn rõ vùng dày đặc
sns.scatterplot(data=df, x='dist_gt', y='abs_error', alpha=0.5, color='purple', s=15)
plt.title('Absolute Error vs Ground Truth (RANSAC - 16 seqs)', fontsize=14, pad=15)
plt.xlabel('Ground Truth Distance (m)', fontsize=12)
plt.ylabel('Absolute Error (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Biểu đồ 2: Boxplot (Sự phân bố của sai số % theo từng dải khoảng cách)
plt.subplot(1, 2, 2)
# palette 'flare' tạo màu gradient từ cam sang tím khá bắt mắt
sns.boxplot(data=df, x='dist_bin', y='error_pct', palette='flare')
plt.title('Percentage Error vs Distance Range (RANSAC - 16 seqs)', fontsize=14, pad=15)
plt.xlabel('Distance Range (m)', fontsize=12)
plt.ylabel('Percentage Error (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45) # Nghiêng nhãn trục X 45 độ cho khỏi đè lên nhau

# Căn chỉnh bố cục và Lưu file ảnh
plt.tight_layout()
plot_img_name = 'final_ransac_error_analysis_1.png'
plt.savefig(plot_img_name, dpi=300) # Lưu với độ phân giải cao (300dpi) để dán vào Word
print(f"Đã lưu biểu đồ vào file: {plot_img_name}")

# (Tùy chọn) Hiện cửa sổ ảnh lên màn hình
# plt.show()