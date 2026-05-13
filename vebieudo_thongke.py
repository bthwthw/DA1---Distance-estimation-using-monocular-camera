import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

file_pattern = 'logs/4_*_details.csv'
all_files = sorted(glob.glob(file_pattern))

if not all_files:
    print(f"Không tìm thấy file nào khớp với '{file_pattern}'. Vui lòng kiểm tra lại đường dẫn!")
    exit()

print(f"Đang xử lý tổng cộng {len(all_files)} file CSV...")

df_list = []
for f in all_files:
    temp_df = pd.read_csv(f)
    temp_df['source_file'] = os.path.basename(f)
    temp_df['csv_line_number'] = temp_df.index + 2 
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

df['abs_error'] = (df['dist_pred'] - df['dist_gt']).abs()
df['error_pct'] = (df['abs_error'] / df['dist_gt']) * 100

bins = [0, 10, 20, 30, 40, 50]
labels = ['0-10m', '10-20m', '20-30m', '30-40m', '40-50m']
df['dist_bin'] = pd.cut(df['dist_gt'], bins=bins, labels=labels, right=False)

agg_df = df.groupby('dist_bin', observed=False).agg(
    count=('dist_gt', 'count'),
    mean_abs_err=('abs_error', 'mean'),
    median_abs_err=('abs_error', 'median'),
    max_abs_err=('abs_error', 'max'),
    mean_err_pct=('error_pct', 'mean'),
    median_err_pct=('error_pct', 'median')
).reset_index()

agg_df_rounded = agg_df.round(2)
print("\n=== Final Error Statistics by Distance Range ===")
print(agg_df_rounded.to_string())

agg_csv_name = 'final_aggregated_error_stats_4.csv'
agg_df.to_csv(agg_csv_name, index=False)
print(f"\nĐã lưu bảng thống kê vào file: {agg_csv_name}")

unique_bins = df['dist_bin'].dropna().unique()
unique_bins = sorted(unique_bins, key=lambda x: float(x.split('-')[0])) 

outliers_list = []
columns_to_print = ['source_file', 'csv_line_number', 'dist_gt', 'dist_pred', 'abs_error', 'error_pct']

for bin_label in unique_bins:
    bin_df = df[df['dist_bin'] == bin_label]
    
    if bin_df.empty:
        continue
        
    worst_cases = bin_df.sort_values(by='abs_error', ascending=False).head(10)
    outliers_list.append(worst_cases)
    
    print(f"\n--- TOP 10 LỖI NẶNG NHẤT DẢI {bin_label} ---")
    print(worst_cases[columns_to_print].round(2).to_string(index=False))

if outliers_list:
    final_outliers_df = pd.concat(outliers_list)
    outliers_csv_name = 'outliers_by_distance_bins_4.csv'
    final_outliers_df.to_csv(outliers_csv_name, index=False)
    print(f"\n=> Đã xuất toàn bộ danh sách phân loại ra file: {outliers_csv_name}")

# ==========================================
# 6. VẼ BIỂU ĐỒ (VISUALIZATION)
# ==========================================
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='dist_gt', y='abs_error', alpha=0.5, color='purple', s=15)
plt.title('Absolute Error vs Ground Truth (16 seqs)', fontsize=14, pad=15)
plt.xlabel('Ground Truth Distance (m)', fontsize=12)
plt.ylabel('Absolute Error (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='dist_bin', y='error_pct', palette='flare')
plt.title('Percentage Error vs Distance Range (16 seqs)', fontsize=14, pad=15)
plt.xlabel('Distance Range (m)', fontsize=12)
plt.ylabel('Percentage Error (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

plt.tight_layout()
plot_img_name = 'final_error_analysis_1.png'
plt.savefig(plot_img_name, dpi=300) 
print(f"Đã lưu biểu đồ vào file: {plot_img_name}")

plt.show()