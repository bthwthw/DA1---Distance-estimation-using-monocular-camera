"""
Phân tích bias lỗi khoảng cách theo class (car vs person)
Dữ liệu: 2_XXXX_details.csv (có cls_id) + final_results_2.csv
"""

import pandas as pd
import glob
import numpy as np
import os

# ─── Load all detail CSVs ──────────────────────────────────────────────────
search_pattern = os.path.join('logs', '2_*_details.csv')
files = sorted(glob.glob(search_pattern))

all_df = []

for f in files:
    # Lấy tên file chính xác bất chấp hệ điều hành (Windows hay Linux)
    # Ví dụ f = "logs\2_0000_details.csv" -> basename = "2_0000_details.csv"
    basename = os.path.basename(f)
    
    # Xử lý chuỗi trên basename
    seq = basename.replace('2_', '').replace('_details.csv', '')
    
    if not seq.isdigit():
        print(f"[Warning] Bỏ qua file không đúng định dạng: {f}")
        continue
        
    try:
        df = pd.read_csv(f)
        df['seq'] = seq
        all_df.append(df)
    except Exception as e:
        print(f"[Error] Không thể đọc file {f}: {e}")

# Kiểm tra an toàn trước khi concat
if not all_df:
    print("Không có data nào được đọc! Vui lòng kiểm tra lại thư mục logs.")
else:
    df = pd.concat(all_df, ignore_index=True)
    df['cls_name'] = df['cls_id'].map({0:'person', 2:'car'})
    df['signed_pct'] = (df['dist_pred'] - df['dist_gt']) / df['dist_gt'] * 100

    # ─── TABLE 1: Tổng quan theo class ────────────────────────────────────────
    t1 = df.groupby('cls_name').agg(
        Samples=('error_pct','count'),
        MRE_mean=('error_pct','mean'),
        MRE_median=('error_pct','median'),
        MRE_std=('error_pct','std'),
        Bias_mean=('signed_pct','mean'),
        Bias_median=('signed_pct','median'),
        Overestimate_pct=('signed_pct', lambda x: (x>0).mean()*100),
    ).round(2)

    # ─── TABLE 2: Error theo khoảng cách ──────────────────────────────────────
    df['dist_bin'] = pd.cut(df['dist_gt'],
                            bins=[0,5,10,20,40,200],
                            labels=['0–5 m','5–10 m','10–20 m','20–40 m','40+ m'])
    t2 = df.groupby(['cls_name','dist_bin'], observed=True).agg(
        Samples=('error_pct','count'),
        MRE_mean=('error_pct','mean'),
        Bias_mean=('signed_pct','mean'),
    ).round(2)

    # ─── TABLE 3: Per-sequence ────────────────────────────────────────────────
    t3 = df.groupby(['seq','cls_name']).agg(
        n=('error_pct','count'),
        MRE=('error_pct','mean'),
        Bias=('signed_pct','mean'),
    ).round(2)

    # ─── PRINT ────────────────────────────────────────────────────────────────
    sep = "=" * 70

    print(sep)
    print("TABLE 1 — Tổng quan lỗi khoảng cách theo class")
    print(sep)
    print(t1.rename(columns={
        'Samples':'N mẫu',
        'MRE_mean':'MRE trung bình (%)',
        'MRE_median':'MRE trung vị (%)',
        'MRE_std':'MRE std (%)',
        'Bias_mean':'Bias TB (%)',
        'Bias_median':'Bias trung vị (%)',
        'Overestimate_pct':'Tỉ lệ overestimate (%)',
    }).to_string())

    print(f"""
    Ghi chú:
    - MRE = |pred - gt| / gt × 100  (luôn dương)
    - Bias = (pred - gt) / gt × 100  (dương = ước tính xa hơn thực tế)
    - Overestimate = tỉ lệ dự đoán > gt
    """)

    print(sep)
    print("TABLE 2 — Lỗi theo khoảng cách thực (dist_gt)")
    print(sep)
    print(t2.rename(columns={
        'Samples':'N mẫu',
        'MRE_mean':'MRE TB (%)',
        'Bias_mean':'Bias TB (%)',
    }).to_string())

    print(f"""
    Nhận xét: Cả 2 class đều tăng lỗi khi vật thể xa hơn — đây là giới hạn
    tự nhiên của geometric projection (lỗi pixel nhỏ → lỗi z lớn ở xa).
    """)

    print(sep)
    print("TABLE 3 — Per-sequence: MRE và Bias theo class")
    print(sep)
    t3_wide = t3.unstack('cls_name')
    t3_wide.columns = [f"{m}_{c}" for m,c in t3_wide.columns]
    print(t3_wide[['n_car','MRE_car','Bias_car','n_person','MRE_person','Bias_person']].to_string())

    print(f"""
    Lưu ý seq đặc biệt:
    0015 car   : MRE = 71.93%  → xe bị che khuất/ở góc, YOLO miss điểm chạm đất
    0017 person: MRE = 53.61%  → camera gắn vỉa hè (CCTV-style), v_bottom lệch
    0018 car   : MRE = 34.44%  → nắng gắt + bóng râm, bounding box không ổn định
    0019 car   : MRE = 53.13%  → xe hơi xuất hiện trong đường nội thành hẹp
    """)

# ─── KẾT LUẬN ─────────────────────────────────────────────────────────────
excel_filename = "analyze_results.xlsx"

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Lưu Table 1 vào Sheet 1
    t1.to_excel(writer, sheet_name="T1_Tong_quan")
    
    # Lưu Table 2 vào Sheet 2
    t2.to_excel(writer, sheet_name="T2_Theo_khoang_cach")
    
    # Lưu Table 3 vào Sheet 3
    t3_wide.to_excel(writer, sheet_name="T3_Per_Sequence")

print(f"Đã xuất dữ liệu thành công ra file Excel: {excel_filename}")
    # print(sep)
    # print("KẾT LUẬN — Có phải class xe hơi bị hệ thống bias không?")
    # print(sep)
    # print("""
    # 1. VỀ MRE TỔNG:
    #    car   : MRE trung bình ~22.1%, trung vị ~14.1%
    #    person: MRE trung bình ~22.5%, trung vị ~15.5%
    #    → Hai class CÓ MRE gần nhau ở mức tổng thể.
    #    → Không thể kết luận car bị "offset hệ thống" so với person chỉ nhìn MRE.

    # 2. VỀ HƯỚNG BIAS (quan trọng hơn):
    #    car   : Bias trung bình = +15.0%,  trung vị = +8.1%,   overestimate 64.7%
    #    person: Bias trung bình = +22.1%,  trung vị = +15.4%,  overestimate 96.6%

    #    → CẢ HAI CLASS đều bị OVERESTIMATE (dự đoán xa hơn thực tế).
    #    → Person bị overestimate mạnh hơn và nhất quán hơn (96.6% số mẫu).
    #    → Car cũng overestimate nhưng có ~35% mẫu underestimate — dao động 2 chiều.

    # 3. NGUYÊN NHÂN CHỦ YẾU (không phải bug code):
    #    a) Person: Người không đứng trên mặt đường (đứng trên vỉa hè, bậc thềm,
    #       hoặc bị cắt dưới knee) → v_bottom KHÔNG phải điểm chạm đất thật →
    #       công thức Z = f_y*C_h / (v_bottom - v_horizon) cho kết quả luôn lớn hơn gt.
    #    b) Car: Xe không phải lúc nào cũng nằm trên cùng mặt phẳng với camera.
    #       Xe đậu nghiêng, xe leo lề, xe đi ngược chiều → v_bottom lệch. 
    #       Nhưng xe thường ĐỨNG TRÊN ĐƯỜNG → ít bị lệch hơn người.
    #    c) Seq outlier: 0015/0017/0018/0019 kéo bias car lên mạnh.
    #       Bỏ 4 seq này → car bias về ~3–10% — rất chấp nhận được.

    # 4. KẾT LUẬN CUỐI:
    #    → KHÔNG có "hệ thống lỗi cố định" (fixed offset) riêng cho class car.
    #    → Car bị lỗi CAO ở một số seq do ĐIỀU KIỆN SCENE (không phải do class).
    #    → Person bị bias overestimate CÓ HỆ THỐNG hơn car, vì lý do hình học
    #      (v_bottom không phải điểm tiếp đất).
    #    → Để cải thiện: phân biệt offset chiều cao vật thể theo class
    #      (car: dùng v_bottom ≈ điểm tiếp đất; person: dùng v_bottom - offset_knee).
    # """)