import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def draw_mre_chart(csv_path):
    df = pd.read_csv(csv_path, dtype={'seq': str})

    colors = []
    for val in df['mre']:
        if val < 0.10:
            colors.append('#2ecc71') 
        elif val < 0.20:
            colors.append('#f1c40f') 
        else:
            colors.append('#e74c3c')

    plt.figure(figsize=(15, 7.5))
    
    # MRE % 
    mre_percent = df['mre'] * 100
    bars = plt.bar(df['seq'], mre_percent, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)

    # Threshold lines
    plt.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    plt.axhline(y=20, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Performance Analysis: Distance Estimation Error per Sequence', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('KITTI Sequence', fontsize=12, labelpad=10)
    plt.ylabel('Mean Relative Error (MRE) %', fontsize=12, labelpad=10)
    
    plt.ylim(0, max(mre_percent) + 5) 
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mre_comparison_chart_2.png', dpi=300)
    print("Đã xuất biểu đồ thành công: mre_comparison_chart_2.png")
    plt.show()


def draw_comparison_chart(seq):
    df = pd.read_csv(f'logs/{seq}_details.csv')
    best_id = df['obj_id'].value_counts().idxmax()
    subset = df[df['obj_id'] == best_id]

    abs_error = (subset['dist_gt'] - subset['dist_pred']).abs()
    relative_error = abs_error / subset['dist_gt']
    mre_id = relative_error.mean()
    mre_percent = mre_id * 100

    plt.figure(figsize=(12, 5))
    
    plt.plot(subset['frame'], subset['dist_gt'], 'k--', label='Thực tế', alpha=0.6)
    plt.plot(subset['frame'], subset['dist_pred'], 'b-', label='Dự đoán', linewidth=2)

    stats_text = f'Object ID: {best_id}\nMRE: {mre_percent:.2f}%'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.title(f'Distance Tracking Analysis - Sequence {seq}', fontsize=14)
    plt.xlabel('Frame Number')
    plt.ylabel('Distance (meters)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(f'plots/comparison_chart_{seq}_2.png')
    plt.show()

draw_mre_chart('final_results_2.csv')
for i in range(21):
    if i == 5 or i == 8 or i == 12 or i == 14:
        continue
    draw_comparison_chart(f"{i:04d}")
# draw_comparison_chart('0010')