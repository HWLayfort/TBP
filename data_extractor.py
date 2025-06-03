import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PINN = ['pinn', 'fpinn', 'respinn']
PINO = ['fno', 'tfno', 'deeponet']

# test,4.9004140987115745
TEST_RESULT_PATTREN = re.compile(r'test,([\d.]+)')

def extract_log_data(log_path):
    line = ''
    
    with open(log_path, 'r') as f:
        line = f.readlines()[-1]
        
    match = TEST_RESULT_PATTREN.search(line)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not find test result in log file: {log_path}")
    
def extract_single_model_data() -> pd.Series:
    data = pd.Series(dtype=float, index=PINN)
    for model in PINN:
        log_path = os.path.join(os.path.dirname(__file__), 'logs', f'{model}.log')
        if os.path.exists(log_path):
            try:
                data[model] = extract_log_data(log_path)
            except ValueError as e:
                print(e)
                data[model] = None
        else:
            print(f"Log file does not exist: {log_path}")
            data[model] = None
            
    return data

def extract_combined_model_data() -> pd.DataFrame:
    data = pd.DataFrame(columns=PINN, index=PINO)
    for pino in PINO:
        for pinn in PINN:
            log_name = f"{pino}_to_{pinn}.log"
            log_path = os.path.join(os.path.dirname(__file__), 'logs', log_name)
            if os.path.exists(log_path):
                try:
                    data.loc[pino, pinn] = extract_log_data(log_path)
                except ValueError as e:
                    print(e)
                    data.loc[pino, pinn] = None
            else:
                print(f"Log file does not exist: {log_path}")
                data.loc[pino, pinn] = None
    
    return data

def plot_single_model_data(data):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(data.index, data.values, width=0.5, color=plt.cm.tab10.colors[:len(data)], edgecolor='lightgray', label=data.index)
    original_value = data.values[0] if len(data) > 0 else 1.0  # 첫 번째 값이 없을 경우 대비
    percentages = ((original_value - data.values) / original_value) * 100

    # 각 bar 위에 % 표시
    for i, (rect, loss, percent) in enumerate(zip(bars, data.values, percentages)):
        if i != 0:
            plt.text(rect.get_x() + rect.get_width() / 2, loss + 0.01 * max(data.values),
                    f"↓{percent:.1f}%", ha='center', va='bottom', fontsize=9)

    # PINN 기준선
    if 'pinn' in data:
        plt.axhline(y=data['pinn'], color='red', linestyle='--', linewidth=2, alpha=0.7, label='PINN Baseline')

    # 기타 설정
    plt.title('Single Model Test Loss(MSE)', fontsize=14, weight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Test Result', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # 저장
    plt.savefig(os.path.join('plots', 'single_model_results.png'), dpi=300)
    plt.close()
    
def plot_combined_model_data():
    single_data = extract_single_model_data()
    combined_data = extract_combined_model_data()

    # 'Original' 행 추가
    full_data = pd.concat([pd.DataFrame([single_data], index=['Original']), combined_data])
    full_data = full_data.astype(float)

    # 그래프 설정
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.2
    index = np.arange(len(PINN))
    offset_map = {
        'Original': -1.5 * bar_width,
        'fno': -0.5 * bar_width,
        'tfno': 0.5 * bar_width,
        'deeponet': 1.5 * bar_width,
    }
    color_map = {
        'Original': '#1f77b4',
        'fno': '#ff7f0e',
        'tfno': '#2ca02c',
        'deeponet': '#d62728',
    }
    
    original_values = full_data.loc['Original'].values

    # 각 행(PINO 또는 Original)을 하나의 그룹으로 그리기
    for i, method in enumerate(full_data.index):
        offset = offset_map.get(method, i * bar_width)
        values = full_data.loc[method].values
        bars = plt.bar(index + offset, values, width=bar_width, label=method, color=color_map.get(method, None))

        # 퍼센트 표시 (Original 제외)
        if method != 'Original':
            for i, (bar, val) in enumerate(zip(bars, values)):
                if not np.isnan(val) and original_values[i] > 0:
                    percent = (original_values[i] - val) / original_values[i] * 100
                    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * max(values),
                             f"↓{percent:.1f}%", ha='center', va='bottom', fontsize=9)

    # 기준선
    if 'pinn' in single_data:
        plt.axhline(y=single_data['pinn'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='PINN Baseline')

    # 각종 라벨 및 눈금
    plt.xticks(index, PINN, fontsize=11)
    plt.ylabel('Test Loss (MSE)', fontsize=12)
    plt.title('Nueral Operator → PINN Transfer Loss', fontsize=14, weight='bold')
    plt.legend(title='Pretrained by', fontsize=10)
    plt.grid(axis='y', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    # 저장
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', 'combined_model_results.png'), dpi=300)
    plt.close()
    
if __name__ == '__main__':
    data = extract_single_model_data()
    plot_single_model_data(data)
    print("Data extraction and plotting completed.")
    plot_combined_model_data()
    
    