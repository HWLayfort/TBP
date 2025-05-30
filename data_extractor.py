import os
import re
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
    
def extract_single_model_data():
    data = pd.Series(dtype=float, index=PINN + PINO)
    for model in PINN + PINO:
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

def extract_combined_model_data():
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
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.index, data.values, width=0.5, color=plt.cm.tab10.colors[:len(data)], edgecolor='lightgray', label=data.index)
    percentages = data.values / data.values[0] * 100

    # 각 bar 위에 % 표시
    for rect, loss, percent in zip(bars, data.values, percentages):
        plt.text(rect.get_x() + rect.get_width() / 2, loss + 0.01 * max(data.values),
                f"{loss:.1f}/{percent:.1f}%", ha='center', va='bottom', fontsize=10)

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
    
def plot_combined_model_data(data):
    combine_data = extract_combined_model_data()
    single_data = extract_single_model_data()
    
    
    
if __name__ == '__main__':
    data = extract_single_model_data()
    plot_single_model_data(data)
    print("Data extraction and plotting completed.")
    
    