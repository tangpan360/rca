import os
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))

def process_sn_metrics():
    # 1. 路径配置
    raw_data_dir = os.path.join(_project_root, "preprocess", "raw_data", "sn", "data")
    label_path = os.path.join(_project_root, "preprocess", "processed_data", "sn", "label_sn.csv")
    output_dir = os.path.join(_project_root, "preprocess", "processed_data", "sn", "metric")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 读取标签文件以获取服务列表
    label_df = pd.read_csv(label_path)
    
    # 3. 获取所有实验文件夹
    exp_folders = sorted([f for f in os.listdir(raw_data_dir) if f.startswith("SN.") and os.path.isdir(os.path.join(raw_data_dir, f))])
    
    all_services = sorted(label_df['service'].unique())
    print(f"正在处理 {len(all_services)} 个服务的指标数据...")
    
    # 4. 处理每个服务
    for service in tqdm(all_services):
        service_dfs = []
        
        # 遍历所有实验文件夹以收集该服务的指标数据
        for folder in exp_folders:
            csv_path = os.path.join(raw_data_dir, folder, "metrics", f"{service}.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # 确保时间戳为整数（秒）并加上8小时偏移
                df['timestamp'] = df['timestamp'].astype(int) + 8 * 3600
                service_dfs.append(df)
        
        if service_dfs:
            # 合并、按时间戳排序并去除重复项
            full_df = pd.concat(service_dfs).sort_values('timestamp').reset_index(drop=True)
            full_df = full_df.drop_duplicates(subset=['timestamp'])
            
            # 保存处理后的指标文件
            save_path = os.path.join(output_dir, f"{service}_metric.csv")
            full_df.to_csv(save_path, index=False)
        else:
            print(f"❌ 错误: 未找到服务 {service} 的指标文件")

if __name__ == "__main__":
    process_sn_metrics()
