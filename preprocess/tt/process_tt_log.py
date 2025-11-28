import os
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
from datetime import datetime

# 添加 extractor/drain 到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
extractor_path = os.path.join(project_root, 'extractor')
sys.path.append(extractor_path)

# Import 统一的 Drain 模块
from drain.drain_template_extractor import extract_templates

def parse_sn_log_timestamp(log_str):
    """
    解析 SN 日志时间戳
    格式: [2022-Apr-17 10:12:50.490796]
    """
    pattern = r"\[(\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]"
    match = re.search(pattern, log_str)
    if not match:
        return None
    
    time_str = match.group(1)
    try:
        # 解析时间字符串 (包含英文月份)
        dt = pd.to_datetime(time_str, format="%Y-%b-%d %H:%M:%S.%f", utc=True)
        ts = dt.timestamp()
        return ts
    except Exception as e:
        return None

def process_sn_logs():
    print("=== 开始处理 SN 日志数据 (Custom Drain Config) ===")
    
    # 1. 配置路径
    raw_data_dir = os.path.join(project_root, "preprocess", "raw_data", "sn", "data")
    label_path = os.path.join(project_root, "preprocess", "processed_data", "sn", "label_sn.csv")
    output_dir = os.path.join(project_root, "preprocess", "processed_data", "sn", "log")
    drain_model_dir = os.path.join(project_root, "preprocess", "processed_data", "sn", "drain_models")
    
    # 指定自定义 Drain 配置文件
    drain_config_path = os.path.join(script_dir, "sn_drain3.ini")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(drain_model_dir, exist_ok=True)
    
    # 2. 加载标签以识别训练集时间段
    print("加载标签数据...")
    label_df = pd.read_csv(label_path)
    
    # 识别训练集时间范围
    train_samples = label_df[label_df['data_type'] == 'train']
    train_intervals = []
    for _, row in train_samples.iterrows():
        st = pd.to_datetime(row['st_time']).timestamp()
        ed = st + row['duration']
        train_intervals.append((st, ed))
    
    # 3. 收集所有日志
    exp_folders = sorted([f for f in os.listdir(raw_data_dir) if f.startswith("SN.") and os.path.isdir(os.path.join(raw_data_dir, f))])
    all_logs = [] 
    
    print(f"正在解析 {len(exp_folders)} 个实验文件夹中的日志...")
    for folder in tqdm(exp_folders, desc="解析日志文件"):
        log_json_path = os.path.join(raw_data_dir, folder, "logs.json")
        if not os.path.exists(log_json_path):
            continue
            
        with open(log_json_path, 'r') as f:
            try:
                logs_dict = json.load(f)
            except json.JSONDecodeError:
                continue
            
        for raw_service, log_list in logs_dict.items():
            service = raw_service
            if service.startswith("socialnetwork-"):
                service = service.replace("socialnetwork-", "")
            if service.endswith("-1"):
                service = service[:-2]
                
            for log_msg in log_list:
                ts = parse_sn_log_timestamp(log_msg)
                if ts is not None:
                    all_logs.append({
                        'timestamp': ts,
                        'service': service,
                        'message': log_msg
                    })
    
    print(f"共收集到 {len(all_logs)} 条有效日志。")
    logs_df = pd.DataFrame(all_logs)
    
    # 4. 筛选训练集日志
    print("筛选训练集日志...")
    logs_df = logs_df.sort_values('timestamp')
    train_intervals.sort()
    merged_intervals = []
    if train_intervals:
        curr_start, curr_end = train_intervals[0]
        for next_start, next_end in train_intervals[1:]:
            if next_start <= curr_end:
                curr_end = max(curr_end, next_end)
            else:
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_intervals.append((curr_start, curr_end))
    
    is_train = np.zeros(len(logs_df), dtype=bool)
    timestamps = logs_df['timestamp'].values
    for start, end in merged_intervals:
        idx_start = np.searchsorted(timestamps, start, side='left')
        idx_end = np.searchsorted(timestamps, end, side='right')
        is_train[idx_start:idx_end] = True
        
    train_logs_df = logs_df[is_train]
    train_messages = train_logs_df['message'].tolist()
    print(f"筛选出 {len(train_messages)} 条训练集日志用于 Drain 训练")
    
    if not train_messages:
        print("⚠️  Error: No training logs found. Check timestamp alignment.")
        return

    # 5. 训练 Drain (使用自定义配置)
    drain_model_path = os.path.join(drain_model_dir, "sn_drain.pkl")
    
    miner = extract_templates(
        log_list=train_messages,
        save_pth=drain_model_path,
        config_path=drain_config_path
    )
    
    # 保存模板统计
    template_csv_path = os.path.join(drain_model_dir, "sn_templates.csv")
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)
    template_data = {
        'template_id': [c.cluster_id for c in sorted_clusters],
        'template': [c.get_template() for c in sorted_clusters],
        'count': [c.size for c in sorted_clusters]
    }
    pd.DataFrame(template_data).to_csv(template_csv_path, index=False)
    
    # 6. 匹配所有日志
    print("正在匹配模板...")
    all_templates = []
    all_template_ids = []
    
    for msg in tqdm(logs_df['message'], desc="Matching"):
        match = miner.match(msg)
        if match:
            all_templates.append(match.get_template())
            all_template_ids.append(match.cluster_id)
        else:
            all_templates.append("Unseen")
            all_template_ids.append(-1)
            
    logs_df['template'] = all_templates
    logs_df['template_id'] = all_template_ids
    
    # 7. 保存
    unique_services = logs_df['service'].unique()
    for service in tqdm(unique_services, desc="Saving CSVs"):
        service_logs = logs_df[logs_df['service'] == service].copy().sort_values('timestamp')
        save_path = os.path.join(output_dir, f"{service}_log.csv")
        service_logs.to_csv(save_path, index=False)
        
    print("=== 完成 ===")

if __name__ == "__main__":
    process_sn_logs()
