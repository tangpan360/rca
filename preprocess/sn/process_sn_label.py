import json
import os
import glob
import pandas as pd
from datetime import datetime, timezone

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))


def _add_window_sample(fault_list, base_start_ts, offset, duration, service, fault_type, data_type):
    """辅助函数：添加一个滑动窗口样本"""
    # 时间戳加上8小时 (8 * 3600s)
    time_shift = 8 * 3600
    win_start_ts = base_start_ts + offset + time_shift
    win_end_ts = win_start_ts + duration
    
    st_time_str = datetime.fromtimestamp(win_start_ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
    ed_time_str = datetime.fromtimestamp(win_end_ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
    date_str = datetime.fromtimestamp(win_start_ts, timezone.utc).strftime('%Y-%m-%d')
    
    row = {
        'datetime': date_str,
        'service': service,
        'instance': service,
        'anomaly_type': fault_type,
        'st_time': st_time_str,
        'st_timestamp': int(win_start_ts),
        'ed_time': ed_time_str,
        'ed_timestamp': int(win_end_ts),
        'duration': duration,
        'data_type': data_type
    }
    fault_list.append(row)

def generate_sn_labels():
    input_dir = os.path.join(_project_root, "data", "raw_data", "sn", "data")
    output_dir = os.path.join(_project_root, "data", "processed_data", "sn")
    
    # 如果输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    label_files = sorted(glob.glob(os.path.join(input_dir, "SN.fault-*.json")))
    
    if not label_files:
        print(f"Error: No label files found in {input_dir}")
        return

    all_faults = []
    
    print(f"Found {len(label_files)} label files. Processing...")
    
    for fpath in label_files:
        with open(fpath, 'r') as f:
            data = json.load(f)
            
        for fault in data['faults']:
            # 提取原始数据
            raw_name = fault['name']
            fault_type = fault['fault']
            start_ts = fault['start']
            duration = fault['duration']
            end_ts = start_ts + duration
            
            # 映射服务名称: "socialnetwork-text-service-1" -> "text-service"
            # 逻辑: 移除 "socialnetwork-" 前缀和 "-1" 后缀
            # 注意潜在的变体，但基于分析它们是一致的
            service_name = raw_name.replace("socialnetwork-", "")
            if service_name.endswith("-1"):
                service_name = service_name[:-2]
            
            # 特殊映射: 将 Label 中的非标准名映射为 Metric/Trace 中的标准名
            service_mapping = {
                "nginx-thrift": "nginx-web-server"
            }
            if service_name in service_mapping:
                service_name = service_mapping[service_name]
            
            # 定义窗口参数
            window_size = 10
            stride = 2
            
            # 定义数据集划分比例 (Train: 50%, Val: 20%, Test: 30%)
            # 对应的截止时间点 (相对于 start_ts)
            # Train: [0, 60)
            # Val:   [60, 84)
            # Test:  [84, 120]
            
            # 确保区间不重叠的有效起始时间计算
            # Train Valid Start: [0, split_1 - window_size]
            # Val Valid Start:   [split_1, split_2 - window_size]
            # Test Valid Start:  [split_2, duration - window_size]
            
            split_1 = int(duration * 0.5)  # 60s
            split_2 = int(duration * 0.7)  # 84s
            
            # 生成滑动窗口样本
            # 1. Train
            for offset in range(0, split_1 - window_size + 1, stride):
                _add_window_sample(all_faults, start_ts, offset, window_size, service_name, fault_type, 'train')
                
            # 2. Val
            for offset in range(split_1, split_2 - window_size + 1, stride):
                _add_window_sample(all_faults, start_ts, offset, window_size, service_name, fault_type, 'val')
                
            # 3. Test
            for offset in range(split_2, duration - window_size + 1, stride):
                _add_window_sample(all_faults, start_ts, offset, window_size, service_name, fault_type, 'test')
            
    # 创建 DataFrame
    df = pd.DataFrame(all_faults)
    
    # 按开始时间 (st_time) 排序
    df = df.sort_values('st_time').reset_index(drop=True)
    
    # 添加 index 列 (从 0 开始)
    df['index'] = df.index
    
    # 调整列顺序，将 index 放在第一列
    cols = ['index'] + [c for c in df.columns if c != 'index']
    df = df[cols]
    
    # 保存为 CSV
    output_path = os.path.join(output_dir, "label_sn.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated {output_path}")
    print(f"Total sliding window samples: {len(df)}")
    print("\nSample distribution by type:")
    print(df['data_type'].value_counts())
    print("\nFirst 5 samples:")
    print(df[['service', 'anomaly_type', 'st_time', 'duration', 'data_type']].head().to_string(index=False))

if __name__ == "__main__":
    generate_sn_labels()
