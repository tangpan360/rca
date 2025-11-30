import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
import pickle

# 定义模块级私有变量
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)

from utils.template_utils import get_log_template_count

# 1. 全局配置
SERVICES = [
    'ts-assurance-service', 'ts-auth-service', 'ts-basic-service', 'ts-cancel-service', 
    'ts-config-service', 'ts-contacts-service', 'ts-food-map-service', 'ts-food-service', 
    'ts-inside-payment-service', 'ts-notification-service', 'ts-order-other-service', 
    'ts-order-service', 'ts-payment-service', 'ts-preserve-service', 'ts-price-service', 
    'ts-route-plan-service', 'ts-route-service', 'ts-seat-service', 'ts-security-service', 
    'ts-station-service', 'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-plan-service', 
    'ts-travel-service', 'ts-travel2-service', 'ts-user-service', 'ts-verification-code-service'
]

# Metric 列名 (从CSV确认)
METRIC_COLUMNS = [
    'cpu_usage_system', 'cpu_usage_total', 'cpu_usage_user', 
    'memory_usage', 'memory_working_set', 'rx_bytes', 'tx_bytes'
]

NUM_METRICS = len(METRIC_COLUMNS) # 7 (不包含timestamp)
NUM_INSTANCES = len(SERVICES)
NUM_TIME_STEPS = 10 # 10秒
STEP_DURATION = 1   # 1秒
# Log Template 数量 (动态获取)
NUM_LOG_TEMPLATES = get_log_template_count('tt')

# 全局缓存
METRIC_DATA_CACHE = {}
LOG_DATA_CACHE = {}
TRACE_DATA_CACHE = {}

# 归一化统计信息
NORMALIZATION_STATS = {
    'metric': None,
    'log': None,
    'trace': None
}

def preload_all_data():
    """
    预加载所有模态的数据到内存
    注意：TT数据的时间戳已经是秒(int/float)，不需要像Gaia那样除以10**6，也不需要乘1000
    """
    print("=" * 50)
    print("开始预加载所有 TT 数据到内存...")
    print("=" * 50)
    
    # 1. Metric
    print("\n[1/3] 加载 Metric 数据...")
    metric_dir = os.path.join(_project_root, 'preprocess', 'processed_data', 'tt', 'metric')
    for instance_name in tqdm(SERVICES, desc="Metric"):
        fpath = os.path.join(metric_dir, f"{instance_name}_metric.csv")
        if os.path.exists(fpath):
            try:
                # 读取所有列，处理潜在的列名空格
                df = pd.read_csv(fpath, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                METRIC_DATA_CACHE[instance_name] = df
            except Exception as e:
                print(f"❌ Error reading {instance_name} metric: {e}")
    
    # 2. Log
    print("\n[2/3] 加载 Log 数据...")
    log_dir = os.path.join(_project_root, 'preprocess', 'processed_data', 'tt', 'log')
    for instance_name in tqdm(SERVICES, desc="Log"):
        fpath = os.path.join(log_dir, f"{instance_name}_log.csv")
        if os.path.exists(fpath):
            try:
                # 读取所有列，处理列名空格
                df = pd.read_csv(fpath, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                
                if 'timestamp' in df.columns and 'template_id' in df.columns:
                    LOG_DATA_CACHE[instance_name] = df[['timestamp', 'template_id']]
                else:
                    print(f"⚠️  Warning: {instance_name} log file missing columns. Found: {df.columns.tolist()}")
            except Exception as e:
                print(f"❌ Error reading {instance_name} log: {e}")
            
    # 3. Trace
    print("\n[3/3] 加载 Trace 数据...")
    trace_dir = os.path.join(_project_root, 'preprocess', 'processed_data', 'tt', 'trace')
    for instance_name in tqdm(SERVICES, desc="Trace"):
        fpath = os.path.join(trace_dir, f"{instance_name}_trace.csv")
        if os.path.exists(fpath):
            try:
                # 读取所有列，处理列名空格
                df = pd.read_csv(fpath, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                
                required = ['start_time_ts', 'duration', 'status_code']
                if all(col in df.columns for col in required):
                    TRACE_DATA_CACHE[instance_name] = df[required]
                else:
                    print(f"⚠️  Warning: {instance_name} trace file missing columns. Found: {df.columns.tolist()}")
            except Exception as e:
                print(f"❌ Error reading {instance_name} trace: {e}")
            
    print(f"\n加载完成: Metric({len(METRIC_DATA_CACHE)}), Log({len(LOG_DATA_CACHE)}), Trace({len(TRACE_DATA_CACHE)})")

def compute_normalization_stats(label_df):
    """
    基于 Train 样本计算统计信息
    """
    train_df = label_df[label_df['data_type'] == 'train']
    print(f"\n从 {len(train_df)} 个训练样本计算统计信息...")
    
    all_metrics = [[] for _ in range(NUM_METRICS)]
    all_logs = [[] for _ in range(NUM_LOG_TEMPLATES)]
    all_traces = [[] for _ in range(NUM_INSTANCES)] # Trace Duration 按服务统计
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="收集训练数据"):
        # 时间戳处理 (秒)
        st_time = pd.to_datetime(row['st_time']).timestamp()
        # 10s 窗口
        ed_time = st_time + (NUM_TIME_STEPS * STEP_DURATION)
        
        # 获取原始数据 (normalize=False)
        metric, _ = _process_metric_for_sample(st_time, ed_time, normalize=False)
        log, _ = _process_log_for_sample(st_time, ed_time, normalize=False)
        trace, _ = _process_trace_for_sample(st_time, ed_time, normalize=False)
        
        # 收集 Metric (非NaN)
        for i in range(NUM_METRICS):
            vals = metric[:, :, i].flatten()
            all_metrics[i].extend(vals[~np.isnan(vals)])
            
        # 收集 Log (非0)
        for i in range(NUM_LOG_TEMPLATES):
            vals = log[:, i].flatten()
            all_logs[i].extend(vals[vals != 0])
            
        # 收集 Trace Duration (非NaN, 非0)
        for i in range(NUM_INSTANCES):
            vals = trace[i, :, 0] # Ch0: Duration
            all_traces[i].extend(vals[~np.isnan(vals) & (vals != 0)])
            
    # 计算 Mean/Std
    metric_stats = {'mean': np.zeros(NUM_METRICS), 'std': np.ones(NUM_METRICS)}
    for i in range(NUM_METRICS):
        if all_metrics[i]:
            metric_stats['mean'][i] = np.mean(all_metrics[i])
            metric_stats['std'][i] = np.std(all_metrics[i]) or 1.0
            
    log_stats = {'mean': np.zeros(NUM_LOG_TEMPLATES), 'std': np.ones(NUM_LOG_TEMPLATES)}
    for i in range(NUM_LOG_TEMPLATES):
        if all_logs[i]:
            log_stats['mean'][i] = np.mean(all_logs[i])
            log_stats['std'][i] = np.std(all_logs[i]) or 1.0
            
    trace_stats = []
    for i in range(NUM_INSTANCES):
        mean, std = 0.0, 1.0
        if all_traces[i]:
            mean = np.mean(all_traces[i])
            std = np.std(all_traces[i]) or 1.0
        trace_stats.append({'mean': mean, 'std': std})
        
    print("统计计算完成。")
    return {'metric': metric_stats, 'log': log_stats, 'trace': trace_stats}

def _process_metric_for_sample(st_time, ed_time, normalize=True):
    """
    处理单个样本的指标数据（使用预加载的缓存）
    
    Args:
        st_time: 故障开始时间戳（秒）
        ed_time: 故障结束时间戳（秒）
        normalize: 是否进行归一化，默认True
    
    Returns:
        tuple: (metric_data, availability)
            - metric_data: numpy array, shape [12, 10, 7]
            - availability: bool - 整个metric模态是否可用
    """
    metric_data = np.full((NUM_INSTANCES, NUM_TIME_STEPS, NUM_METRICS), np.nan)
    
    for instance_idx, instance_name in enumerate(SERVICES):
        if instance_name not in METRIC_DATA_CACHE:
            continue
            
        df = METRIC_DATA_CACHE[instance_name]
        # 筛选时间 [st, ed)
        mask = (df['timestamp'] >= st_time) & (df['timestamp'] < ed_time)
        sample = df[mask].sort_values('timestamp')
        
        if not sample.empty:
            # 对齐到 1s 时间步
            # sample['timestamp'] 应该是整数秒
            # 计算相对索引
            rel_idx = (sample['timestamp'] - st_time).astype(int)
            valid_mask = (rel_idx >= 0) & (rel_idx < NUM_TIME_STEPS)
            
            valid_sample = sample[valid_mask]
            valid_idx = rel_idx[valid_mask]
            
            # 填入数据
            metric_data[instance_idx, valid_idx, :] = valid_sample[METRIC_COLUMNS].values
            
    # 线性插值/前向填充 (在时间轴上)
    # 这里简单起见，如果 normalize=True，我们用 Mean 填充 NaN
    
    availability = not np.all(np.isnan(metric_data))
    
    if normalize and NORMALIZATION_STATS['metric']:
        stats = NORMALIZATION_STATS['metric']
        for metric_idx in range(NUM_METRICS):
            nan_mask = np.isnan(metric_data[:, :, metric_idx])
            if nan_mask.any():
                metric_data[:, :, metric_idx][nan_mask] = stats['mean'][metric_idx]
            metric_data[:, :, metric_idx] = (metric_data[:, :, metric_idx] - stats['mean'][metric_idx]) / stats['std'][metric_idx]
            
    return metric_data, availability

def _process_log_for_sample(st_time, ed_time, normalize=True):
    """
    处理单个样本的log数据（使用预加载的缓存）
    
    Args:
        st_time: 故障开始时间戳（秒）
        ed_time: 故障结束时间戳（秒）
        normalize: 是否进行归一化，默认True
    
    Returns:
        tuple: (log_data, availability)
            - log_data: numpy array, shape [12, 13]
            - availability: bool - 整个log模态是否可用
    """
    log_data = np.zeros((NUM_INSTANCES, NUM_LOG_TEMPLATES))
    
    for instance_idx, instance_name in enumerate(SERVICES):
        if instance_name not in LOG_DATA_CACHE:
            continue
            
        df = LOG_DATA_CACHE[instance_name]
        # SN Log timestamp 是 float seconds
        mask = (df['timestamp'] >= st_time) & (df['timestamp'] < ed_time)
        sample_data = df[mask]
        
        if not sample_data.empty:
            template_counts = sample_data['template_id'].value_counts()
            for template_id, count in template_counts.items():
                # template_id 从 1 开始，数组索引从 0 开始
                if 1 <= template_id <= NUM_LOG_TEMPLATES:
                    log_data[instance_idx, template_id - 1] = count
                    
    availability = not np.all(log_data == 0)
    
    if normalize and NORMALIZATION_STATS['log']:
        stats = NORMALIZATION_STATS['log']
        log_data = (log_data - stats['mean']) / stats['std']
        
    return log_data, availability

def _process_trace_for_sample(st_time, ed_time, normalize=True):
    """
    处理单个样本的trace数据（使用预加载的缓存）
    
    双通道特征提取：
    - Channel 0: Duration (响应时间)
    - Channel 1: Error Rate (错误率, based on status_code >= 400)
    
    Args:
        st_time: 故障开始时间戳（秒）
        ed_time: 故障结束时间戳（秒）
        normalize: 是否进行归一化，默认True
    
    Returns:
        tuple: (trace_data, availability)
            - trace_data: numpy array, shape [12, 10, 2]
            - availability: bool - 整个trace模态是否可用
    """
    trace_data = np.full((NUM_INSTANCES, NUM_TIME_STEPS, 2), np.nan)
    
    for instance_idx, instance_name in enumerate(SERVICES):
        if instance_name not in TRACE_DATA_CACHE:
            continue
            
        df = TRACE_DATA_CACHE[instance_name]
        # 筛选
        mask = (df['start_time_ts'] >= st_time) & (df['start_time_ts'] < ed_time)
        sample_data = df[mask]
        
        if not sample_data.empty:
            timestamps = sample_data['start_time_ts'].values
            durations = sample_data['duration'].values
            status_codes = sample_data['status_code'].values
            
            # 计算时间步索引 (0-9)
            segment_indices = ((timestamps - st_time) // STEP_DURATION).astype(int)
            
            # 聚合
            for seg_idx in range(NUM_TIME_STEPS):
                seg_mask = segment_indices == seg_idx
                if seg_mask.any():
                    # Duration Mean
                    trace_data[instance_idx, seg_idx, 0] = np.mean(durations[seg_mask])
                    # Error Rate
                    seg_status = status_codes[seg_mask]
                    error_count = np.sum(seg_status >= 400)
                    error_rate = error_count / len(seg_status)
                    trace_data[instance_idx, seg_idx, 1] = error_rate
                    
    availability = not np.all(np.isnan(trace_data))
    
    if normalize and NORMALIZATION_STATS['trace']:
        # 对所有服务进行处理
        for instance_idx in range(NUM_INSTANCES):
            instance_name = SERVICES[instance_idx]
            
            if instance_name in TRACE_DATA_CACHE:
                # 有 trace 数据的服务：正常归一化
                stats = NORMALIZATION_STATS['trace'][instance_idx]  # 使用服务索引，不是trace索引
                
                # Duration: Fill Mean -> Normalize
                nan_mask_0 = np.isnan(trace_data[instance_idx, :, 0])
                if nan_mask_0.any():
                    trace_data[instance_idx, :, 0][nan_mask_0] = stats['mean']
                trace_data[instance_idx, :, 0] = (trace_data[instance_idx, :, 0] - stats['mean']) / stats['std']
                
                # ErrorRate: Fill 0 -> No Normalize
                nan_mask_1 = np.isnan(trace_data[instance_idx, :, 1])
                if nan_mask_1.any():
                    trace_data[instance_idx, :, 1][nan_mask_1] = 0.0
            else:
                # 没有 trace 数据的服务：填充默认值（表示无调用活动）
                trace_data[instance_idx, :, 0] = 0.0  # Duration 填充0（表示无调用）
                trace_data[instance_idx, :, 1] = 0.0  # Error Rate 填充0（表示无错误）
                
    return trace_data, availability

def _process_single_sample(row) -> Dict[str, Any]:
    """
    处理单个故障样本
    """
    sample_id = row['index']
    fault_service = row['instance']
    fault_type = row['anomaly_type']
    st_time = pd.to_datetime(row['st_time']).timestamp()
    ed_time = st_time + (NUM_TIME_STEPS * STEP_DURATION)
    data_type = row['data_type']

    processed_sample = {
        'sample_id': sample_id,
        'fault_service': fault_service,
        'fault_type': fault_type,
        'st_time': st_time,
        'ed_time': ed_time,
        'data_type': data_type,
    }

    # 处理各模态数据（返回数据和可用性标记）
    metric_data, metric_available = _process_metric_for_sample(st_time, ed_time)
    log_data, log_available = _process_log_for_sample(st_time, ed_time)
    trace_data, trace_available = _process_trace_for_sample(st_time, ed_time)
    
    processed_sample['metric_data'] = metric_data
    processed_sample['log_data'] = log_data
    processed_sample['trace_data'] = trace_data
    
    # 添加可用性标记（整个模态级别）
    processed_sample['metric_available'] = metric_available  # bool
    processed_sample['log_available'] = log_available        # bool
    processed_sample['trace_available'] = trace_available    # bool
    
    return processed_sample


def process_all_sample(label_df) -> Dict[int, Dict[str, Any]]:
    """
    处理所有故障样本
    """
    processed_data = {}
    
    print(f"\n开始处理 {len(label_df)} 个故障样本...")
    
    for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Processing samples"):
        sample_id = row['index']
        try:
            processed_sample = _process_single_sample(row)
            processed_data[sample_id] = processed_sample
        except Exception as e:
            print(f"\n❌ Failed to process sample {sample_id}: {e}")
            continue
    
    print(f"\n✅ 完成！成功处理 {len(processed_data)}/{len(label_df)} 个样本")
    return processed_data

if __name__ == "__main__":
    label_path = os.path.join(_project_root, "preprocess", "processed_data", "tt", "label_tt.csv")
    label_df = pd.read_csv(label_path)
    
    # 1. Preload
    preload_all_data()
    
    # 2. Stats
    stats_file = os.path.join(_project_root, "preprocess", "processed_data", "tt", "tt_norm_stats.pkl")
    if os.path.exists(stats_file):
        print(f"Loading stats from {stats_file}")
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
    else:
        stats = compute_normalization_stats(label_df)
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
            
    NORMALIZATION_STATS['metric'] = stats['metric']
    NORMALIZATION_STATS['log'] = stats['log']
    NORMALIZATION_STATS['trace'] = stats['trace']
    
    # 3. Process
    dataset = process_all_sample(label_df)
    
    # 4. Save
    out_path = os.path.join(_project_root, "preprocess", "processed_data", "tt", "dataset.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"\nDataset saved to {out_path}")
    print(f"Total samples: {len(dataset)}")
