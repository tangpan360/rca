import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

# 全局定义：所有instance的固定顺序（用于所有模态）
SERVICES = ['dbservice1', 'dbservice2', 'logservice1', 'logservice2', 
            'mobservice1', 'mobservice2', 'redisservice1', 'redisservice2', 
            'webservice1', 'webservice2']

# 全局缓存：预加载的数据
METRIC_DATA_CACHE = {}
LOG_DATA_CACHE = {}
TRACE_DATA_CACHE = {}

# 全局归一化统计信息（从训练集计算）
NORMALIZATION_STATS = {
    'metric': None,  # {'mean': [12], 'std': [12]}
    'log': None,     # {'mean': [40], 'std': [40]}
    'trace': None    # [{'mean': float, 'std': float}] * 10
}


def preload_all_data():
    """
    预加载所有模态的数据到内存
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 50)
    print("开始预加载所有数据到内存...")
    print("=" * 50)
    
    # 1. 预加载 Metric 数据
    print("\n[1/3] 加载 Metric 数据...")
    metric_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'metric')
    for instance_name in tqdm(SERVICES, desc="Metric"):
        metric_file = os.path.join(metric_data_dir, f"{instance_name}_metric.csv")
        if os.path.exists(metric_file):
            df = pd.read_csv(metric_file)
            METRIC_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} 行数据")
    
    # 2. 预加载 Log 数据
    print("\n[2/3] 加载 Log 数据...")
    log_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'log')
    for instance_name in tqdm(SERVICES, desc="Log"):
        log_file = os.path.join(log_data_dir, f"{instance_name}_log.csv")
        if os.path.exists(log_file):
            # 只读取需要的列以节省内存
            df = pd.read_csv(log_file, usecols=['timestamp_ts', 'template_id'])
            LOG_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} 行数据")
    
    # 3. 预加载 Trace 数据
    print("\n[3/3] 加载 Trace 数据...")
    trace_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'trace')
    for instance_name in tqdm(SERVICES, desc="Trace"):
        trace_file = os.path.join(trace_data_dir, f"{instance_name}_trace.csv")
        if os.path.exists(trace_file):
            # 只读取需要的列以节省内存
            df = pd.read_csv(trace_file, usecols=['start_time_ts', 'duration'])
            TRACE_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} 行数据")
    
    print("\n" + "=" * 50)
    print("数据预加载完成！")
    print(f"  Metric: {len(METRIC_DATA_CACHE)} 个实例")
    print(f"  Log: {len(LOG_DATA_CACHE)} 个实例")
    print(f"  Trace: {len(TRACE_DATA_CACHE)} 个实例")
    print("=" * 50 + "\n")


def load_anomaly_periods(label_file_path):
    """
    加载异常时间段数据（固定600秒窗口）
    
    Args:
        label_file_path (str): 标签文件路径
        
    Returns:
        list: 异常时间段列表，每个元素为(start_timestamp, end_timestamp, data_type)的三元组
    """
    print("正在加载异常时间段数据...")
    
    # 读取标签文件
    label_df = pd.read_csv(label_file_path)
    
    # 转换时间格式为时间戳，异常时间段为开始时间往后600秒
    anomaly_periods = []
    for _, row in label_df.iterrows():
        # 将开始时间字符串转换为时间戳（毫秒）
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000  # 开始时间 + 600秒
        data_type = row.get('data_type', 'unknown')  # 获取data_type，默认为unknown
        anomaly_periods.append((st_time, ed_time, data_type))
    
    print(f"共加载 {len(anomaly_periods)} 个异常时间段")
    return anomaly_periods


def compute_normalization_stats(label_df):
    """
    从训练集计算归一化统计信息
    """
    train_df = label_df[label_df['data_type'] == 'train']
    print(f"\n从 {len(train_df)} 个训练样本计算归一化统计...")
    
    all_metrics, all_logs, all_traces = [], [], [[] for _ in range(10)]
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="收集训练数据"):
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000
        
        # 收集原始数据
        metric = _process_metric_for_sample(st_time, ed_time)
        log = _process_log_for_sample(st_time, ed_time)
        trace = _process_trace_for_sample(st_time, ed_time)
        
        all_metrics.append(metric.reshape(-1, 12))
        all_logs.append(log.reshape(-1, 40))
        
        for i in range(10):
            vals = trace[i, :, 0]
            all_traces[i].extend(vals[vals != 0])
    
    # 计算统计信息
    all_metrics = np.vstack(all_metrics)
    all_logs = np.vstack(all_logs)
    
    metric_stats = {
        'mean': np.mean(all_metrics, axis=0),
        'std': np.std(all_metrics, axis=0)
    }
    metric_stats['std'][metric_stats['std'] == 0] = 1.0
    
    log_stats = {
        'mean': np.mean(all_logs, axis=0),
        'std': np.std(all_logs, axis=0)
    }
    log_stats['std'][log_stats['std'] == 0] = 1.0
    
    trace_stats = []
    for i in range(10):
        if len(all_traces[i]) > 0:
            mean, std = np.mean(all_traces[i]), np.std(all_traces[i])
            trace_stats.append({'mean': mean, 'std': std if std > 0 else 1.0})
        else:
            trace_stats.append({'mean': 0.0, 'std': 1.0})
    
    print("✅ 统计信息计算完成")
    return {'metric': metric_stats, 'log': log_stats, 'trace': trace_stats}


def _process_metric_for_sample(st_time, ed_time) -> np.ndarray:
    """
    处理单个样本的指标数据（使用预加载的缓存）
    
    Args:
        st_time: 故障开始时间戳（毫秒）
        ed_time: 故障结束时间戳（毫秒）
    
    Returns:
        numpy array, shape [10, 20, 12] - 10个instance，20个时间步，12个指标
    """    
    # 使用全局定义的服务顺序
    num_instances = len(SERVICES)
    
    # 初始化结果数组 [num_instances, 20 time_steps, 12 metrics]
    metric_data = np.zeros((num_instances, 20, 12))
    metric_names = None
    
    # 按照固定顺序遍历每个instance
    for instance_idx, instance_name in enumerate(SERVICES):
        # 从缓存中读取数据
        if instance_name not in METRIC_DATA_CACHE:
            continue
        
        try:
            df = METRIC_DATA_CACHE[instance_name]
            mask = (df['timestamp'] >= st_time) & (df['timestamp'] <= ed_time)
            sample_data = df[mask].sort_values('timestamp')
            
            if metric_names is None:
                metric_names = [col for col in sample_data.columns if col != 'timestamp']
            
            # 一次性赋值所有指标数据
            num_time_steps = min(len(sample_data), 20)
            metric_data[instance_idx, :num_time_steps, :] = sample_data[metric_names].values[:num_time_steps]
        
        except Exception:
            continue
    
    # 将NaN值替换为0
    metric_data = np.nan_to_num(metric_data, nan=0.0)
    
    # 归一化
    if NORMALIZATION_STATS['metric'] is not None:
        stats = NORMALIZATION_STATS['metric']
        metric_data = (metric_data - stats['mean']) / stats['std']
    
    return metric_data

def _process_log_for_sample(st_time, ed_time) -> np.ndarray:
    """
    处理单个样本的log数据（使用预加载的缓存）
    
    Args:
        st_time: 故障开始时间戳（毫秒）
        ed_time: 故障结束时间戳（毫秒）
    
    Returns:
        numpy array, shape [10, 40] - 10个instance，每个40维template统计
    """
    # 使用全局定义的服务顺序
    num_instances = len(SERVICES)
    num_templates = 40  # 固定40个template
    
    # 初始化结果数组 [num_instances, num_templates]
    log_data = np.zeros((num_instances, num_templates))
    
    # 按照固定顺序遍历每个instance
    for instance_idx, instance_name in enumerate(SERVICES):
        # 从缓存中读取数据
        if instance_name not in LOG_DATA_CACHE:
            continue
        
        try:
            df = LOG_DATA_CACHE[instance_name]
            
            # 筛选时间范围内的数据
            mask = (df['timestamp_ts'] >= st_time) & (df['timestamp_ts'] <= ed_time)
            sample_data = df[mask]
            
            if len(sample_data) > 0:
                # 统计每个template_id出现的次数
                template_counts = sample_data['template_id'].value_counts()
                
                # 将统计结果填入对应位置（template_id从1开始，数组索引从0开始）
                for template_id, count in template_counts.items():
                    if 1 <= template_id <= num_templates:
                        log_data[instance_idx, template_id - 1] = count
        
        except Exception:
            continue
    
    # 归一化
    if NORMALIZATION_STATS['log'] is not None:
        stats = NORMALIZATION_STATS['log']
        log_data = (log_data - stats['mean']) / stats['std']
    
    return log_data


def _process_trace_for_sample(st_time, ed_time) -> np.ndarray:
    """
    处理单个样本的trace数据（使用预加载的缓存）
    
    Args:
        st_time: 故障开始时间戳（毫秒）
        ed_time: 故障结束时间戳（毫秒）
    
    Returns:
        np.ndarray: shape [10, 20, 1] - 10个instance，20个时间段，每个时间段的平均duration
                    如果某个时间段没有数据，则为0.0
    """
    # 使用全局定义的服务顺序
    num_instances = len(SERVICES)
    num_time_segments = 20  # 20个时间段
    segment_duration = 30 * 1000  # 每个时间段30秒（毫秒）
    
    # 初始化结果数组 [num_instances, num_time_segments, 1]，默认值为0.0
    trace_data = np.zeros((num_instances, num_time_segments, 1))
    
    # 按照固定顺序遍历每个instance
    for instance_idx, instance_name in enumerate(SERVICES):
        # 从缓存中读取数据
        if instance_name not in TRACE_DATA_CACHE:
            continue
        
        try:
            df = TRACE_DATA_CACHE[instance_name]
            
            # 筛选时间范围内的数据
            mask = (df['start_time_ts'] >= st_time) & (df['start_time_ts'] <= ed_time)
            sample_data = df[mask]
            
            if len(sample_data) > 0:
                # 向量化计算：批量计算所有trace的时间段索引
                timestamps = sample_data['start_time_ts'].values
                durations = sample_data['duration'].values
                
                # 批量计算时间偏移和段索引
                time_offsets = timestamps - st_time
                segment_indices = (time_offsets // segment_duration).astype(int)
                
                # 筛选有效的段索引
                valid_mask = (segment_indices >= 0) & (segment_indices < num_time_segments)
                valid_segments = segment_indices[valid_mask]
                valid_durations = durations[valid_mask]
                
                # 按段索引分组计算平均值
                for seg_idx in range(num_time_segments):
                    seg_mask = valid_segments == seg_idx
                    if seg_mask.any():
                        trace_data[instance_idx, seg_idx, 0] = valid_durations[seg_mask].mean()
        
        except Exception:
            continue
    
    # 归一化（按instance）
    if NORMALIZATION_STATS['trace'] is not None:
        for i in range(num_instances):
            stats = NORMALIZATION_STATS['trace'][i]
            trace_data[i, :, 0] = (trace_data[i, :, 0] - stats['mean']) / stats['std']
    
    return trace_data


def _process_single_sample(row) -> Dict[str, Any]:
    """
    处理单个故障样本
    """
    sample_id = row['index']
    fault_service = row['instance']
    fault_type = row['anomaly_type']
    st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
    ed_time = st_time + 600 * 1000  # 开始时间 + 600秒
    data_type = row['data_type']

    processed_sample = {
        'sample_id': sample_id,
        'fault_service': fault_service,
        'fault_type': fault_type,
        'st_time': st_time,
        'ed_time': ed_time,
        'data_type': data_type,
    }

    # 处理各模态数据
    processed_sample['metric_data'] = _process_metric_for_sample(st_time, ed_time)
    processed_sample['log_data'] = _process_log_for_sample(st_time, ed_time)
    processed_sample['trace_data'] = _process_trace_for_sample(st_time, ed_time)
    
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
    import pickle
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_file = os.path.join(project_dir, "extractor", "MicroSS", "label.csv")
    label_df = pd.read_csv(label_file)
    
    # 1. 预加载所有数据到内存
    preload_all_data()
    
    # 2. 计算或加载归一化统计信息
    stats_file = os.path.join(project_dir, "preprocess", "processed_data", "norm_stats.pkl")
    
    if os.path.exists(stats_file):
        print(f"\n📂 加载归一化统计: {stats_file}")
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
    else:
        print("\n🔄 计算归一化统计...")
        stats = compute_normalization_stats(label_df)
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        print(f"✅ 统计信息已保存: {stats_file}")
    
    # 设置全局统计信息
    NORMALIZATION_STATS['metric'] = stats['metric']
    NORMALIZATION_STATS['log'] = stats['log']
    NORMALIZATION_STATS['trace'] = stats['trace']
    
    # 3. 处理所有样本
    processed_data = process_all_sample(label_df)
    
    # 4. 保存处理后的数据
    output_file = os.path.join(project_dir, "preprocess", "processed_data", "dataset.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"\n💾 数据集已保存: {output_file}")
    print(f"   - 样本数: {len(processed_data)}")
    print(f"   - 已归一化: Metric, Log, Trace")