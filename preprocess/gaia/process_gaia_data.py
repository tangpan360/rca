import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

# 添加项目根目录到路径，以便导入utils
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)

from utils.template_utils import get_log_template_count

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
    'log': None,     # {'mean': [48], 'std': [48]}
    'trace': None    # [{'mean': float, 'std': float}] * 10 (for duration only)
}


def preload_all_data():
    """
    预加载所有模态的数据到内存
    """
    print("=" * 50)
    print("开始预加载所有数据到内存...")
    print("=" * 50)
    
    # 1. 预加载 Metric 数据
    print("\n[1/3] 加载 Metric 数据...")
    metric_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    for instance_name in tqdm(SERVICES, desc="Metric"):
        metric_file = os.path.join(metric_data_dir, f"{instance_name}_metric.csv")
        if os.path.exists(metric_file):
            df = pd.read_csv(metric_file)
            METRIC_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} 行数据")
    
    # 2. 预加载 Log 数据
    print("\n[2/3] 加载 Log 数据...")
    log_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'log')
    for instance_name in tqdm(SERVICES, desc="Log"):
        log_file = os.path.join(log_data_dir, f"{instance_name}_log.csv")
        if os.path.exists(log_file):
            # 只读取需要的列以节省内存
            df = pd.read_csv(log_file, usecols=['timestamp_ts', 'template_id'])
            LOG_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} 行数据")
    
    # 3. 预加载 Trace 数据
    print("\n[3/3] 加载 Trace 数据 (包含status_code)...")
    trace_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'trace')
    for instance_name in tqdm(SERVICES, desc="Trace"):
        trace_file = os.path.join(trace_data_dir, f"{instance_name}_trace.csv")
        if os.path.exists(trace_file):
            # 读取 duration 和 status_code
            df = pd.read_csv(trace_file, usecols=['start_time_ts', 'duration', 'status_code'])
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
    
    统计方式:
        - Metric: 排除NaN，包含0（0是真实值）
        - Log: 排除0（0是真实的'未出现'）
        - Trace: 排除NaN和0（都是缺失），仅对Duration统计
    """
    train_df = label_df[label_df['data_type'] == 'train']
    print(f"\n从 {len(train_df)} 个训练样本计算归一化统计...")
    
    # 按指标分别收集
    all_metrics = [[] for _ in range(12)]  # 12个metric指标
    all_logs = [[] for _ in range(48)]     # 48个log模板
    all_traces = [[] for _ in range(10)]   # 10个instance
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="收集训练数据"):
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000
        
        # 收集原始数据（不归一化）
        metric, _ = _process_metric_for_sample(st_time, ed_time, normalize=False)
        log, _ = _process_log_for_sample(st_time, ed_time, normalize=False)
        trace, _ = _process_trace_for_sample(st_time, ed_time, normalize=False)
        
        # Metric: 按指标收集非NaN值（包含0）
        for i in range(12):
            vals = metric[:, :, i].flatten()
            valid_vals = vals[~np.isnan(vals)]  # 排除NaN，保留0
            all_metrics[i].extend(valid_vals)
        
        # Log: 按模板收集非0值
        for i in range(48):
            vals = log[:, i].flatten()
            non_zero_vals = vals[vals != 0]  # 排除0
            all_logs[i].extend(non_zero_vals)
        
        # Trace: 按instance收集非NaN且非0的值 (只收集通道0: Duration)
        for i in range(10):
            vals = trace[i, :, 0]  # Channel 0: Duration
            valid_vals = vals[~np.isnan(vals) & (vals != 0)]  # 排除NaN和0
            all_traces[i].extend(valid_vals)
    
    # 计算每个指标的均值和标准差
    print("\n计算统计信息:")
    
    # Metric统计
    metric_means = np.zeros(12)
    metric_stds = np.zeros(12)
    for i in range(12):
        if len(all_metrics[i]) > 0:
            metric_means[i] = np.mean(all_metrics[i])
            metric_stds[i] = np.std(all_metrics[i])
            if metric_stds[i] == 0:
                metric_stds[i] = 1.0
            print(f"  Metric[{i}]: mean={metric_means[i]:.4f}, std={metric_stds[i]:.4f}, samples={len(all_metrics[i])}")
        else:
            metric_means[i] = 0.0
            metric_stds[i] = 1.0
            print(f"  Metric[{i}]: 无有效数据")
    
    # Log统计
    log_means = np.zeros(48)
    log_stds = np.zeros(48)
    for i in range(48):
        if len(all_logs[i]) > 0:
            log_means[i] = np.mean(all_logs[i])
            log_stds[i] = np.std(all_logs[i])
            if log_stds[i] == 0:
                log_stds[i] = 1.0
        else:
            log_means[i] = 0.0
            log_stds[i] = 1.0
    print(f"  Log: {np.sum([len(all_logs[i]) > 0 for i in range(48)])}/48 个模板有数据")
    
    # Trace统计 (只对Duration)
    trace_stats = []
    for i in range(10):
        if len(all_traces[i]) > 0:
            mean, std = np.mean(all_traces[i]), np.std(all_traces[i])
            trace_stats.append({'mean': mean, 'std': std if std > 0 else 1.0})
            print(f"  Trace[{SERVICES[i]}]: mean={mean:.4f}, std={std:.4f}, samples={len(all_traces[i])}")
        else:
            trace_stats.append({'mean': 0.0, 'std': 1.0})
            print(f"  Trace[{SERVICES[i]}]: 无有效数据")
    
    metric_stats = {'mean': metric_means, 'std': metric_stds}
    log_stats = {'mean': log_means, 'std': log_stds}
    
    print("\n✅ 统计信息计算完成")
    return {'metric': metric_stats, 'log': log_stats, 'trace': trace_stats}


def _process_metric_for_sample(st_time, ed_time, normalize=True):
    """
    处理单个样本的指标数据（使用预加载的缓存）
    
    Args:
        st_time: 故障开始时间戳（毫秒）
        ed_time: 故障结束时间戳（毫秒）
        normalize: 是否进行归一化，默认True
    
    Returns:
        tuple: (metric_data, availability)
            - metric_data: numpy array, shape [10, 20, 12]
            - availability: bool - 整个metric模态是否可用
    """    
    # 使用全局定义的服务顺序
    num_instances = len(SERVICES)
    
    # 初始化结果数组 [num_instances, 20 time_steps, 12 metrics]
    # 初始化为NaN以便后续识别缺失
    metric_data = np.full((num_instances, 20, 12), np.nan)
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
            if num_time_steps > 0:
                metric_data[instance_idx, :num_time_steps, :] = sample_data[metric_names].values[:num_time_steps]
        
        except Exception:
            continue
    
    # 计算整个模态的可用性（如果所有数据都是NaN，则整个模态不可用）
    availability = not np.all(np.isnan(metric_data))
    
    # 处理和归一化
    if normalize and NORMALIZATION_STATS['metric'] is not None:
        stats = NORMALIZATION_STATS['metric']
        
        # 用均值填充NaN
        for i in range(12):  # 12个指标
            nan_mask = np.isnan(metric_data[:, :, i])
            if nan_mask.any():
                metric_data[:, :, i][nan_mask] = stats['mean'][i]
        
        # 归一化
        metric_data = (metric_data - stats['mean']) / stats['std']
    else:
        # 如果不归一化（统计阶段），将NaN替换为NaN保持原样
        pass
    
    return metric_data, availability

def _process_log_for_sample(st_time, ed_time, normalize=True):
    """
    处理单个样本的log数据（使用预加载的缓存）
    
    Args:
        st_time: 故障开始时间戳（毫秒）
        ed_time: 故障结束时间戳（毫秒）
        normalize: 是否进行归一化，默认True
    
    Returns:
        tuple: (log_data, availability)
            - log_data: numpy array, shape [10, 48]
            - availability: bool - 整个log模态是否可用
    """
    # 使用全局定义的服务顺序
    num_instances = len(SERVICES)
    num_templates = get_log_template_count('gaia')  # 动态获取模板数量
    
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
    
    # 计算整个模态的可用性（如果所有数据都是0，则整个模态不可用）
    availability = not np.all(log_data == 0)
    
    # 归一化（不填充，保持0值）
    if normalize and NORMALIZATION_STATS['log'] is not None:
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
        st_time: 故障开始时间戳（毫秒）
        ed_time: 故障结束时间戳（毫秒）
        normalize: 是否进行归一化，默认True
    
    Returns:
        tuple: (trace_data, availability)
            - trace_data: numpy array, shape [10, 20, 2]
            - availability: bool - 整个trace模态是否可用
    """
    # 使用全局定义的服务顺序
    num_instances = len(SERVICES)
    num_time_segments = 20  # 20个时间段
    segment_duration = 30 * 1000  # 每个时间段30秒（毫秒）
    num_channels = 2 # Duration + ErrorRate
    
    # 初始化结果数组 [num_instances, num_time_segments, 2]，默认值为NaN
    trace_data = np.full((num_instances, num_time_segments, num_channels), np.nan)
    
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
                status_codes = sample_data['status_code'].values
                
                # 批量计算时间偏移和段索引
                time_offsets = timestamps - st_time
                segment_indices = (time_offsets // segment_duration).astype(int)
                
                # 筛选有效的段索引
                valid_mask = (segment_indices >= 0) & (segment_indices < num_time_segments)
                valid_segments = segment_indices[valid_mask]
                valid_durations = durations[valid_mask]
                valid_status = status_codes[valid_mask]
                
                # 按段索引分组计算
                for seg_idx in range(num_time_segments):
                    seg_mask = valid_segments == seg_idx
                    if seg_mask.any():
                        # 1. Duration Mean
                        mean_duration = valid_durations[seg_mask].mean()
                        trace_data[instance_idx, seg_idx, 0] = mean_duration
                        
                        # 2. Error Rate (status_code >= 400)
                        # 计算错误请求的比例
                        seg_status = valid_status[seg_mask]
                        error_count = np.sum(seg_status >= 400)
                        error_rate = error_count / len(seg_status)
                        trace_data[instance_idx, seg_idx, 1] = error_rate
        
        except Exception:
            continue
    
    # 计算整个模态的可用性（如果所有数据都是NaN，则整个模态不可用）
    availability = not np.all(np.isnan(trace_data))
    
    # 处理和归一化
    if normalize and NORMALIZATION_STATS['trace'] is not None:
        for i in range(num_instances):
            stats = NORMALIZATION_STATS['trace'][i]
            
            # Channel 0 (Duration): 用均值填充NaN，然后归一化
            nan_mask_0 = np.isnan(trace_data[i, :, 0])
            if nan_mask_0.any():
                trace_data[i, :, 0][nan_mask_0] = stats['mean']
            trace_data[i, :, 0] = (trace_data[i, :, 0] - stats['mean']) / stats['std']
            
            # Channel 1 (Error Rate): 用0填充NaN (没有请求就没有错误)，不归一化(本身0-1)
            nan_mask_1 = np.isnan(trace_data[i, :, 1])
            if nan_mask_1.any():
                trace_data[i, :, 1][nan_mask_1] = 0.0
            # Error Rate 不需要 Z-Score 归一化
            
    else:
        # 如果不归一化（统计阶段），保持NaN
        pass
    
    return trace_data, availability


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
    label_file = os.path.join(_project_root, "data", "processed_data", "gaia", "label_gaia.csv")
    label_df = pd.read_csv(label_file)
    
    # 1. 预加载所有数据到内存
    preload_all_data()
    
    # 2. 计算或加载归一化统计信息
    stats_file = os.path.join(_project_root, "data", "processed_data", "gaia", "norm_stats.pkl")
    
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
    output_file = os.path.join(_project_root, "data", "processed_data", "gaia", "dataset.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\n💾 数据集已保存: {output_file}")
    print(f"   - 样本数: {len(processed_data)}")
    print(f"\n数据处理策略:")
    print(f"   Metric: 排除NaN统计，NaN填充为均值")
    print(f"   Log: 排除0统计，不填充（0是真实的'未出现'）")
    print(f"   Trace: 双通道 (Duration, ErrorRate)")
    print(f"     - Ch0(Duration): 归一化，NaN填充均值")
    print(f"     - Ch1(ErrorRate): 不归一化，NaN填充0")
    print(f"\n可用性标记: 每个样本包含模态级别标记")
    print(f"   - metric_available: bool（整个模态是否可用）")
    print(f"   - log_available: bool（整个模态是否可用）")
    print(f"   - trace_available: bool（整个模态是否可用）")
