#!/usr/bin/env python3
"""
trace数据处理工具 - 提取正常/异常数据并分析调用模式
异常时间段定义：从标签开始时间往后600秒固定窗口
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
extractor_path = os.path.join(project_root, 'extractor')
sys.path.append(extractor_path)
from utils.time_util import coast_time as time_decorator
warnings.filterwarnings('ignore')


@time_decorator
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


def extract_anomaly_trace_data(trace_dir, anomaly_periods, output_dir):
    """
    从MicroSS/trace目录中的所有文件提取异常时间段的trace数据
    按原文件名分别保存到preprocess/trace目录
    
    Args:
        trace_dir (str): 源trace文件目录
        anomaly_periods (list): 异常时间段列表
        output_dir (str): 输出目录
    """
    print("=== 开始提取异常trace数据 ===")
    
    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有trace文件
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    print(f"发现 {len(trace_files)} 个trace文件: {trace_files}")
    
    # 处理每个文件    
    for trace_file in trace_files:
        print(f"\n处理文件: {trace_file}")
        trace_file_path = os.path.join(trace_dir, trace_file)
        
        # 读取trace文件
        trace_df = pd.read_csv(trace_file_path)
        original_count = len(trace_df)
        print(f"  读取数据: {original_count:,} 条")
        
        # 将start_time转换为时间戳（毫秒）
        trace_df['start_time_ts'] = pd.to_datetime(trace_df['start_time'], format='mixed').astype('int64') // 10**6
        trace_df['end_time_ts'] = pd.to_datetime(trace_df['end_time'], format='mixed').astype('int64') // 10**6
        
        # 使用create_selection_mask函数创建异常数据掩码，然后取反得到正常数据掩码
        anomaly_mask = create_selection_mask(trace_df['start_time_ts'], anomaly_periods)
        normal_mask = ~anomaly_mask
        
        # 提取异常数据
        anomaly_data = trace_df[anomaly_mask].copy()
        anomaly_count = len(anomaly_data)
        
        # 计算duration（单位为毫秒）
        anomaly_data['duration'] = anomaly_data['end_time_ts'] - anomaly_data['start_time_ts']
        
        # 从文件名中提取instance名字（第3个部分）
        splits = trace_file.replace('.csv', '').split('_')
        instance_name = splits[2]  # 第3个部分作为instance名字
        
        # 保存到新命名的文件
        output_filename = f"{instance_name}_trace.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        anomaly_data.to_csv(output_file_path, index=False)
        
        print(f"  提取异常数据: {anomaly_count:,} 条 ({anomaly_count/original_count:.2%})")
        print(f"  保存至: {output_file_path}")


def process_single_trace_file(args):
    """
    处理单个trace文件的工作函数（用于多进程）
    
    Args:
        args (tuple): 包含(trace_file, trace_dir, anomaly_periods, output_dir)的元组
        
    Returns:
        dict: 处理结果统计
    """
    trace_file, trace_dir, anomaly_periods, output_dir = args
    
    try:
        print(f"\n[进程] 处理文件: {trace_file}")
        trace_file_path = os.path.join(trace_dir, trace_file)
        
        # 读取trace文件
        trace_df = pd.read_csv(trace_file_path)
        original_count = len(trace_df)
        print(f"  [进程] 读取数据: {original_count:,} 条")
        
        # 将start_time转换为时间戳（毫秒）
        trace_df['start_time_ts'] = pd.to_datetime(trace_df['start_time'], format='mixed').astype('int64') // 10**6
        trace_df['end_time_ts'] = pd.to_datetime(trace_df['end_time'], format='mixed').astype('int64') // 10**6
        
        # 使用create_selection_mask函数创建异常数据掩码，然后取反得到正常数据掩码
        anomaly_mask = create_selection_mask(trace_df['start_time_ts'], anomaly_periods)
        
        # 提取异常数据
        anomaly_data = trace_df[anomaly_mask].copy()
        anomaly_count = len(anomaly_data)
        
        # 计算duration（单位为毫秒）
        anomaly_data['duration'] = anomaly_data['end_time_ts'] - anomaly_data['start_time_ts']
        
        # 从文件名中提取instance名字（第3个部分）
        splits = trace_file.replace('.csv', '').split('_')
        instance_name = splits[2]  # 第3个部分作为instance名字
        
        # 保存到新命名的文件
        output_filename = f"{instance_name}_trace.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        anomaly_data.to_csv(output_file_path, index=False)
        
        print(f"  [进程] 提取异常数据: {anomaly_count:,} 条 ({anomaly_count/original_count:.2%})")
        print(f"  [进程] 保存至: {output_file_path}")
        
        return {
            'file': trace_file,
            'original_count': original_count,
            'anomaly_count': anomaly_count,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"  [进程] 处理文件 {trace_file} 时出错: {e}")
        return {
            'file': trace_file,
            'original_count': 0,
            'anomaly_count': 0,
            'status': 'error',
            'error': str(e)
        }


def extract_anomaly_trace_data_multiprocess(trace_dir, anomaly_periods, output_dir, n_processes=None):
    """
    使用多进程从MicroSS/trace目录中的所有文件提取异常时间段的trace数据
    按原文件名分别保存到preprocess/trace目录
    
    Args:
        trace_dir (str): 源trace文件目录
        anomaly_periods (list): 异常时间段列表
        output_dir (str): 输出目录
        n_processes (int, optional): 进程数，默认为CPU核心数
    """
    print("=== 开始提取异常trace数据 (多进程版本) ===")
    
    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有trace文件
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    print(f"发现 {len(trace_files)} 个trace文件")
    
    # 确定进程数
    if n_processes is None:
        n_processes = min(cpu_count(), len(trace_files))  # 不超过文件数
    
    print(f"使用 {n_processes} 个进程并行处理")
    
    # 准备参数列表
    args_list = [(trace_file, trace_dir, anomaly_periods, output_dir) for trace_file in trace_files]
    
    # 使用多进程处理
    start_time = datetime.now()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_trace_file, args_list)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    total_original = sum(r['original_count'] for r in results if r['status'] == 'success')
    total_anomaly = sum(r['anomaly_count'] for r in results if r['status'] == 'success')
    
    print(f"\n=== 多进程处理完成 ===")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"总原始数据: {total_original:,} 条")
    print(f"总异常数据: {total_anomaly:,} 条 ({total_anomaly/total_original:.2%} 异常率)")
    
    # 显示失败的文件
    if error_count > 0:
        print("\n失败的文件:")
        for r in results:
            if r['status'] == 'error':
                print(f"  {r['file']}: {r['error']}")
    
    return results


def extract_service_durations_by_timesegments(trace_dir, anomaly_periods, output_file):
    """
    读取所有trace文件，根据anomaly_periods的600秒周期将数据划分为20个30秒时间段，
    提取每个时间段内各个service的所有duration值，保存为JSON格式
    
    Args:
        trace_dir (str): trace文件目录
        anomaly_periods (list): 异常时间段列表，每个元素为(start_timestamp, end_timestamp)的元组
        output_file (str): 输出JSON文件路径
    """
    import json
    
    print("=== 开始提取服务duration时序数据 ===")
    
    # 1. 获取所有trace文件并一次性读取到内存
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    print(f"发现 {len(trace_files)} 个trace文件")
    
    # 2. 一次性读取所有文件到内存
    print("正在读取所有trace文件到内存...")
    trace_dataframes = {}  # {instance_name: dataframe}
    
    from tqdm import tqdm
    for trace_file in tqdm(trace_files, desc="读取文件", unit="file"):
        # 获取instance名称
        parts = trace_file.split('_')
        if len(parts) < 3:
            continue
        instance_name = parts[2]  # 如：dbservice1
        
        try:
            trace_file_path = os.path.join(trace_dir, trace_file)
            df = pd.read_csv(trace_file_path)
            trace_dataframes[instance_name] = df
            print(f"  读取 {instance_name}: {len(df):,} 条数据")
        except Exception as e:
            print(f"  警告: 读取文件 {trace_file} 失败: {e}")
            continue
    
    print(f"成功读取 {len(trace_dataframes)} 个文件到内存")
    
    # 3. 初始化结果字典
    result_data = {}
    
    print(f"\n开始处理 {len(anomaly_periods)} 个异常周期...")
    
    # 4. 处理每个异常周期
    for period_idx, (start_time, end_time, _) in enumerate(tqdm(anomaly_periods, desc="处理异常周期", unit="period")):
        # 将600秒周期划分为20个30秒时间段
        for segment in range(20):
            segment_start = int(start_time + segment * 30 * 1000)  # 毫秒时间戳
            segment_end = int(segment_start + 30 * 1000)
            
            segment_key = str(segment_start)
            if segment_key not in result_data:
                result_data[segment_key] = {}
            
            # 5. 处理每个instance的内存数据
            for instance_name, df in trace_dataframes.items():
                try:
                    # 筛选该时间段内的数据
                    mask = (df['start_time_ts'] >= segment_start) & (df['start_time_ts'] < segment_end)
                    segment_data = df[mask]
                    
                    # 如果该时间段有数据，添加到结果中
                    if len(segment_data) > 0:
                        durations = segment_data['duration'].tolist()
                        
                        # 使用完整的instance名称作为service key（如：dbservice1, dbservice2）
                        if instance_name not in result_data[segment_key]:
                            result_data[segment_key][instance_name] = []
                        
                        result_data[segment_key][instance_name].extend(durations)
                
                except Exception as e:
                    print(f"    警告: 处理 {instance_name} 在时间段 {segment_key} 时出错: {e}")
                    continue
    
    # 5. 清理空的时间段
    result_data = {k: v for k, v in result_data.items() if v}
    
    # 6. 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # 7. 统计信息
    total_segments = len(result_data)
    total_services = set()
    total_values = 0
    
    for segment_data in result_data.values():
        total_services.update(segment_data.keys())
        for service_values in segment_data.values():
            total_values += len(service_values)
    
    print(f"\n=== 数据提取完成 ===")
    print(f"异常周期数: {len(anomaly_periods)}")
    print(f"有效时间段: {total_segments}")
    print(f"服务类型数: {len(total_services)} ({list(total_services)})")
    print(f"总数据点数: {total_values:,}")
    print(f"平均每时间段: {total_values/total_segments:.1f} 个数据点")
    print(f"输出文件: {output_file}")
    
    return result_data


def create_selection_mask(times, target_periods):
    """
    创建指定时间段的选择掩码，用于标记时间序列中属于目标周期的部分
    
    Args:
        times (pd.Series): 时间戳系列
        target_periods (list): 目标时间段列表，每个元素为(start_time, end_time, ...)
        
    Returns:
        pd.Series: 布尔掩码，True表示属于目标周期，False表示不属于
    """
    # 初始化所有时间点为未选中（False）
    is_in_target = pd.Series(False, index=times.index)
    
    # 对每个目标时间段，标记其中的时间点为选中（True）
    for start_time, end_time, _ in target_periods:
        # 找出当前目标时间段内的所有时间点
        in_current_period = (times >= start_time) & (times <= end_time)
        
        # 将这些时间点标记为选中（True）
        is_in_target.loc[in_current_period] = True
    
    return is_in_target

@time_decorator
def main():
    """
    主函数
    """
    # 获取项目根目录
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 定义文件路径（使用相对路径）
    label_file = os.path.join(project_dir, "extractor", "MicroSS", "label.csv")
    trace_dir = os.path.join(project_dir, "extractor", "MicroSS", "trace")
    output_dir = os.path.join(project_dir, "preprocess", "processed_data", "trace")
    
    # 1. 加载异常时间段
    anomaly_periods = load_anomaly_periods(label_file)

    # 提取异常trace数据（多进程版本）
    extract_anomaly_trace_data_multiprocess(trace_dir, anomaly_periods, output_dir)
    
    """
    # 8. 提取服务duration时序数据（可选）
    # json_output_file = os.path.join(output_dir, "service_durations_timeseries.json")
    # extract_service_durations_by_timesegments(output_dir, anomaly_periods, json_output_file)  # 这里提取出来的数据有时间重叠，且重叠部分时间
    """


if __name__ == "__main__":
    main()
