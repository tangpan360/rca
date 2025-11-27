#!/usr/bin/env python3
"""
log数据处理工具 - 提取正常/异常数据并分析调用模式
异常时间段定义：从标签开始时间往后600秒固定窗口
"""

import os
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# Drain相关导入
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
extractor_path = os.path.join(project_root, 'extractor')
sys.path.append(extractor_path)
from drain.drain_template_extractor import init_drain, extract_templates


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


# def extract_anomaly_log_data(log_dir, anomaly_periods, output_dir):
#     """
#     从MicroSS/business目录中的所有文件提取异常时间段的log数据
#     按原文件名分别保存到preprocess/logs目录
    
#     Args:
#         log_dir (str): 源log文件目录
#         anomaly_periods (list): 异常时间段列表
#         output_dir (str): 输出目录
#     """
#     print("=== 开始提取异常log数据 ===")
    
#     # 自动创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 获取所有log文件
#     log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
#     print(f"发现 {len(log_files)} 个log文件: {log_files}")
    
#     # 处理每个文件    
#     for log_file in log_files:
#         print(f"\n处理文件: {log_file}")
#         log_file_path = os.path.join(log_dir, log_file)
        
#         # 读取log文件
#         log_df = pd.read_csv(log_file_path)
#         original_count = len(log_df)
#         print(f"  读取数据: {original_count:,} 条")
        
#         # 从message字段中提取时间戳并转换为时间戳（毫秒）
#         # 假设message格式为: "2021-07-01 10:54:22,639 | ..."
#         log_df['timestamp'] = log_df['message'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
#         log_df['timestamp_ts'] = pd.to_datetime(log_df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f').astype('int64') // 10**6
        
#         # 移除无法解析时间戳的行
#         log_df = log_df.dropna(subset=['timestamp_ts'])
        
#         # 【异常数据提取】创建掩码标识正常时间段数据（为了取反提取异常数据）
#         normal_mask = create_selection_mask(log_df['timestamp_ts'], anomaly_periods, data_type_filter=None)
        
#         # 取反掩码，提取异常时间段内的数据
#         anomaly_data = log_df[~normal_mask].copy()
#         anomaly_count = len(anomaly_data)
        
#         # 保存到同名文件
#         output_file_path = os.path.join(output_dir, log_file)
#         anomaly_data.to_csv(output_file_path, index=False)
        
#         print(f"  提取异常数据: {anomaly_count:,} 条 ({anomaly_count/original_count:.2%})")
#         print(f"  保存至: {output_file_path}")


def process_single_log_file(args):
    """
    处理单个log文件的工作函数（用于多进程）
    
    Args:
        args (tuple): 包含(log_file, log_dir, anomaly_periods, output_dir)的元组
        
    Returns:
        dict: 处理结果统计
    """
    log_file, log_dir, anomaly_periods, output_dir = args
    
    try:
        print(f"\n[进程] 处理文件: {log_file}")
        log_file_path = os.path.join(log_dir, log_file)
        
        # 读取log文件
        log_df = pd.read_csv(log_file_path)
        original_count = len(log_df)
        print(f"  [进程] 读取数据: {original_count:,} 条")
        
        # 从message字段中提取时间戳并转换为时间戳（毫秒）
        # 假设message格式为: "2021-07-01 10:54:22,639 | ..."
        log_df['timestamp'] = log_df['message'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
        log_df['timestamp_ts'] = pd.to_datetime(log_df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f').astype('int64') // 10**6
        
        # 移除无法解析时间戳的行
        log_df = log_df.dropna(subset=['timestamp_ts'])
        
        # 创建掩码标识异常时间段数据
        anomaly_mask = create_selection_mask(log_df['timestamp_ts'], anomaly_periods)
        
        # 提取异常时间段内的数据
        anomaly_data = log_df[anomaly_mask]
        anomaly_count = len(anomaly_data)
        
        # 从文件名中提取instance名字（第3个部分）
        splits = log_file.replace('.csv', '').split('_')
        instance_name = splits[2]  # 第3个部分作为instance名字
        
        # 保存到新命名的文件
        output_filename = f"{instance_name}_log.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        anomaly_data.to_csv(output_file_path, index=False)
        
        print(f"  [进程] 提取异常数据: {anomaly_count:,} 条 ({anomaly_count/original_count:.2%})")
        print(f"  [进程] 保存至: {output_file_path}")
        
        return {
            'file': log_file,
            'original_count': original_count,
            'anomaly_count': anomaly_count,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"  [进程] 处理文件 {log_file} 时出错: {e}")
        return {
            'file': log_file,
            'original_count': 0,
            'anomaly_count': 0,
            'status': 'error',
            'error': str(e)
        }


def extract_anomaly_log_data_multiprocess(log_dir, anomaly_periods, output_dir, n_processes=None):
    """
    使用多进程从MicroSS/business目录中的所有文件提取异常时间段的log数据
    按原文件名分别保存到preprocess/logs目录
    
    Args:
        log_dir (str): 源log文件目录
        anomaly_periods (list): 异常时间段列表
        output_dir (str): 输出目录
        n_processes (int, optional): 进程数，默认为CPU核心数
    """
    print("=== 开始提取异常log数据 (多进程版本) ===")
    
    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有log文件
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    print(f"发现 {len(log_files)} 个log文件: {log_files}")
    
    # 确定进程数
    if n_processes is None:
        n_processes = min(cpu_count(), len(log_files))  # 不超过文件数
    
    print(f"使用 {n_processes} 个进程并行处理")
    
    # 准备参数列表
    args_list = [(log_file, log_dir, anomaly_periods, output_dir) for log_file in log_files]
    
    # 使用多进程处理
    start_time = datetime.now()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_log_file, args_list)
    
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
    for period_idx, (start_time, end_time) in enumerate(tqdm(anomaly_periods, desc="处理异常周期", unit="period")):
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


def train_drain_from_logs(log_dir, output_dir, anomaly_periods=None):
    """
    使用log数据训练drain模板，筛选出data_type='train'的异常时间段

    Args:
        log_dir (str): 源log文件目录
        output_dir (str): 输出目录
        anomaly_periods (list, optional): 异常时间段列表(三元组格式)，如果为None则使用全部数据

    Returns:
        TemplateMiner: 训练好的drain模型
    """
    print("=== 开始使用log数据训练drain模板 ===")
    print("筛选data_type='train'的异常时间段用于训练")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 筛选出训练集的异常时间段    
    train_anomaly_periods = [(st, ed, dt) for st, ed, dt in anomaly_periods if dt == 'train']
    print(f"从 {len(anomaly_periods)} 个异常时间段中筛选出 {len(train_anomaly_periods)} 个训练集时间段")

    # 获取所有log文件
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]

    # 收集所有日志消息
    all_log_messages = []
    total_logs = 0

    for log_file in tqdm(log_files, desc="读取log文件"):
        log_file_path = os.path.join(log_dir, log_file)

        try:
            # 读取log文件
            log_df = pd.read_csv(log_file_path)
            print(f"\n处理文件 {log_file}: {len(log_df):,} 条记录")

            # 移除无法解析时间戳的行
            log_df = log_df.dropna(subset=['timestamp_ts'])
            print(f"  有效时间戳数据: {len(log_df):,} 条")

            # 为训练数据创建掩码标识（用于提取正常数据训练drain）
            train_mask = create_selection_mask(log_df['timestamp_ts'], train_anomaly_periods)
            filtered_df = log_df[train_mask]  # 直接使用掩码，提取训练时间段的数据
            print(f"  筛选后训练集数据: {len(filtered_df):,} 条")

            # 提取消息内容
            messages = filtered_df['message'].tolist()
            all_log_messages.extend(messages)
            total_logs += len(messages)

        except Exception as e:
            print(f"  警告: 处理文件 {log_file} 时出错: {e}")
            continue

    print(f"\n总共收集到 {total_logs:,} 条日志消息用于训练")

    # 使用drain训练模板
    print("\n开始训练drain模板...")
    drain_model_path = os.path.join(output_dir, "gaia_drain.pkl")

    miner = extract_templates(
        log_list=all_log_messages,
        save_pth=drain_model_path
    )

    # 保存模板信息
    template_csv_path = os.path.join(output_dir, "gaia_templates.csv")
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)

    template_data = {
        'template_id': [],
        'template': [],
        'count': [],
        'percentage': []
    }

    for cluster in sorted_clusters:
        template_data['template_id'].append(cluster.cluster_id)
        template_data['template'].append(cluster.get_template())
        template_data['count'].append(cluster.size)
        template_data['percentage'].append(cluster.size / total_logs * 100)

    template_df = pd.DataFrame(template_data)
    template_df.to_csv(template_csv_path, index=False)

    return miner


def load_drain_model(model_path):
    """
    加载训练好的drain模型
    """
    with open(model_path, 'rb') as f:
        miner = pickle.load(f)
    return miner

def add_template_columns_single_file(args):
    """
    为单个csv文件添加template列和template_id列

    Args:
        args (tuple): 包含(file_path, model_path)的元组

    Returns:
        dict: 处理结果统计
    """
    file_path, model_path = args
    miner = load_drain_model(model_path)

    try:
        df = pd.read_csv(file_path)

        if 'template' in df.columns and 'template_id' in df.columns:
            print(f"文件 {os.path.basename(file_path)} 已经包含template和template_id列，跳过处理")
            return f"跳过 {os.path.basename(file_path)}"

        templates = []
        template_ids = []
        for message in tqdm(df['message'], desc=f"处理 {os.path.basename(file_path)}"):
            match = miner.match(message)
            if match:
                template = match.get_template()
                template_id = match.cluster_id
            else:
                template = "Unseen"
                template_id = -1  # 对于未见过的日志，使用-1作为ID
            templates.append(template)
            template_ids.append(template_id)
        
        df['template'] = templates
        df['template_id'] = template_ids

        df.to_csv(file_path, index=False)

        return f"完成 {os.path.basename(file_path)}: 处理了 {len(df)} 行数据"
    
    except Exception as e:
        return f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}"


def add_template_columns_multiprocess(logs_dir, model_path, numprocess=None):
    """
    为logs目录下的所有csv文件添加template列和template_id列

    Args:
        logs_dir (str): 日志文件目录路径
        model_path (str): 训练好的drain模型路径
        num_processes (int): 进程数，默认为CPU核心数
    """
    csv_files = [f for f in os.listdir(logs_dir) if f.endswith('.csv')]
    file_paths = [os.path.join(logs_dir, f) for f in csv_files]
    
    if numprocess is None:
        num_processes = min(cpu_count(), len(file_paths))
    
    print(f"使用 {num_processes} 个进程处理")

    args_list = [(file_path, model_path) for file_path in file_paths]

    with Pool(processes=num_processes) as pool:
        results = pool.map(add_template_columns_single_file, args_list)

    print("\n处理结果:")
    for result in results:
        print(result)


def main():
    """
    主函数
    """
    # 获取项目根目录
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 定义文件路径（使用相对路径）
    label_file = os.path.join(project_dir, "preprocess", "raw_data", "gaia", "label_gaia.csv")
    log_dir = os.path.join(project_dir, "preprocess", "raw_data", "gaia", "business")
    output_dir = os.path.join(project_dir, "preprocess", "processed_data", "gaia", "log")
    drain_dir = os.path.join(project_dir, "preprocess", "processed_data", "gaia", "drain_models")    
    
    # 1. 加载异常时间段
    anomaly_periods = load_anomaly_periods(label_file)

    # 提取异常log数据（多进程版本）
    extract_anomaly_log_data_multiprocess(log_dir, anomaly_periods, output_dir)
    
    # 训练drain模板（使用训练集的正常数据）
    train_drain_from_logs(output_dir, drain_dir, anomaly_periods)

    # 添加模板列
    model_path = os.path.join(drain_dir, "gaia_drain.pkl")
    add_template_columns_multiprocess(output_dir, model_path)
    

if __name__ == "__main__":
    main()
