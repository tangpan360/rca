import os
import subprocess
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from datetime import datetime


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def copy_valid_metric_files() -> None:
    """
    该函数将原始 metric 文件夹（preprocess/raw_data/gaia/metric）下同时包含特定服务名和关键Docker指标的文件，
    复制到本模块下的 processed_data 文件夹中。

    输入:
        无

    返回值：
        无
    """
    # print(os.path.dirname(os.path.abspath(__file__)))
    
    metric_dir = os.path.join(project_dir, 'preprocess', 'raw_data', 'gaia', 'metric')
    # print(metric_dir)

    file_names = os.listdir(metric_dir)

    services = ['dbservice', 'mobservice', 'logservice', 'webservice', 'redisservice']
    
    docker_metrics = [
        "docker_cpu_total_norm_pct",    # 总CPU使用率
        # "docker_cpu_user_pct",          # 用户态CPU使用
        # "docker_cpu_kernel_pct",        # 内核态CPU使用
        "docker_memory_usage_pct",      # 内存使用百分比
        # "docker_memory_usage_total",    # 内存使用总量
        # "docker_memory_limit",          # 内存限制
        "docker_memory_fail_count",     # 内存分配失败次数
        "docker_diskio_read_bytes",     # 读取字节数
        "docker_diskio_write_bytes",    # 写入字节数
        # "docker_diskio_read_ops",       # 读操作数
        # "docker_diskio_write_ops",      # 写操作数
        "docker_diskio_read_service_time",  # 读服务时间
        "docker_diskio_write_service_time", # 写服务时间
        "docker_network_in_bytes",      # 入站流量
        "docker_network_out_bytes",     # 出站流量
        # "docker_network_in_packets",    # 入站包数
        # "docker_network_out_packets",   # 出站包数
        "docker_network_in_errors",     # 入站错误
        "docker_network_out_errors",    # 出站错误
        # "docker_network_in_dropped",    # 入站丢包
        "docker_network_out_dropped",   # 出站丢包
    ]
    
    # 筛选同时包含服务名和关键指标的文件
    valid_files = []
    for fn in file_names:
        # 检查是否包含任一服务名
        has_service = any(s in fn for s in services)
        # 检查是否包含任一关键指标
        has_metric = any(m in fn for m in docker_metrics)
        
        if has_service and has_metric:
            valid_files.append(fn)
    
    print(f"筛选出 {len(valid_files)} 个包含关键指标的文件，从总共 {len(file_names)} 个文件中")
    
    # 将符合条件的metric文件复制到processed_data中
    for file in valid_files:
        file_path = os.path.join(metric_dir, file)
        processed_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'metric')
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        target_path = os.path.join(processed_data_dir, file)
        subprocess.run(['cp', file_path, target_path])

def merge_date_range_files() -> None:
    """
    合并相同服务、主机、指标的不同时间段文件。
    将 2021-07-01_2021-07-15 和 2021-07-15_2021-07-31 合并成 2021-07-01_2021-07-31
    
    输入:
        无
    
    返回值:
        无
    """
    processed_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'metric')
    
    if not os.path.exists(processed_data_dir):
        print("processed_data 目录不存在")
        return
    
    # 获取所有07-01到07-15的文件
    first_period_files = []
    for f in os.listdir(processed_data_dir):
        if "2021-07-01_2021-07-15" in f and f.endswith('.csv'):
            first_period_files.append(f)
    
    print(f"找到 {len(first_period_files)} 个第一时间段的文件")
    
    merged_count = 0
    failed_count = 0
    
    for first_file in tqdm(first_period_files, desc="合并不同时间段的同一指标文件"):
        # 构造对应的第二时间段文件名
        second_file = first_file.replace("2021-07-01_2021-07-15", "2021-07-15_2021-07-31")
        merged_file = first_file.replace("2021-07-01_2021-07-15", "2021-07-01_2021-07-31")
        
        first_path = os.path.join(processed_data_dir, first_file)
        second_path = os.path.join(processed_data_dir, second_file)
        merged_path = os.path.join(processed_data_dir, merged_file)
        
        # 检查第二个文件是否存在
        if not os.path.exists(second_path):
            print(f"警告：找不到对应的第二时间段文件: {second_file}")
            failed_count += 1
            continue
        
        try:
            # 读取两个CSV文件
            df1 = pd.read_csv(first_path)
            df2 = pd.read_csv(second_path)
            
            # 合并数据
            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # 按时间戳排序
            timestamp_col = 'timestamp'
            merged_df = merged_df.sort_values(timestamp_col).reset_index(drop=True)
            
            # 去除重复时间戳，保留第一次出现的记录
            merged_df = merged_df.drop_duplicates(subset=[timestamp_col], keep='first').reset_index(drop=True)
            
            # 保存合并后的文件
            merged_df.to_csv(merged_path, index=False)
            
            # 删除原来的两个文件
            os.remove(first_path)
            os.remove(second_path)
            
            merged_count += 1
            
        except Exception as e:
            print(f"合并文件 {first_file} 时出错: {e}")
            failed_count += 1
            continue
    
    print(f"成功合并 {merged_count} 对文件")
    print(f"失败 {failed_count} 对文件")
    print(f"合并后文件总数: {len([f for f in os.listdir(processed_data_dir) if f.endswith('.csv')])}")

def merge_metrics_by_service_instance() -> None:
    """
    将相同服务实例的不同指标文件合并成多列格式
    例如：
    - dbservice1_0.0.0.4_docker_cpu_kernel_pct_2021-07-01_2021-07-31.csv
    - dbservice1_0.0.0.4_docker_cpu_user_pct_2021-07-01_2021-07-31.csv
    合并为：
    - dbservice1_0.0.0.4_2021-07-01_2021-07-31.csv (包含多个指标列)
    
    输入:
        无
    
    返回值:
        无
    """
    processed_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'metric')
    
    # 获取所有CSV文件
    csv_files = os.listdir(processed_data_dir)
    csv_files = sorted(csv_files)
    print(f"找到 {len(csv_files)} 个需要合并的指标文件")
    
    # 按服务实例分组文件
    service_groups = {}
    for filename in csv_files:
        # 解析文件名: service_ip_metric_daterange.csv
        # 例如: dbservice1_0.0.0.4_docker_cpu_kernel_pct_2021-07-01_2021-07-31.csv
        
        # 通过"_"分割文件名
        splits = filename.replace('.csv', '').split('_')
        
        if len(splits) >= 6:  # 至少需要: service_ip_docker_metric_2021-07-01_2021-07-31
            # 只取第一个部分作为服务实例: dbservice1
            service_instance = splits[0]
            
            # 从第4个部分开始到倒数第3个部分是指标名称
            # 例如: splits = ['dbservice1', '0.0.0.4', 'docker', 'cpu', 'kernel', 'pct', '2021-07-01', '2021-07-31']
            # 指标名称应该是: splits[3:len(splits)-2] = ['cpu', 'kernel', 'pct']
            metric_name = '_'.join(splits[3:len(splits)-2])
            
            if service_instance not in service_groups:
                service_groups[service_instance] = {}
            
            service_groups[service_instance][metric_name] = {
                'filename': filename
            }
        else:
            print(f"⚠️ 无法解析文件名: {filename} (分割后只有{len(splits)}个部分)")
    
    print(f"识别出 {len(service_groups)} 个服务实例")
    
    # 收集所有指标名称并排序，确保所有instance使用相同的列顺序
    all_metrics = set()
    for metrics in service_groups.values():
        all_metrics.update(metrics.keys())
    sorted_metrics = sorted(all_metrics)
    print(f"识别出 {len(sorted_metrics)} 个不同的指标")
    print(f"指标列表（按字母顺序）: {sorted_metrics}")
    
    merged_count = 0
    
    for service_instance, metrics in service_groups.items():
        
        print(f"🔄 合并 {service_instance}: {len(metrics)} 个指标")
        
        try:
            merged_df = None
            
            for metric_name, info in metrics.items():
                file_path = os.path.join(processed_data_dir, info['filename'])
                df = pd.read_csv(file_path)
                
                # 重命名value列为指标名称
                df = df.rename(columns={'value': metric_name})
                
                if merged_df is None:
                    merged_df = df
                else:
                    # 基于timestamp进行外连接合并
                    merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')
            
            # 按时间戳排序
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            
            # 确保列顺序一致：timestamp列在前，然后是按字母顺序排列的指标列
            # 只保留该instance实际拥有的指标列
            available_metrics = [m for m in sorted_metrics if m in merged_df.columns]
            column_order = ['timestamp'] + available_metrics
            merged_df = merged_df[column_order]
            
            # 生成合并后的文件名
            merged_filename = f"{service_instance}_metric.csv"
            merged_path = os.path.join(processed_data_dir, merged_filename)
            
            # 保存合并后的文件
            merged_df.to_csv(merged_path, index=False)
            
            # 删除原始的单指标文件
            for metric_name, info in metrics.items():
                original_path = os.path.join(processed_data_dir, info['filename'])
                os.remove(original_path)
            
            print(f"✅ {service_instance}: 合并完成 -> {merged_filename}")
            print(f"   指标列: {list(metrics.keys())}")
            print(f"   数据行数: {len(merged_df)}")
            
            merged_count += 1
            
        except Exception as e:
            print(f"❌ 合并 {service_instance} 时出错: {e}")
            continue
    
    print(f"\n合并完成:")
    print(f"  成功合并的服务实例: {merged_count}")
    print(f"  最终文件数: {len([f for f in os.listdir(processed_data_dir) if f.endswith('.csv')])}")


def process_single_file_resample(file_path: str) -> tuple:
    """
    处理单个文件的30秒间隔重采样，直接覆盖原文件
    
    参数:
        file_path: 文件完整路径
    
    返回:
        tuple: (文件名, 是否成功, 原始行数, 重采样后行数, 错误信息或None)
    """
    try:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        
        # 检查是否有timestamp列
        if 'timestamp' not in df.columns:
            return (filename, False, 0, 0, "缺少timestamp列")
        
        original_rows = len(df)
        
        # 获取数据的时间范围
        min_timestamp = df['timestamp'].min()
        max_timestamp = df['timestamp'].max()
        
        # 固定的起始时间戳（毫秒）和间隔
        start_timestamp = 1625133601000  # 2021-07-01 00:00:01 GMT，固定起始时间
        interval_ms = 30 * 1000  # 30秒间隔，转换为毫秒
        
        # 从固定的起始时间戳开始处理
        current_timestamp = start_timestamp
        
        # 存储重采样后的数据
        resampled_data = []
        
        while current_timestamp <= max_timestamp:
            # 定义30秒窗口的结束时间
            window_end = current_timestamp + interval_ms
            
            # 查找在当前30秒窗口内的数据
            window_data = df[(df['timestamp'] >= current_timestamp) & 
                           (df['timestamp'] < window_end)]
            
            if not window_data.empty:
                # 如果有数据，取第一行（时间最早的）
                first_record = window_data.iloc[0].copy()
                # 将时间戳设置为窗口起始时间
                first_record['timestamp'] = current_timestamp
                resampled_data.append(first_record)
            else:
                # 如果没有数据，创建一个空值记录，保留时间戳
                empty_record = pd.Series(index=df.columns)
                empty_record['timestamp'] = current_timestamp
                # 将所有metric列设置为空值（NaN）
                metric_columns = [col for col in df.columns if col != 'timestamp']
                for col in metric_columns:
                    empty_record[col] = pd.NA
                resampled_data.append(empty_record)
            
            # 移动到下一个30秒窗口
            current_timestamp += interval_ms
        
        if resampled_data:
            # 创建重采样后的DataFrame
            resampled_df = pd.DataFrame(resampled_data)
            
            # 直接覆盖原文件
            resampled_df.to_csv(file_path, index=False)
            
            resampled_rows = len(resampled_df)
            return (filename, True, original_rows, resampled_rows, None)
        else:
            return (filename, False, original_rows, 0, "在指定时间范围内没有找到数据")
            
    except Exception as e:
        return (os.path.basename(file_path), False, 0, 0, str(e))


def resample_metrics_30s_interval(num_processes: int = None) -> None:
    """
    使用多进程对processed_data目录下的所有文件进行30秒间隔重采样，直接覆盖原文件。
    从固定时间戳1625133601000开始，每30秒查找是否有值。
    如果有多个值，只取第一个值，并将其放置到该时间戳处。
    如果没有指标值，则不记录该时间戳。
    
    参数:
        num_processes: 使用的进程数，默认为CPU核心数
    
    返回值:
        无
    """
    processed_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'metric')
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
    
    # 构建完整文件路径
    file_paths = [os.path.join(processed_data_dir, file) for file in csv_files]
    
    # 确定进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(file_paths))
    
    print(f"开始对 {len(file_paths)} 个文件进行30秒间隔重采样（覆盖原文件），使用 {num_processes} 个进程...")
    print(f"固定起始时间戳: 1625133601000 (毫秒)")
    
    # 使用多进程池处理文件
    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度条
        results = list(tqdm(
            pool.imap(process_single_file_resample, file_paths),
            total=len(file_paths),
            desc="重采样文件"
        ))
    
    # 统计结果
    successful_count = 0
    failed_count = 0
    total_original_rows = 0
    total_resampled_rows = 0
    
    for filename, success, original_rows, resampled_rows, error in results:
        if success:
            successful_count += 1
            total_original_rows += original_rows
            total_resampled_rows += resampled_rows
        else:
            failed_count += 1
            if error:
                print(f"❌ {filename}: {error}")
    
    print(f"\n重采样完成:")
    print(f"  成功处理的文件: {successful_count}")
    print(f"  失败的文件: {failed_count}")
    print(f"  原始数据总行数: {total_original_rows:,}")
    print(f"  重采样后总行数: {total_resampled_rows:,}")
    if total_original_rows > 0:
        print(f"  数据压缩率: {((total_original_rows-total_resampled_rows)/total_original_rows*100):.1f}%")
    print(f"  重采样间隔: 30秒")
    print(f"  固定起始时间戳: 1625133601000 (毫秒)")


def process_single_anomaly_sample_all_files(args: tuple) -> tuple:
    """
    处理单个异常样本，从所有10个文件中提取故障时间窗口内的数据
    只要有一个文件有数据，就保留该样本的所有文件数据
    
    参数:
        args: (sample_idx, sample_row, metric_files_dict)
    
    返回:
        tuple: (sample_idx, success, error_msg, all_files_data_dict)
    """
    try:
        sample_idx, sample_row, metric_files_dict = args
        
        # 获取标签信息
        service = sample_row['service']
        instance = sample_row['instance']
        st_time_str = sample_row['st_time']
        anomaly_type = sample_row['anomaly_type']
        
        # 将时间字符串转换为毫秒时间戳
        dt = datetime.strptime(st_time_str, '%Y-%m-%d %H:%M:%S.%f')
        start_timestamp_ms = int(dt.timestamp() * 1000)
        end_timestamp_ms = start_timestamp_ms + 600 * 1000  # 600秒后
        
        # 存储所有文件的数据和时间戳信息
        all_files_data = {}
        all_actual_timestamps = set()
        has_any_data = False
        
        # 第一轮：检查所有文件，收集时间戳信息
        for file_key, metric_file_path in metric_files_dict.items():
            try:
                # 读取文件
                df = pd.read_csv(metric_file_path)
                df.columns = df.columns.str.strip()
                
                # 获取列名信息（只需要一次）
                metric_columns = [col for col in df.columns if col != 'timestamp']
                
                # 提取时间窗口内的数据
                window_data = df[(df['timestamp'] >= start_timestamp_ms) & 
                                (df['timestamp'] <= end_timestamp_ms)]
                
                if not window_data.empty:
                    has_any_data = True
                    actual_timestamps = sorted(window_data['timestamp'].unique())
                    all_actual_timestamps.update(actual_timestamps)
                    all_files_data[file_key] = {'metric_columns': metric_columns, 'window_data': window_data}
                else:
                    all_files_data[file_key] = {'metric_columns': metric_columns, 'window_data': pd.DataFrame()}
                    
            except Exception as e:
                # 如果文件读取失败，创建空数据结构
                default_columns = ['docker_memory_fail_count', 'docker_network_out_dropped', 
                                 'docker_diskio_write_service_time', 'docker_diskio_write_bytes',
                                 'docker_diskio_read_service_time', 'docker_network_in_bytes',
                                 'docker_network_out_bytes', 'docker_diskio_read_bytes',
                                 'docker_network_in_errors', 'docker_network_out_errors',
                                 'docker_cpu_total_norm_pct', 'docker_memory_usage_pct']
                all_files_data[file_key] = {'metric_columns': default_columns, 'window_data': pd.DataFrame()}
        
        # 如果所有文件都没有数据，跳过该样本
        if not has_any_data:
            return (sample_idx, False, "所有文件在该时间窗口内都没有数据", None)
        
        # 计算统一的时间戳序列
        if not all_actual_timestamps:
            return (sample_idx, False, "没有找到有效时间戳", None)
        
        # 使用最早的时间戳计算理想起始点
        interval_ms = 30 * 1000
        reference_ts = min(all_actual_timestamps)
        
        # 如果第一个实际时间戳与故障开始时间差>=30秒，需要往前推
        if (reference_ts - start_timestamp_ms) >= interval_ms:
            steps_back = (reference_ts - start_timestamp_ms) // interval_ms
            ideal_first_timestamp = reference_ts - steps_back * interval_ms
        else:
            ideal_first_timestamp = reference_ts
        
        # 生成20个连续时间戳
        expected_timestamps = [ideal_first_timestamp + i * interval_ms for i in range(20)]
        
        # 第二轮：为每个文件生成完整数据
        result_data = {}
        
        for file_key, file_data in all_files_data.items():
            metric_columns = file_data['metric_columns']
            window_data = file_data['window_data']
            
            # 创建该文件的完整数据框架
            complete_data = []
            
            for expected_ts in expected_timestamps:
                # 查找该时间戳的数据
                if not window_data.empty:
                    exact_match = window_data[window_data['timestamp'] == expected_ts]
                else:
                    exact_match = pd.DataFrame()
                
                if not exact_match.empty:
                    # 如果找到精确匹配，使用该数据（只保留原始metric数据列）
                    row_data = exact_match.iloc[0][['timestamp'] + metric_columns].to_dict()
                else:
                    # 如果没有找到精确匹配，创建空值行
                    row_data = {'timestamp': float(expected_ts)}
                    # 对所有metric列设置为None
                    for col in metric_columns:
                        row_data[col] = None
                
                complete_data.append(row_data)
            
            # 转换为DataFrame
            complete_df = pd.DataFrame(complete_data)
            
            # 确保timestamp列在第一位
            cols = ['timestamp'] + [col for col in complete_df.columns if col != 'timestamp']
            complete_df = complete_df[cols]
            
            result_data[file_key] = complete_df
        
        return (sample_idx, True, None, result_data)
        
    except Exception as e:
        return (sample_idx, False, str(e), None)


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


def extract_anomaly_metric_data(metric_dir, anomaly_periods):
    """
    从processed_data/metric目录中的所有文件提取异常时间段的metric数据
    直接覆盖原始文件
    
    Args:
        metric_dir (str): metric文件目录
        anomaly_periods (list): 异常时间段列表
    """
    print("=== 开始提取异常metric数据（覆盖原文件）===")
    
    # 获取所有metric文件
    metric_files = [f for f in os.listdir(metric_dir) if f.endswith('.csv')]
    
    # 处理每个文件    
    for metric_file in metric_files:
        print(f"\n处理文件: {metric_file}")
        metric_file_path = os.path.join(metric_dir, metric_file)
        
        # 读取metric文件
        metric_df = pd.read_csv(metric_file_path)
        original_count = len(metric_df)
        print(f"  读取数据: {original_count:,} 条")
        
        # 检查是否有timestamp列
        if 'timestamp' not in metric_df.columns:
            print(f"  ⚠️ 跳过文件 {metric_file}: 缺少timestamp列")
            continue
        
        # 使用create_selection_mask函数创建异常数据掩码
        anomaly_mask = create_selection_mask(metric_df['timestamp'], anomaly_periods)
        
        # 提取异常数据
        anomaly_data = metric_df[anomaly_mask].copy()
        anomaly_count = len(anomaly_data)
        
        # 直接覆盖原文件
        anomaly_data.to_csv(metric_file_path, index=False)
        
        print(f"  提取异常数据: {anomaly_count:,} 条 ({anomaly_count/original_count:.2%})")
        print(f"  已覆盖原文件: {metric_file_path}")
    
    print(f"\n=== 异常metric数据提取完成 ===")


def extract_anomaly_samples(num_processes: int = None) -> None:
    """
    并行提取异常样本的metric数据
    根据标签文件中的故障开始时间戳，提取600秒时间窗口内的数据
    
    参数:
        num_processes: 使用的进程数，默认为CPU核心数
    
    返回值:
        无
    """
    # 文件路径
    label_file = os.path.join(project_dir, 'data', 'gaia', 'label.csv')
    processed_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anomaly_samples')
        
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    
    # 读取标签文件
    print("📖 读取标签文件...")
    labels_df = pd.read_csv(label_file)
    print(f"✅ 成功读取 {len(labels_df)} 个异常样本")
    
    # 构建metric文件映射
    print("🔍 构建metric文件映射...")
    metric_files_dict = {}
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.csv'):
            # 从文件名中提取服务实例名
            # 例如: dbservice1_0.0.0.4_2021-07-01_2021-07-31.csv -> dbservice1
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                instance_name = parts[0]  # 取服务实例名作为键，如 dbservice1, mobservice2
                metric_files_dict[instance_name] = os.path.join(processed_data_dir, filename)
    
    print(f"✅ 找到 {len(metric_files_dict)} 个metric文件:")
    for instance, path in metric_files_dict.items():
        print(f"   {instance}: {os.path.basename(path)}")
    
    # 准备并行处理的参数
    process_args = []
    for idx, row in labels_df.iterrows():
        process_args.append((idx, row, metric_files_dict))
    
    # 确定进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(process_args))
    
    print(f"\n🚀 开始提取 {len(process_args)} 个异常样本，使用 {num_processes} 个进程...")
    
    # 使用多进程池处理
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_anomaly_sample, process_args),
            total=len(process_args),
            desc="提取异常样本"
        ))
    
    # 处理结果并保存
    successful_samples = []
    failed_count = 0
    
    for sample_idx, success, error_msg, extracted_data in results:
        if success and extracted_data is not None:
            successful_samples.append(extracted_data)
        else:
            failed_count += 1
            if error_msg:
                print(f"❌ 样本 {sample_idx}: {error_msg}")
    
    # 合并所有成功提取的数据并保存
    if successful_samples:
        print(f"\n💾 保存提取的数据...")
        all_samples_df = pd.concat(successful_samples, ignore_index=True)
        
        # 保存完整数据集
        all_samples_path = os.path.join(output_dir, 'all_anomaly_samples.csv')
        all_samples_df.to_csv(all_samples_path, index=False)
        print(f"✅ 完整数据集已保存: {all_samples_path}")
        
        # 按服务分别保存
        for service in all_samples_df['service'].unique():
            service_data = all_samples_df[all_samples_df['service'] == service]
            service_path = os.path.join(output_dir, f'{service}_anomaly_samples.csv')
            service_data.to_csv(service_path, index=False)
            print(f"✅ {service} 数据已保存: {service_path} ({len(service_data)} 条记录)")
        
        # 统计信息
        print(f"\n📊 提取完成统计:")
        print(f"  成功提取的样本: {len(successful_samples)}")
        print(f"  失败的样本: {failed_count}")
        print(f"  总数据点: {len(all_samples_df):,} (每样本20个时间戳)")
        print(f"  时间窗口长度: 600秒 (30秒间隔)")
        print(f"  涵盖的服务: {', '.join(all_samples_df['service'].unique())}")
        print(f"  数据列数: {len(all_samples_df.columns)}")
        
        # 统计缺失值情况
        total_metric_values = 0
        missing_metric_values = 0
        metric_columns = [col for col in all_samples_df.columns 
                         if col not in ['timestamp', 'sample_idx', 'service', 'instance', 'anomaly_type', 'start_timestamp']]
        
        for col in metric_columns:
            total_values = len(all_samples_df)
            # 统计空值（None）和NaN值
            missing_values = all_samples_df[col].isna().sum() + (all_samples_df[col] == '').sum()
            total_metric_values += total_values
            missing_metric_values += missing_values
        
        completion_rate = ((total_metric_values - missing_metric_values) / total_metric_values * 100) if total_metric_values > 0 else 0
        print(f"  数据完整率: {completion_rate:.1f}% ({total_metric_values - missing_metric_values:,}/{total_metric_values:,})")
    else:
        print("❌ 没有成功提取任何数据样本")


def extract_anomaly_samples_all_files(num_processes: int = None) -> None:
    """
    并行提取异常样本的metric数据，从所有10个文件中提取数据
    只要有一个文件有数据就保留该样本，生成10个对应的新文件
    
    参数:
        num_processes: 使用的进程数，默认为CPU核心数
    
    返回值:
        无
    """
    # 文件路径
    label_file = os.path.join(project_dir, 'preprocess', 'raw_data', 'gaia', 'label_gaia.csv')
    processed_data_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'metric')
    output_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'anomaly_metric')
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    
    # 读取标签文件
    print("📖 读取标签文件...")
    labels_df = pd.read_csv(label_file)
    print(f"✅ 成功读取 {len(labels_df)} 个异常样本")
    
    # 构建所有metric文件映射
    print("🔍 构建所有metric文件映射...")
    metric_files_dict = {}
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.csv'):
            # 使用完整文件名作为键，便于后续按文件保存
            file_key = filename.replace('.csv', '')
            metric_files_dict[file_key] = os.path.join(processed_data_dir, filename)
    
    # 准备并行处理的参数
    process_args = []
    for idx, row in labels_df.iterrows():
        process_args.append((idx, row, metric_files_dict))
    
    # 确定进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(process_args))
    
    print(f"\n🚀 开始从所有文件中提取 {len(process_args)} 个异常样本，使用 {num_processes} 个进程...")
    
    # 使用多进程池处理
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_anomaly_sample_all_files, process_args),
            total=len(process_args),
            desc="提取异常样本"
        ))
    
    # 处理结果并按文件分别保存
    successful_samples = []
    failed_count = 0
    
    # 初始化每个文件的数据列表
    file_data_dict = {file_key: [] for file_key in metric_files_dict.keys()}
    
    for sample_idx, success, error_msg, all_files_data in results:
        if success and all_files_data is not None:
            successful_samples.append(sample_idx)
            # 将每个文件的数据添加到对应列表中
            for file_key, file_df in all_files_data.items():
                file_data_dict[file_key].append(file_df)
        else:
            failed_count += 1
            if error_msg:
                print(f"❌ 样本 {sample_idx}: {error_msg}")
    
    # 合并并保存每个文件的数据
    if successful_samples:
        print(f"\n💾 保存提取的数据到 anomaly_metric...")
        
        total_records = 0
        for file_key, data_list in file_data_dict.items():
            if data_list:  # 如果该文件有数据
                # 合并该文件的所有样本数据
                file_combined_df = pd.concat(data_list, ignore_index=True)
                
                # 保存到processed_data2目录
                output_filename = f"{file_key}_anomaly_samples.csv"
                output_path = os.path.join(output_dir, output_filename)
                file_combined_df.to_csv(output_path, index=False)
                
                total_records += len(file_combined_df)
                print(f"✅ {file_key}: 已保存 {len(file_combined_df)} 条记录")
            else:
                print(f"⚠️  {file_key}: 没有数据")
        
        # 统计信息
        print(f"\n📊 提取完成统计:")
        print(f"  成功提取的样本: {len(successful_samples)}")
        print(f"  失败的样本: {failed_count}")
        print(f"  总数据记录: {total_records:,}")
        print(f"  生成文件数: {len([f for f in file_data_dict.values() if f])}")
        print(f"  每样本时间戳数: 20个 (30秒间隔)")
        print(f"  时间窗口长度: 600秒")
        print(f"  输出目录: {output_dir}")
        
        # 计算数据完整率（基于第一个有数据的文件）
        sample_file_data = None
        for data_list in file_data_dict.values():
            if data_list:
                sample_file_data = pd.concat(data_list, ignore_index=True)
                break
        
        if sample_file_data is not None:
            metric_columns = [col for col in sample_file_data.columns 
                             if col not in ['timestamp', 'sample_idx', 'service', 'instance', 
                                           'anomaly_type', 'start_timestamp', 'file_source']]
            
            total_metric_values = len(sample_file_data) * len(metric_columns)
            missing_metric_values = 0
            for col in metric_columns:
                missing_values = sample_file_data[col].isna().sum()
                missing_metric_values += missing_values
            
            completion_rate = ((total_metric_values - missing_metric_values) / total_metric_values * 100) if total_metric_values > 0 else 0
            print(f"  数据完整率: {completion_rate:.1f}%")
    else:
        print("❌ 没有成功提取任何数据样本")


def remove_empty_samples_from_processed_data2() -> None:
    """
    从processed_data2中移除所有文件中都没有数据的样本
    只有当一个标签（样本）在所有10个文件中的所有metric数据都是空值时，才移除该样本
    
    返回值:
        无
    """
    processed_data2_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'processed_data2')
    
    if not os.path.exists(processed_data2_dir):
        print(f"❌ processed_data2 目录不存在: {processed_data2_dir}")
        return
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(processed_data2_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ processed_data2 目录中没有找到CSV文件")
        return
        
    print(f"🔍 开始清理 {len(csv_files)} 个文件中的空样本...")
    
    # 第一步：读取所有文件并确定样本数量
    all_files_data = {}
    sample_count = 0
    
    for filename in csv_files:
        file_path = os.path.join(processed_data2_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"⚠️  {filename}: 文件为空，跳过")
                continue
                
            all_files_data[filename] = df
            
            # 确定样本数量（所有文件应该有相同的样本数）
            current_sample_count = len(df) // 20 if len(df) % 20 == 0 else 0
            if sample_count == 0:
                sample_count = current_sample_count
            elif sample_count != current_sample_count:
                print(f"⚠️  {filename}: 样本数量不一致 ({current_sample_count} vs {sample_count})")
                
        except Exception as e:
            print(f"❌ 读取文件 {filename} 时出错: {e}")
            continue
    
    if not all_files_data:
        print("❌ 没有成功读取任何文件")
        return
    
    print(f"📊 检测到 {sample_count} 个样本，开始跨文件分析...")
    
    # 第二步：检查每个样本在所有文件中是否都为空
    samples_to_keep = []  # 存储要保留的样本索引
    
    for sample_idx in range(sample_count):
        has_any_data_across_files = False
        
        # 检查该样本在所有文件中是否至少有一个文件有数据
        for filename, df in all_files_data.items():
            sample_start = sample_idx * 20
            sample_end = min(sample_start + 20, len(df))
            sample_data = df.iloc[sample_start:sample_end]
            
            # 获取metric列（排除timestamp列）
            metric_columns = [col for col in df.columns if col != 'timestamp']
            
            # 检查该文件中的该样本是否有任何非空数据
            for col in metric_columns:
                non_null_values = sample_data[col].dropna()
                if len(non_null_values) > 0:
                    has_any_data_across_files = True
                    break
            
            if has_any_data_across_files:
                break
        
        if has_any_data_across_files:
            samples_to_keep.append(sample_idx)
    
    print(f"📊 分析结果: {len(samples_to_keep)}/{sample_count} 个样本有数据，将移除 {sample_count - len(samples_to_keep)} 个空样本")
    
    # 第三步：为每个文件重新生成数据，只保留有效样本
    total_removed_samples = sample_count - len(samples_to_keep)
    
    for filename, df in all_files_data.items():
        try:
            valid_samples = []
            
            for sample_idx in samples_to_keep:
                sample_start = sample_idx * 20
                sample_end = min(sample_start + 20, len(df))
                sample_data = df.iloc[sample_start:sample_end]
                valid_samples.append(sample_data)
            
            if valid_samples:
                # 合并保留的样本
                cleaned_df = pd.concat(valid_samples, ignore_index=True)
                
                # 保存清理后的数据
                file_path = os.path.join(processed_data2_dir, filename)
                cleaned_df.to_csv(file_path, index=False)
                
                final_sample_count = len(cleaned_df) // 20
                removed_count = sample_count - final_sample_count
                print(f"✅ {filename}: 原始 {sample_count} 样本 -> 保留 {final_sample_count} 样本 (移除 {removed_count} 样本)")
            else:
                # 所有样本都被移除，创建空文件
                empty_df = pd.DataFrame(columns=df.columns)
                file_path = os.path.join(processed_data2_dir, filename)
                empty_df.to_csv(file_path, index=False)
                print(f"🗑️  {filename}: 所有样本都被移除，文件已清空")
                
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {e}")
            continue
    
    print(f"\n📊 清理完成统计:")
    print(f"  原始样本总数: {sample_count}")
    print(f"  移除的空样本: {total_removed_samples}")
    print(f"  保留的有效样本: {len(samples_to_keep)}")
    print(f"  清理比例: {(total_removed_samples/sample_count*100):.1f}%" if sample_count > 0 else "  清理比例: 0%")


def keep_only_complete_samples_from_processed_data2() -> None:
    """
    从processed_data2中只保留所有文件中所有指标都没有空值的样本
    移除任何在任一文件中有任何指标为空值的样本，只保留完全完整的数据
    
    返回值:
        无
    """
    processed_data2_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'processed_data2')
    
    if not os.path.exists(processed_data2_dir):
        print(f"❌ processed_data2 目录不存在: {processed_data2_dir}")
        return
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(processed_data2_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ processed_data2 目录中没有找到CSV文件")
        return
        
    print(f"🔍 开始筛选 {len(csv_files)} 个文件中的完整样本...")
    
    # 第一步：读取所有文件并确定样本数量
    all_files_data = {}
    sample_count = 0
    
    for filename in csv_files:
        file_path = os.path.join(processed_data2_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"⚠️  {filename}: 文件为空，跳过")
                continue
                
            all_files_data[filename] = df
            
            # 确定样本数量（所有文件应该有相同的样本数）
            current_sample_count = len(df) // 20 if len(df) % 20 == 0 else 0
            if sample_count == 0:
                sample_count = current_sample_count
            elif sample_count != current_sample_count:
                print(f"⚠️  {filename}: 样本数量不一致 ({current_sample_count} vs {sample_count})")
                
        except Exception as e:
            print(f"❌ 读取文件 {filename} 时出错: {e}")
            continue
    
    if not all_files_data:
        print("❌ 没有成功读取任何文件")
        return
    
    print(f"📊 检测到 {sample_count} 个样本，开始跨文件完整性分析...")
    
    # 第二步：检查每个样本在所有文件中是否都没有空值
    samples_to_keep = []  # 存储要保留的样本索引
    
    for sample_idx in range(sample_count):
        is_complete_across_all_files = True
        
        # 检查该样本在所有文件中是否都完全没有空值
        for filename, df in all_files_data.items():
            sample_start = sample_idx * 20
            sample_end = min(sample_start + 20, len(df))
            sample_data = df.iloc[sample_start:sample_end]
            
            # 获取metric列（排除timestamp列）
            metric_columns = [col for col in df.columns if col != 'timestamp']
            
            # 检查该文件中的该样本是否有任何空值
            for col in metric_columns:
                null_count = sample_data[col].isna().sum()
                if null_count > 0:
                    # 发现空值，该样本不完整
                    is_complete_across_all_files = False
                    break
            
            if not is_complete_across_all_files:
                break
        
        if is_complete_across_all_files:
            samples_to_keep.append(sample_idx)
    
    print(f"📊 分析结果: {len(samples_to_keep)}/{sample_count} 个样本完全无空值，将移除 {sample_count - len(samples_to_keep)} 个不完整样本")
    
    # 第三步：为每个文件重新生成数据，只保留完整样本
    total_removed_samples = sample_count - len(samples_to_keep)
    
    for filename, df in all_files_data.items():
        try:
            complete_samples = []
            
            for sample_idx in samples_to_keep:
                sample_start = sample_idx * 20
                sample_end = min(sample_start + 20, len(df))
                sample_data = df.iloc[sample_start:sample_end]
                complete_samples.append(sample_data)
            
            if complete_samples:
                # 合并保留的完整样本
                cleaned_df = pd.concat(complete_samples, ignore_index=True)
                
                # 保存筛选后的数据
                file_path = os.path.join(processed_data2_dir, filename)
                cleaned_df.to_csv(file_path, index=False)
                
                final_sample_count = len(cleaned_df) // 20
                removed_count = sample_count - final_sample_count
                print(f"✅ {filename}: 原始 {sample_count} 样本 -> 保留 {final_sample_count} 完整样本 (移除 {removed_count} 不完整样本)")
            else:
                # 没有完整样本，创建空文件
                empty_df = pd.DataFrame(columns=df.columns)
                file_path = os.path.join(processed_data2_dir, filename)
                empty_df.to_csv(file_path, index=False)
                print(f"🗑️  {filename}: 没有完整样本，文件已清空")
                
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {e}")
            continue
    
    print(f"\n📊 完整性筛选完成统计:")
    print(f"  原始样本总数: {sample_count}")
    print(f"  移除的不完整样本: {total_removed_samples}")
    print(f"  保留的完整样本: {len(samples_to_keep)}")
    print(f"  筛选比例: {(total_removed_samples/sample_count*100):.1f}%" if sample_count > 0 else "  筛选比例: 0%")
    print(f"  数据完整率: 100.0% (所有保留样本都没有空值)")


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    label_file = os.path.join(project_dir, 'preprocess', 'raw_data', 'gaia', 'label_gaia.csv')
    metric_dir = os.path.join(project_dir, 'preprocess', 'processed_data', 'gaia', 'metric')
    
    # 复制选定metric文件到processed_data/metric目录下
    copy_valid_metric_files()

    # 合并不同时间段的同一指标文件
    merge_date_range_files()

    # 重采样30秒间隔的指标数据
    resample_metrics_30s_interval()

    # 合并相同服务实例的不同指标文件
    merge_metrics_by_service_instance()

    # 加载异常时间段
    anomaly_periods = load_anomaly_periods(label_file)
    
    # 提取异常时间段的metric数据（直接覆盖原文件）
    extract_anomaly_metric_data(metric_dir, anomaly_periods)
    

    """
    # 清理样本 (二选一)
    # remove_empty_samples_from_processed_data2()        # 宽松清理：只移除所有文件中都完全为空的样本
    # keep_only_complete_samples_from_processed_data2()   # 严格清理：只保留所有文件中都完全没有空值的样本
    """