#!/usr/bin/env python3
"""
数据集时区转换工具（支持SN和TT数据集）
将所有时间戳统一转换为UTC时区

转换内容：
1. 文件夹名：减16小时
2. JSON文件名：减16小时
3. JSON标签文件：不变（已经是UTC）
4. logs.json：时间字符串减8小时
5. spans.json：startTime减8小时并转为秒，duration转为秒
6. metrics：不变（已经是UTC）

支持的数据集：SN（Social Network）、TT（Train Ticket）
"""

import json
import shutil
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time

# 模块级私有变量：自动获取脚本和项目路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

class TimezoneConverter:
    def __init__(self, base_dir: str = None, dataset_type: str = None):
        """
        初始化转换器
        
        Args:
            base_dir: 项目根目录，默认自动识别
            dataset_type: 数据集类型 ('sn', 'tt', 或 None表示全部)
        """
        # 使用传入的base_dir或默认使用项目根目录
        if base_dir is None:
            base_dir = _project_root
        
        self.base_dir = Path(base_dir)
        self.dataset_type = dataset_type
        self.raw_data_dir = self.base_dir / "data" / "raw_data"
        self.backup_dir = self.base_dir / "data" / "raw_data_backup"
    
    def backup_data(self) -> bool:
        """创建备份（仅备份需要处理的数据集：sn 和 tt）"""
        print("检查备份...")
        
        if self.backup_dir.exists():
            # 读取备份时间
            timestamp_file = self.backup_dir / "backup_time.txt"
            if timestamp_file.exists():
                with open(timestamp_file, 'r') as f:
                    backup_info = f.read()
                print(f"✓ 备份已存在:")
                print(f"  {backup_info.strip()}")
                print("  跳过备份（保留现有备份）")
            else:
                print("⚠ 备份目录已存在但无时间戳文件")
            return True
        
        print("开始创建备份（仅备份 SN 和 TT 数据集）...")
        try:
            start_time = time.time()
            
            # 创建备份目录
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 只备份 sn 和 tt 目录（不备份 gaia）
            dataset_types = ['sn', 'tt'] if self.dataset_type is None else [self.dataset_type]
            
            for dt in dataset_types:
                src_dir = self.raw_data_dir / dt
                dst_dir = self.backup_dir / dt
                
                if src_dir.exists():
                    print(f"  备份 {dt.upper()} 数据集...")
                    shutil.copytree(src_dir, dst_dir)
                else:
                    print(f"  ⚠ {dt.upper()} 目录不存在，跳过")
            
            # 创建备份时间戳文件
            timestamp_file = self.backup_dir / "backup_time.txt"
            with open(timestamp_file, 'w') as f:
                f.write(f"备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"原始路径: {self.raw_data_dir}\n")
                datasets = ', '.join(dataset_types)
                f.write(f"数据集: {datasets}\n")
                f.write(f"注: 未备份 gaia 数据集（不进行修改）\n")
            
            elapsed = time.time() - start_time
            print(f"✓ 备份完成！耗时: {elapsed:.1f}秒")
            return True
        except Exception as e:
            print(f"✗ 备份失败: {e}")
            return False
    
    def convert_folder_name(self, folder_name: str) -> str:
        """转换文件夹名（减16小时）"""
        # 解析时间：支持 SN.xxx 或 TT.xxx 格式
        pattern = r'((?:SN|TT)\.)(\d{4}-\d{2}-\d{2}T\d{6})D(\d{4}-\d{2}-\d{2}T\d{6})'
        match = re.match(pattern, folder_name)
        
        if not match:
            return folder_name
        
        prefix, start_str, end_str = match.groups()
        
        # 转换起始时间
        start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H%M%S")
        start_utc = start_dt - timedelta(hours=16)
        
        # 转换结束时间
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H%M%S")
        end_utc = end_dt - timedelta(hours=16)
        
        new_name = f"{prefix}{start_utc.strftime('%Y-%m-%dT%H%M%S')}D{end_utc.strftime('%Y-%m-%dT%H%M%S')}"
        return new_name
    
    def convert_logs_json(self, file_path: Path) -> int:
        """转换logs.json中的时间字符串（减8小时）
        
        支持两种格式：
        1. SN格式: [2022-Apr-17 10:12:50.490796]
        2. TT格式: 2022-04-17 13:22:01.494
        """
        with open(file_path, 'r') as f:
            logs_data = json.load(f)
        
        # SN格式的月份映射
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month_map_reverse = {v: k for k, v in month_map.items()}
        
        # 两种日志格式的正则表达式
        pattern_sn = r'\[(\d{4})-(\w+)-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)\]'  # SN格式
        pattern_tt = r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)'    # TT格式
        
        def convert_sn_format(match):
            """转换SN格式: [2022-Apr-17 10:12:50.490796]"""
            year, month_str, day, hour, minute, second, microsec = match.groups()
            
            dt = datetime(
                int(year), 
                month_map[month_str], 
                int(day), 
                int(hour), 
                int(minute), 
                int(second), 
                int(microsec)
            )
            dt_utc = dt - timedelta(hours=8)
            
            new_month = month_map_reverse[dt_utc.month]
            return f"[{dt_utc.year}-{new_month}-{dt_utc.day:02d} {dt_utc.hour:02d}:{dt_utc.minute:02d}:{dt_utc.second:02d}.{microsec}]"
        
        def convert_tt_format(match):
            """转换TT格式: 2022-04-17 13:22:01.494"""
            year, month, day, hour, minute, second, millisec = match.groups()
            
            dt = datetime(
                int(year), 
                int(month), 
                int(day), 
                int(hour), 
                int(minute), 
                int(second), 
                int(millisec) * 1000  # 毫秒转微秒
            )
            dt_utc = dt - timedelta(hours=8)
            
            # TT格式保持原格式（年月日用数字，毫秒精度）
            return f"{dt_utc.year:04d}-{dt_utc.month:02d}-{dt_utc.day:02d} {dt_utc.hour:02d}:{dt_utc.minute:02d}:{dt_utc.second:02d}.{millisec}"
        
        # 转换所有日志
        total_logs = 0
        for service in logs_data:
            converted_logs = []
            for log in logs_data[service]:
                # 先尝试SN格式
                new_log = re.sub(pattern_sn, convert_sn_format, log)
                # 如果SN格式没匹配（log没变），再尝试TT格式
                if new_log == log:
                    new_log = re.sub(pattern_tt, convert_tt_format, log)
                converted_logs.append(new_log)
            
            logs_data[service] = converted_logs
            total_logs += len(converted_logs)
        
        # 写回文件
        with open(file_path, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        return total_logs
    
    def convert_spans_json(self, file_path: Path) -> Tuple[int, int]:
        """转换spans.json中的时间戳（减8小时并转为秒）"""
        print(f"  正在读取 {file_path.name}...")
        with open(file_path, 'r') as f:
            spans_data = json.load(f)
        
        total_spans = 0
        total_traces = len(spans_data)
        
        # 转换每个span的startTime和duration
        for trace in spans_data:
            for span in trace.get('spans', []):
                # startTime: 减8小时并转为秒
                if 'startTime' in span:
                    # 原值是微秒，先减8小时（28800000000微秒），再转为秒
                    span['startTime'] = (span['startTime'] - 28800000000) / 1000000.0
                
                # duration: 只转单位为秒（不减时间）
                if 'duration' in span:
                    span['duration'] = span['duration'] / 1000000.0
                
                total_spans += 1
        
        print(f"  正在写入 {file_path.name}...")
        
        # 写回文件
        with open(file_path, 'w') as f:
            json.dump(spans_data, f, indent=2)
        
        return total_traces, total_spans
    
    def convert_dataset(self, dataset_type: str, category: str, dataset_name: str) -> Dict:
        """转换单个数据集"""
        print("=" * 80)
        print(f"开始转换数据集: [{dataset_type}/{category}] {dataset_name}")
        print("=" * 80)
        
        data_dir = self.raw_data_dir / dataset_type / category
        dataset_dir = data_dir / dataset_name
        
        # 根据数据集类型确定前缀
        prefix = dataset_name.split('.')[0]  # SN 或 TT
        fault_file = data_dir / f"{dataset_name.replace(f'{prefix}.', f'{prefix}.fault-')}.json"
        
        if not dataset_dir.exists():
            print(f"✗ 数据集目录不存在: {dataset_dir}")
            return {"success": False}
        
        start_time = time.time()
        stats = {
            "success": True,
            "dataset": dataset_name,
            "category": category,
            "logs_count": 0,
            "traces_count": 0,
            "spans_count": 0,
            "elapsed_time": 0
        }
        
        try:
            # 步骤1: 转换文件夹名和JSON文件名（最简单的操作）
            new_dataset_name = self.convert_folder_name(dataset_name)
            if new_dataset_name != dataset_name:
                print(f"步骤1: 重命名文件夹和JSON文件")
                print(f"  {dataset_name}")
                print(f"  → {new_dataset_name}")
                
                # 重命名文件夹
                new_dataset_dir = data_dir / new_dataset_name
                dataset_dir.rename(new_dataset_dir)
                dataset_dir = new_dataset_dir  # 更新路径
                
                # 重命名JSON文件
                new_fault_file = data_dir / f"{new_dataset_name.replace(f'{prefix}.', f'{prefix}.fault-')}.json"
                if fault_file.exists():
                    fault_file.rename(new_fault_file)
                
                stats["new_name"] = new_dataset_name
                print(f"  ✓ 重命名完成")
            
            # 步骤2: 转换logs.json（中等复杂度）
            logs_file = dataset_dir / "logs.json"
            if logs_file.exists():
                print("步骤2: 转换 logs.json...")
                log_start = time.time()
                stats["logs_count"] = self.convert_logs_json(logs_file)
                print(f"  ✓ 完成! 处理了 {stats['logs_count']} 条日志 ({time.time()-log_start:.1f}秒)")
            
            # 步骤3: 转换spans.json（最复杂的操作）
            spans_file = dataset_dir / "spans.json"
            if spans_file.exists():
                print("步骤3: 转换 spans.json...")
                span_start = time.time()
                stats["traces_count"], stats["spans_count"] = self.convert_spans_json(spans_file)
                print(f"  ✓ 完成! 处理了 {stats['traces_count']} 个traces, {stats['spans_count']} 个spans ({time.time()-span_start:.1f}秒)")
            
            stats["elapsed_time"] = time.time() - start_time
            print(f"✓ 数据集转换完成! 总耗时: {stats['elapsed_time']:.1f}秒")
            
        except Exception as e:
            print(f"✗ 转换失败: {e}")
            import traceback
            print(traceback.format_exc())
            stats["success"] = False
            stats["error"] = str(e)
        
        return stats
    
    def convert_all(self) -> List[Dict]:
        """转换所有数据集"""
        print("=" * 80)
        print("开始批量转换数据集")
        print("=" * 80)
        
        all_stats = []
        
        # 确定要处理的数据集类型
        if self.dataset_type:
            dataset_types = [self.dataset_type]
        else:
            # 自动检测存在的数据集类型
            dataset_types = []
            for dt in ['sn', 'tt']:
                if (self.raw_data_dir / dt).exists():
                    dataset_types.append(dt)
        
        print(f"处理数据集类型: {', '.join(dataset_types).upper()}")
        
        for dataset_type in dataset_types:
            print(f"\n{'='*80}")
            print(f"数据集: {dataset_type.upper()}")
            print('='*80)
            
            type_dir = self.raw_data_dir / dataset_type
            if not type_dir.exists():
                print(f"⚠ 目录不存在: {type_dir}")
                continue
            
            for category in ["data", "no fault"]:
                category_dir = type_dir / category
                if not category_dir.exists():
                    continue
                
                # 获取所有数据集文件夹（SN.xxx 或 TT.xxx）
                prefix = dataset_type.upper()
                datasets = [d.name for d in category_dir.iterdir() 
                           if d.is_dir() and d.name.startswith(f"{prefix}.")]
                
                if not datasets:
                    continue
                
                print(f"\n在 [{dataset_type}/{category}] 中找到 {len(datasets)} 个数据集")
                
                for dataset_name in sorted(datasets):
                    stats = self.convert_dataset(dataset_type, category, dataset_name)
                    stats["dataset_type"] = dataset_type
                    all_stats.append(stats)
        
        return all_stats


def main():
    """主函数"""
    import sys
    
    # 自动获取项目根目录，处理所有数据集（sn 和 tt）
    converter = TimezoneConverter()
    
    print("=" * 80)
    print("数据集时区转换工具（SN + TT）")
    print("=" * 80)
    print(f"项目目录: {converter.base_dir}")
    print(f"数据目录: {converter.raw_data_dir}")
    print(f"备份目录: {converter.backup_dir}")
    print()
    
    # 步骤1: 备份
    print("步骤 1/2: 创建备份")
    if not converter.backup_data():
        print("\n备份失败，终止转换！")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    input("备份完成！按Enter继续转换，或Ctrl+C取消...")
    print()
    
    # 步骤2: 转换
    print("步骤 2/2: 转换数据")
    all_stats = converter.convert_all()
    
    # 显示总结
    print("\n" + "=" * 80)
    print("转换总结")
    print("=" * 80)
    
    success_count = sum(1 for s in all_stats if s.get("success"))
    total_logs = sum(s.get("logs_count", 0) for s in all_stats)
    total_spans = sum(s.get("spans_count", 0) for s in all_stats)
    
    # 按数据集类型分组统计
    sn_count = sum(1 for s in all_stats if s.get("dataset_type") == "sn")
    tt_count = sum(1 for s in all_stats if s.get("dataset_type") == "tt")
    
    print(f"成功转换: {success_count}/{len(all_stats)} 个数据集")
    if sn_count > 0:
        print(f"  - SN数据集: {sn_count} 个")
    if tt_count > 0:
        print(f"  - TT数据集: {tt_count} 个")
    print(f"总日志数: {total_logs:,}")
    print(f"总Span数: {total_spans:,}")
    
    if success_count == len(all_stats):
        print("\n✓ 所有数据集转换成功！")
    else:
        print("\n⚠ 部分数据集转换失败，请查看日志")


if __name__ == "__main__":
    main()
