import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def process_sn_traces():
    print("=== 开始处理 SN Trace 数据 ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    raw_data_dir = os.path.join(project_root, "preprocess/raw_data/sn/data")
    output_dir = os.path.join(project_root, "preprocess/processed_data/sn/trace")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 收集所有实验文件夹
    exp_folders = sorted([f for f in os.listdir(raw_data_dir) if f.startswith("SN.") and os.path.isdir(os.path.join(raw_data_dir, f))])
    
    # 暂存所有 Span 数据： {service_name: [span_records]}
    # 考虑到内存，我们可能需要分批处理，但这里先尝试一次性处理，如果内存不足再优化
    service_spans = {} 
    
    print(f"正在处理 {len(exp_folders)} 个实验文件夹中的 Trace...")
    
    for folder in tqdm(exp_folders, desc="解析 Trace 文件"):
        spans_json_path = os.path.join(raw_data_dir, folder, "spans.json")
        if not os.path.exists(spans_json_path):
            continue
            
        try:
            with open(spans_json_path, 'r') as f:
                traces_data = json.load(f)
        except Exception as e:
            print(f"❌ 读取 {spans_json_path} 失败: {e}")
            continue
            
        # traces_data 是一个 list，每个元素是一个 trace 对象
        # trace 对象包含 traceID, spans, processes 等
        
        for trace_obj in traces_data:
            trace_id = trace_obj.get('traceID', '')
            processes = trace_obj.get('processes', {})
            spans = trace_obj.get('spans', [])
            
            # 建立 processID -> serviceName 映射
            pid_to_service = {}
            for pid, p_info in processes.items():
                s_name = p_info.get('serviceName', 'unknown')
                # 处理前缀
                if s_name.startswith("socialnetwork-"):
                    s_name = s_name.replace("socialnetwork-", "")
                if s_name.endswith("-1"):
                    s_name = s_name[:-2]

                pid_to_service[pid] = s_name
                
            for span in spans:
                # 获取基础信息
                pid = span.get('processID')
                service_name = pid_to_service.get(pid, 'unknown')
                
                start_time_us = span.get('startTime') # 微秒
                duration_us = span.get('duration') # 微秒
                
                if start_time_us is None or duration_us is None:
                    continue
                    
                # 时间戳转换: 微秒 -> 秒，不减去 8 小时 (根据用户指令)
                # Eadro 原始脚本使用了秒 (startTime // 1_000_000)
                # 1 s = 1,000,000 us
                start_time_ts = start_time_us / 1_000_000.0
                duration_ts = duration_us / 1_000_000.0 # 也转换为秒
                
                # 获取 Status Code
                # 检查 tags
                status_code = 200 # 默认成功
                tags = span.get('tags', [])
                error_tag = False
                http_status = None
                
                for tag in tags:
                    key = tag.get('key', '')
                    val = tag.get('value', '')
                    
                    if key == 'error' and val is True:
                        error_tag = True
                    if key == 'http.status_code':
                        try:
                            http_status = int(val)
                        except:
                            pass
                            
                if http_status is not None:
                    status_code = http_status
                elif error_tag:
                    status_code = 500
                    
                # 获取 spanID 和 references (parentID)
                span_id = span.get('spanID', '')
                parent_id = ''
                references = span.get('references', [])
                for ref in references:
                    # 只处理 CHILD_OF 关系，其他关系（如 FOLLOWS_FROM）不作为父子关系
                    if ref.get('refType') == 'CHILD_OF':
                        parent_id = ref.get('spanID', '')
                        break
                # 如果没有 CHILD_OF 引用，parent_id 保持为空（根节点）

                # 存储
                if service_name not in service_spans:
                    service_spans[service_name] = []
                    
                service_spans[service_name].append({
                    'start_time_ts': start_time_ts,
                    'duration': duration_ts,
                    'trace_id': trace_id,
                    'span_id': span_id,
                    'parent_id': parent_id,
                    'status_code': status_code
                })
                
    # 保存为 CSV
    print("正在保存 Trace CSV 文件...")
    for service, records in tqdm(service_spans.items(), desc="Saving CSVs"):
        if not records:
            continue
            
        df = pd.DataFrame(records)
        # 排序
        df = df.sort_values('start_time_ts')
        
        save_path = os.path.join(output_dir, f"{service}_trace.csv")
        df.to_csv(save_path, index=False)
        
    print("=== SN Trace 处理完成 ===")

if __name__ == "__main__":
    process_sn_traces()

