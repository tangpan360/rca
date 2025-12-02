#!/usr/bin/env python3
"""
提取服务依赖图结构（nodes和edges）
- 从trace数据提取服务调用关系
- 从metric文件名提取服务-节点映射关系
- 生成nodes（服务列表）和edges（依赖关系）

支持两种提取模式：

1. Dynamic模式（默认）：
   - 每个故障案例单独提取edges
   - 基于该案例时间窗口内的实际trace调用
   - 更符合原始deployment_extractor.py的实现
   
2. Static模式：
   - 所有案例共享全局edges
   - 基于所有trace数据构建统一的服务拓扑
   - 适合固定架构的GNN训练场景

支持配置是否包含同节点影响边（influences）：
   - 默认不包含：edges = 仅trace调用关系（纯调用拓扑）
   - 可选包含：edges = trace调用关系 + 同节点影响

输出文件命名格式：
   - nodes_{mode}_with_influence.json   # 包含影响边
   - edges_{mode}_with_influence.json
   - nodes_{mode}_no_influence.json     # 不包含影响边
   - edges_{mode}_no_influence.json

使用方法：
    # 基本用法
    python process_gaia_graph.py                              # 默认: dynamic + 不包含influences
    python process_gaia_graph.py --mode dynamic               # 显式dynamic + 不包含influences
    python process_gaia_graph.py --mode static                # static + 不包含influences
    
    # 包含同节点影响边
    python process_gaia_graph.py --with-influences            # dynamic + 包含influences
    python process_gaia_graph.py --mode static --with-influences # static + 包含influences
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json

# 添加项目路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)


def add_parent_name_to_trace(trace_df):
    """
    通过左外连接添加parent_name列
    将parent_id与span_id进行匹配，获取父服务的service_name
    
    Args:
        trace_df (pd.DataFrame): 原始trace数据
        
    Returns:
        pd.DataFrame: 添加了parent_name列的trace数据
    """
    print("正在添加parent_name列...")
    
    # 创建一个只包含span_id和service_name的临时表
    span_service = trace_df[['span_id', 'service_name']].copy()
    span_service.rename(columns={'service_name': 'parent_name'}, inplace=True)
    
    # 左外连接：将parent_id与span_id进行匹配
    trace_df = trace_df.merge(
        span_service, 
        left_on='parent_id', 
        right_on='span_id', 
        how='left',
        suffixes=('', '_parent')
    )
    
    # 删除多余的span_id_parent列
    if 'span_id_parent' in trace_df.columns:
        trace_df.drop(columns=['span_id_parent'], inplace=True)
    
    # 统计有parent_name的比例
    has_parent = trace_df['parent_name'].notna().sum()
    total = len(trace_df)
    print(f"成功添加parent_name列，{has_parent:,}/{total:,} ({has_parent/total:.2%}) 条记录有父服务")
    
    return trace_df


def load_trace_data(trace_dir):
    """
    加载所有trace数据到一个DataFrame，并添加parent_name列
    
    Args:
        trace_dir (str): trace文件目录路径
        
    Returns:
        pd.DataFrame: 合并后的trace数据（包含parent_name列）
    """
    print(f"正在从 {trace_dir} 加载trace数据...")
    
    dfs = []
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    
    for trace_file in tqdm(trace_files, desc="读取trace文件"):
        trace_path = os.path.join(trace_dir, trace_file)
        df = pd.read_csv(trace_path)
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"在 {trace_dir} 中没有找到trace文件")
    
    # 合并所有trace数据
    trace_df = pd.concat(dfs, ignore_index=True)
    print(f"成功加载 {len(trace_df):,} 条trace记录")
    
    # 添加parent_name列
    trace_df = add_parent_name_to_trace(trace_df)
    
    return trace_df


def extract_service_node_mapping(metric_dir):
    """
    从metric目录的文件名中提取服务-节点映射关系
    文件名格式：服务名_节点名_指标名_时间.csv
    
    Args:
        metric_dir (str): metric文件目录路径
        
    Returns:
        dict: {节点名: [服务列表]}
    """
    print(f"正在从 {metric_dir} 提取服务-节点映射...")
    
    node2svcs = defaultdict(list)
    
    for filename in os.listdir(metric_dir):
        if not filename.endswith('.csv'):
            continue
            
        splits = filename.split('_')
        if len(splits) < 2:
            continue
            
        svc, host = splits[0], splits[1]
        
        # 过滤系统服务
        if svc in ['system', 'redis', 'zookeeper']:
            continue
            
        # 避免重复添加
        if svc not in node2svcs[host]:
            node2svcs[host].append(svc)
    
    total_services = sum(len(svcs) for svcs in node2svcs.values())
    print(f"发现 {len(node2svcs)} 个节点，共 {total_services} 个服务实例")
    
    return dict(node2svcs)


def build_same_node_influences(node2svcs):
    """
    构建同节点服务的双向影响关系
    同一节点上的服务两两之间建立双向边
    
    Args:
        node2svcs (dict): {节点名: [服务列表]}
        
    Returns:
        tuple: (服务列表, 同节点影响边列表)
    """
    svcs = []
    influences = []
    
    for node, pods in node2svcs.items():
        svcs.extend(pods)
        
        # 同节点服务两两建立双向边
        for i in range(len(pods)):
            for j in range(i + 1, len(pods)):
                influences.append([pods[i], pods[j]])
                influences.append([pods[j], pods[i]])
    
    # 去重并排序服务列表
    svcs = sorted(list(set(svcs)))
    
    print(f"构建了 {len(influences)} 条同节点影响边")
    
    return svcs, influences


def extract_trace_call_relations(trace_df):
    """
    从trace数据中提取服务调用关系
    parent_name -> service_name 的调用边
    
    Args:
        trace_df (pd.DataFrame): trace数据
        
    Returns:
        list: 调用关系列表 [[parent_name, service_name], ...]
    """
    # print("正在从trace数据提取服务调用关系...")
    
    # 选择有父服务的记录
    edge_columns = ['service_name', 'parent_name']
    calls = trace_df.dropna(subset=['parent_name']).drop_duplicates(
        subset=edge_columns
    )[edge_columns].values.tolist()
    
    # print(f"提取了 {len(calls)} 条服务调用关系")
    
    return calls


def build_graph_structure(svcs, influences, calls, include_influences=True):
    """
    构建图结构：合并同节点影响和调用关系，转换为索引表示
    
    Args:
        svcs (list): 服务列表
        influences (list): 同节点影响边
        calls (list): 服务调用边
        include_influences (bool): 是否包含同节点影响边（默认True）
        
    Returns:
        tuple: (节点列表, 边列表)
    """
    # print(f"正在构建图结构... (include_influences={include_influences})")
    
    # 根据参数决定是否合并influences
    if include_influences:
        all_edges = calls + influences
        # print(f"  包含 {len(calls)} 条调用边 + {len(influences)} 条同节点影响边")
    else:
        all_edges = calls
        # print(f"  仅包含 {len(calls)} 条调用边（跳过同节点影响边）")
    
    # 使用DataFrame去重
    if len(all_edges) > 0:
        all_edges = pd.DataFrame(all_edges).drop_duplicates().reset_index(drop=True).values.tolist()
    else:
        all_edges = []
    
    # print(f"  合并去重后共 {len(all_edges)} 条边")
    
    # 转换为索引表示
    edges = []
    for edge in all_edges:
        # edge[0] 是 service_name (target), edge[1] 是 parent_name (source)
        # 调用方向：parent -> service
        source, target = edge[1], edge[0]
        
        # 确保两个服务都在服务列表中
        if source not in svcs or target not in svcs:
            continue
            
        source_idx = svcs.index(source)
        target_idx = svcs.index(target)
        edges.append([source_idx, target_idx])
    
    # print(f"  生成了 {len(edges)} 条有效边（索引表示）")
    
    return svcs, edges


def extract_graph_for_all_cases(label_file, trace_dir, metric_dir, output_dir, 
                                mode='dynamic', include_influences=True):
    """
    为所有故障案例提取图结构
    
    Args:
        label_file (str): 标签文件路径
        trace_dir (str): trace数据目录
        metric_dir (str): metric数据目录
        output_dir (str): 输出目录
        mode (str): 提取模式
            - 'dynamic': 每个案例单独提取edges（默认，更符合原始方案）
            - 'static': 所有案例共享全局edges（适用于固定拓扑分析）
        include_influences (bool): 是否包含同节点影响边（默认True）
            - True: edges包含调用关系 + 同节点影响（与原始方案一致）
            - False: edges仅包含调用关系（纯trace拓扑）
    """
    print("=" * 60)
    print(f"开始提取所有故障案例的图结构（nodes和edges）")
    print(f"模式: {mode.upper()}")
    print(f"包含同节点影响边: {include_influences}")
    print("=" * 60)
    
    if mode not in ['dynamic', 'static']:
        raise ValueError(f"不支持的模式: {mode}，请使用 'dynamic' 或 'static'")
    
    # 1. 加载标签文件
    print(f"\n1. 加载标签文件: {label_file}")
    label_df = pd.read_csv(label_file, index_col=0)
    print(f"   共 {len(label_df)} 个故障案例")
    
    # 2. 提取服务-节点映射
    print(f"\n2. 提取服务-节点映射")
    node2svcs = extract_service_node_mapping(metric_dir)
    
    # 3. 构建同节点影响关系
    print(f"\n3. 构建同节点影响关系")
    svcs, influences = build_same_node_influences(node2svcs)
    
    nodes_dict = {}
    edges_dict = {}
    
    if mode == 'static':
        # 静态模式：所有案例共享全局edges
        print(f"\n4. [静态模式] 加载所有trace数据")
        trace_df = load_trace_data(trace_dir)
        
        print(f"\n5. [静态模式] 提取全局trace调用关系")
        calls = extract_trace_call_relations(trace_df)
        
        print(f"\n6. [静态模式] 构建全局图结构")
        nodes, edges = build_graph_structure(svcs, influences, calls, include_influences)
        
        print(f"\n7. [静态模式] 为所有案例分配相同的图结构")
        for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="处理故障案例"):
            nodes_dict[idx] = nodes
            edges_dict[idx] = edges
        
        # 统计信息
        unique_edges = len(edges)
        print(f"\n   全局edges数: {unique_edges}")
        
    else:  # mode == 'dynamic'
        # 动态模式：每个案例单独提取edges
        # 性能优化：一次性加载所有trace到内存，然后快速筛选（避免重复读取文件）
        print(f"\n4. [动态模式] 加载所有trace数据到内存")
        all_trace_df = load_trace_data(trace_dir)
        print(f"   成功加载，共 {len(all_trace_df):,} 条trace记录")
        
        print(f"\n5. [动态模式] 为每个案例单独提取edges")
        edge_stats = []  # 统计每个案例的edges数量
        
        for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="处理故障案例"):
            # 获取该案例的时间范围
            st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
            ed_time = st_time + 600 * 1000  # 600秒窗口
            
            # 从内存中筛选该案例的trace数据
            if 'start_time_ts' in all_trace_df.columns:
                mask = (all_trace_df['start_time_ts'] >= st_time) & (all_trace_df['start_time_ts'] <= ed_time)
                case_trace_df = all_trace_df[mask]
            else:
                case_trace_df = pd.DataFrame()
            
            if len(case_trace_df) == 0:
                # 如果没有trace数据，使用空的调用关系
                calls = []
            else:
                # 提取该案例的调用关系
                calls = extract_trace_call_relations(case_trace_df)
            
            # 构建该案例的图结构
            nodes, edges = build_graph_structure(svcs, influences, calls, include_influences)
            nodes_dict[idx] = nodes
            edges_dict[idx] = edges
            edge_stats.append(len(edges))
        
        # 统计信息
        import numpy as np
        print(f"\n   Edges统计:")
        print(f"     最小: {min(edge_stats)}")
        print(f"     最大: {max(edge_stats)}")
        print(f"     平均: {np.mean(edge_stats):.1f}")
        print(f"     中位数: {np.median(edge_stats):.1f}")
    
    # 8. 保存结果
    print(f"\n8. 保存结果")
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件名包含模式和influences信息
    influence_suffix = "_with_influence" if include_influences else "_no_influence"
    nodes_file = os.path.join(output_dir, f'nodes_{mode}{influence_suffix}.json')
    edges_file = os.path.join(output_dir, f'edges_{mode}{influence_suffix}.json')
    
    with open(nodes_file, 'w') as f:
        json.dump(nodes_dict, f)
    with open(edges_file, 'w') as f:
        json.dump(edges_dict, f)
    
    print(f"   节点数据已保存至: {nodes_file}")
    print(f"   边数据已保存至: {edges_file}")
    
    # 9. 统计信息
    print("\n" + "=" * 60)
    print("提取完成！统计信息：")
    print("=" * 60)
    print(f"模式: {mode.upper()}")
    print(f"故障案例数: {len(nodes_dict)}")
    print(f"服务节点数: {len(nodes_dict[list(nodes_dict.keys())[0]])}")
    if mode == 'static':
        print(f"依赖边数: {unique_edges} (所有案例相同)")
        print(f"图密度: {unique_edges / (len(nodes) * (len(nodes) - 1)):.4f}")
    else:
        print(f"依赖边数: 平均 {np.mean(edge_stats):.1f} (范围: {min(edge_stats)}-{max(edge_stats)})")
    print("=" * 60)
    
    return nodes_dict, edges_dict


def main():
    """
    主函数
    
    支持命令行参数：
        --mode: 提取模式，可选 'dynamic' 或 'static'（默认: dynamic）
        --with-influences: 包含同节点影响边（默认不包含）
        
    示例：
        python process_gaia_graph.py                              # 默认：dynamic + 不包含influences
        python process_gaia_graph.py --mode static                # 静态模式 + 不包含influences
        python process_gaia_graph.py --with-influences            # 动态模式 + 包含influences
        python process_gaia_graph.py --mode static --with-influences # 静态模式 + 包含influences
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='提取服务依赖图结构（nodes和edges）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明：
  dynamic: 动态模式（默认）
           - 每个故障案例单独提取edges
           - 更符合原始方案，体现每个案例实际的调用关系
           - edges数量在不同案例间可能有差异
           
  static:  静态模式
           - 所有案例共享全局edges
           - 基于所有trace数据构建统一的服务拓扑
           - 更适合固定架构的分析场景

影响边说明：
  默认情况下不包含同节点影响边（influences），只保留trace调用关系。
  使用 --with-influences 可以包含同节点影响边，即同一物理节点上的服务之间的双向边。
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='static',
        choices=['dynamic', 'static'],
        help='提取模式：dynamic（每个案例单独edges）或 static（全局共享edges）'
    )
    parser.add_argument(
        '--with-influences',
        action='store_true',
        help='包含同节点影响边（默认不包含）'
    )
    
    args = parser.parse_args()
    
    # 转换为 include_influences 参数
    include_influences = args.with_influences
    
    # 输入路径
    label_file = os.path.join(_project_root, "data", "processed_data", "gaia", "label_gaia.csv")
    trace_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "trace")
    metric_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "metric")
    
    # 输出路径
    output_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "graph")
    
    # 执行提取
    print(f"\n配置:")
    print(f"  模式: {args.mode}")
    print(f"  包含同节点影响边: {include_influences}")
    extract_graph_for_all_cases(label_file, trace_dir, metric_dir, output_dir, 
                                mode=args.mode, include_influences=include_influences)


if __name__ == "__main__":
    main()
