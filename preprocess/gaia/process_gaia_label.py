import os
import shutil

# 获取当前脚本所在目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
_project_root = os.path.dirname(os.path.dirname(_script_dir))

def process_gaia_label():
    """
    将 label_gaia.csv 从 raw_data 移动/复制到 processed_data
    """
    # 源文件路径
    source_path = os.path.join(_project_root, 'data', 'raw_data', 'gaia', 'label_gaia.csv')
    
    # 目标目录
    output_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia')
    
    # 目标文件路径
    target_path = os.path.join(output_dir, 'label_gaia.csv')

    # 创建目标目录
    os.makedirs(output_dir, exist_ok=True)
        
    # 复制文件
    shutil.copy2(source_path, target_path)

    print(f"Successfully copied label file to: {target_path}")

if __name__ == "__main__":
    process_gaia_label()
