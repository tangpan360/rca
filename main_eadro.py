import time
import random
import torch
import numpy as np
from config.exp_config import Config
from helper.logger import get_logger
from core.TVDiagEadro import TVDiagEadro
from helper.Result import Result
from process.DatasetProcess import DatasetProcess
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(config: Config, logger):
    processor = DatasetProcess(config, logger)
    train_data, val_data, aug_data, test_data = processor.process()
    return train_data, val_data, aug_data, test_data


def train_and_evaluate(config: Config, log_dir, exp_name):
    set_seed(config.seed)
    logger = get_logger(log_dir, exp_name)
    logger.info("="*50)
    logger.info("TVDiag with Eadro Encoders")
    logger.info("="*50)
    logger.info("Load dataset")
    train_data, val_data, aug_data, test_data = build_dataloader(config, logger)
    logger.info("Training...")
    model = TVDiagEadro(config, logger, log_dir)
    model.train(train_data, val_data, aug_data)
    
    # 测试4种情况：完整模态 + 3种单模态缺失
    test_scenarios = [
        (None, "完整模态"),
        (['metric'], "缺失Metric"),
        (['log'], "缺失Log"),
        (['trace'], "缺失Trace")
    ]
    
    logger.info("="*60)
    logger.info("开始多场景测试")
    logger.info("="*60)
    
    results = {}
    for missing_modalities, scenario_name in test_scenarios:
        logger.info(f"\n--- {scenario_name} ---")
        
        if missing_modalities is None:
            # 完整模态使用原始评估方法
            res = model.evaluate(test_data)
            rcl_res = {
                'HR@1': res.hr_1,
                'HR@2': res.hr_2, 
                'HR@3': res.hr_3,
                'HR@4': res.hr_4,
                'HR@5': res.hr_5,
                'MRR@3': res.mrr_3
            }
            fti_res = {
                'pre': res.pre,
                'rec': res.rec,
                'f1': res.f1
            }
            avg_time = getattr(res, 'avg_inference_time', 0.0)
        else:
            # 缺失模态使用新评估方法
            rcl_res, fti_res, avg_time = model.evaluate_with_missing_modalities(
                test_data, missing_modalities
            )
        
        results[scenario_name] = {
            'rcl': rcl_res,
            'fti': fti_res,
            'time': avg_time
        }
    
    # 汇总打印结果
    logger.info("="*60)
    logger.info("测试结果汇总")
    logger.info("="*60)
    logger.info(f"{'场景':<12} {'HR@1':<8} {'HR@3':<8} {'MRR@3':<8} {'F1':<8} {'时间(s)':<8}")
    logger.info("-" * 60)
    
    for scenario_name, result in results.items():
        rcl = result['rcl']
        fti = result['fti']
        time_cost = result['time']
        
        logger.info(f"{scenario_name:<12} {rcl['HR@1']:<8.3f} {rcl['HR@3']:<8.3f} "
                   f"{rcl['MRR@3']:<8.3f} {fti['f1']:<8.3f} {time_cost:<8.4f}")
    
    # 返回结果字典
    return results


if __name__ == '__main__':
    dataset = 'gaia'
    config = Config(dataset)
    train_and_evaluate(config, f'./logs/{dataset}', f'{dataset}_eadro')



