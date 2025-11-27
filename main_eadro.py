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
    
    # 打印使用的模态配置
    if config.use_partial_modalities:
        logger.info(f"部分模态模式启用")
        logger.info(f"训练模态: {config.training_modalities}")
        logger.info(f"测试模态: {config.testing_modalities}")
    else:
        logger.info(f"使用全部模态: {config.modalities}")
    
    logger.info("Load dataset")
    train_data, val_data, aug_data, test_data = build_dataloader(config, logger)
    logger.info("Training...")
    model = TVDiagEadro(config, logger, log_dir)
    model.train(train_data, val_data, aug_data)
    res: Result = model.evaluate(test_data)
    return res.export_df(exp_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TVDiag with Eadro Encoders')
    parser.add_argument('--dataset', default='gaia', choices=['gaia', 'sn'],
                       help='Dataset to use: gaia or sn')
    parser.add_argument('--gpu', default='0', help='GPU device ID')
    parser.add_argument('--seed', default=2, type=int, help='Random seed')
    args = parser.parse_args()
    
    dataset = args.dataset
    config = Config(dataset)
    config.gpu_device = args.gpu
    config.seed = args.seed
    
    train_and_evaluate(config, f'./logs/{dataset}', f'{dataset}_eadro')

