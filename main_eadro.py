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
    res: Result = model.evaluate(test_data)
    return res.export_df(exp_name)


if __name__ == '__main__':
    dataset = 'gaia'
    config = Config(dataset)
    train_and_evaluate(config, f'./logs/{dataset}', f'{dataset}_eadro')



